import numpy as np
import os
import nd2reader
from PIL import Image
import cv2

def process_image(array: np.ndarray) -> np.ndarray:
    """
    画像処理関数：正規化とスケーリングを行う。
    """
    array = array.astype(np.float32)  # Convert to float
    array -= array.min()  # Normalize to 0
    array /= array.max()  # Normalize to 1
    array *= 255  # Scale to 0-255
    return array.astype(np.uint8)

def extract_nd2(file_name: str):
    """
    タイムラプスnd2ファイルをフレームごとにTIFF形式で保存する。
    チャンネル0: ph
    チャンネル1: fluo1
    チャンネル2: fluo2
    """
    base_output_dir = os.path.join(
        "experimental", "TLengine2.0", "output", file_name.split("/")[-1].split(".")[0]
    )
    os.makedirs(base_output_dir, exist_ok=True)

    with nd2reader.ND2Reader(file_name) as images:
        print(f"Available axes: {images.axes}")
        print(f"Sizes: {images.sizes}")

        # (channel, y, x) の順に束ねる
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        # フィールド(視野)ごとに処理
        images.iter_axes = "v"

        num_fields = images.sizes.get("v", 1)
        num_channels = images.sizes.get("c", 1)
        num_timepoints = images.sizes.get("t", 1)

        for field_idx in range(num_fields):
            # Fieldごとのフォルダを作成
            field_folder = os.path.join(base_output_dir, f"Field_{field_idx + 1}")
            os.makedirs(field_folder, exist_ok=True)

            # チャンネルごとのサブフォルダを作成
            ph_folder = os.path.join(field_folder, "ph")
            fluo1_folder = os.path.join(field_folder, "fluo1")
            fluo2_folder = os.path.join(field_folder, "fluo2")
            os.makedirs(ph_folder, exist_ok=True)
            os.makedirs(fluo1_folder, exist_ok=True)
            os.makedirs(fluo2_folder, exist_ok=True)

            for time_idx in range(num_timepoints):
                images.default_coords.update({"v": field_idx, "t": time_idx})
                image_data = images[0]

                if len(image_data.shape) == 2:
                    # チャンネルが1つのみの場合
                    ph_image = process_image(image_data)
                    tiff_filename = os.path.join(
                        ph_folder, f"time_{time_idx + 1}_channel_0.tif"
                    )
                    Image.fromarray(ph_image).save(tiff_filename)
                    print(f"Saved: {tiff_filename}")
                else:
                    # 複数チャンネルの場合
                    for i in range(image_data.shape[0]):
                        channel_image = process_image(image_data[i])
                        if i == 0:
                            folder = ph_folder
                            channel_name = "ph"
                        elif i == 1:
                            folder = fluo1_folder
                            channel_name = "fluo1"
                        elif i == 2:
                            folder = fluo2_folder
                            channel_name = "fluo2"
                        else:
                            folder = field_folder
                            channel_name = f"channel_{i}"

                        tiff_filename = os.path.join(
                            folder, f"time_{time_idx + 1}_{channel_name}.tif"
                        )
                        Image.fromarray(channel_image).save(tiff_filename)
                        print(f"Saved: {tiff_filename}")

def get_channel_image(
    file_name: str, field_idx: int, time_idx: int, channel_idx: int
) -> np.ndarray:
    """
    指定したファイル・Field・Timeから channel_idx (0=ph,1=fluo1,2=fluo2など)
    の画像を取得し、process_imageを通して正規化した numpy.ndarray を返す。
    """
    with nd2reader.ND2Reader(file_name) as images:
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"
        images.default_coords.update({"v": field_idx, "t": time_idx})

        image_data = images[0]  # shape: (channels, y, x) or (y, x)

        # チャンネル軸が存在するかを確認
        if len(image_data.shape) == 3:
            # channel軸がある場合は channel_idx を取り出す
            channel_image_data = image_data[channel_idx]
        else:
            # channelが1つしかない場合はそれがそのまま画像
            channel_image_data = image_data

        return process_image(channel_image_data)

def shift_image(img: np.ndarray, vertical: int, horizontal: int) -> np.ndarray:
    """
    画像を縦(vertical), 横(horizontal)ピクセル数だけずらす。
    符号によって上下左右を決める。
    """
    M = np.float32([
        [1, 0, horizontal],  # x方向シフト
        [0, 1, vertical]     # y方向シフト
    ])
    shifted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted_img

def calc_shift_by_phase_correlation(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[int, int]:
    """
    phaseCorrelateを使ってref_imgに対するtarget_imgのx,yズレを推定する。
    返すのは (vertical_shift, horizontal_shift) のタプル。
    """
    ref_float = ref_img.astype(np.float32)
    tgt_float = target_img.astype(np.float32)
    # DC成分(平均)を除去
    ref_float -= np.mean(ref_float)
    tgt_float -= np.mean(tgt_float)

    (shift_x, shift_y), _ = cv2.phaseCorrelate(ref_float, tgt_float)

    # phaseCorrelateの戻り値は "targetを(shift_x, shift_y)動かせばrefに重なる"
    # shift_imageの引数: shift_image(img, vertical, horizontal) の順
    vertical_shift = -int(round(shift_y))
    horizontal_shift = -int(round(shift_x))
    return (vertical_shift, horizontal_shift)

def align_all_channels_and_create_gifs(
    file_name: str,
    field_idx: int,
    output_prefix: str,
    use_reference_first_frame: bool = True,
    duration_ms: int = 300
):
    """
    ph, fluo1, fluo2 の３つのチャネルについて、
    phを基準に位置合わせしたシフト量を fluo1, fluo2 にも適用し、
    各チャネルのGIFを出力する。
    
    Parameters
    ----------
    file_name : str
        nd2ファイルパス
    field_idx : int
        field(視野)のindex
    output_prefix : str
        GIFファイル名の出力パスのプレフィックス (例: "aligned_field0_")
    use_reference_first_frame : bool
        Trueなら最初のフレームを常に基準にして他フレームを合わせる。
        Falseなら一つ前のフレームに対して逐次合わせる（累積的）
    duration_ms : int
        GIFの1フレームあたりの表示時間(ミリ秒)
    """

    with nd2reader.ND2Reader(file_name) as images:
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"
        num_timepoints = images.sizes.get("t", 1)
        # チャンネル数を一応確認（3つ以上ある可能性を考慮）
        num_channels = images.sizes.get("c", 1)
    
    # 全フレームの ph, fluo1, fluo2 画像をまとめて取得
    # ※ fluo1, fluo2 が存在しないケースを考慮して min(1,2,num_channels-1) などチェックしてもよい
    ph_frames = [get_channel_image(file_name, field_idx, t, 0) for t in range(num_timepoints)]
    fluo1_frames = [get_channel_image(file_name, field_idx, t, 1) for t in range(num_timepoints)] if num_channels > 1 else []
    fluo2_frames = [get_channel_image(file_name, field_idx, t, 2) for t in range(num_timepoints)] if num_channels > 2 else []

    # アライン後の画像をチャネルごとに格納
    aligned_ph_images: list[Image.Image] = []
    aligned_fluo1_images: list[Image.Image] = []
    aligned_fluo2_images: list[Image.Image] = []

    # 最初のフレームを基準
    ref_img_ph = ph_frames[0]
    aligned_ph_images.append(Image.fromarray(ref_img_ph))
    if fluo1_frames: aligned_fluo1_images.append(Image.fromarray(fluo1_frames[0]))
    if fluo2_frames: aligned_fluo2_images.append(Image.fromarray(fluo2_frames[0]))

    prev_img_ph = ref_img_ph  # 累積モード用

    for t in range(1, num_timepoints):
        target_ph = ph_frames[t]

        # ph同士でシフト量を計算
        if use_reference_first_frame:
            v_shift, h_shift = calc_shift_by_phase_correlation(ref_img_ph, target_ph)
        else:
            v_shift, h_shift = calc_shift_by_phase_correlation(prev_img_ph, target_ph)

        # phをシフト
        shifted_ph = shift_image(target_ph, v_shift, h_shift)
        aligned_ph_images.append(Image.fromarray(shifted_ph))
        if not use_reference_first_frame:
            prev_img_ph = shifted_ph

        # 同じシフト量を fluo1, fluo2 にも適用
        if fluo1_frames:
            shifted_fluo1 = shift_image(fluo1_frames[t], v_shift, h_shift)
            aligned_fluo1_images.append(Image.fromarray(shifted_fluo1))

        if fluo2_frames:
            shifted_fluo2 = shift_image(fluo2_frames[t], v_shift, h_shift)
            aligned_fluo2_images.append(Image.fromarray(shifted_fluo2))

    # それぞれGIFとして保存
    # ph
    ph_gif_path = f"{output_prefix}ph.gif"
    aligned_ph_images[0].save(
        ph_gif_path,
        save_all=True,
        append_images=aligned_ph_images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"ph GIF saved to: {ph_gif_path}")

    # fluo1 (チャンネルが存在する場合のみ)
    if fluo1_frames:
        fluo1_gif_path = f"{output_prefix}fluo1.gif"
        aligned_fluo1_images[0].save(
            fluo1_gif_path,
            save_all=True,
            append_images=aligned_fluo1_images[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"fluo1 GIF saved to: {fluo1_gif_path}")

    # fluo2 (チャンネルが存在する場合のみ)
    if fluo2_frames:
        fluo2_gif_path = f"{output_prefix}fluo2.gif"
        aligned_fluo2_images[0].save(
            fluo2_gif_path,
            save_all=True,
            append_images=aligned_fluo2_images[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"fluo2 GIF saved to: {fluo2_gif_path}")


# =============================================================================
# 動作テスト (必要に応じて実行)
# =============================================================================
if __name__ == "__main__":
    filename = "experimental/TLengine2.0/sk450gen120min-tl.nd2"
    test_field = 0
    output_prefix = "aligned_field0_"  # 出力の先頭文字列

    align_all_channels_and_create_gifs(
        file_name=filename,
        field_idx=test_field,
        output_prefix=output_prefix,
        use_reference_first_frame=True,  # 全フレームを最初のフレームに合わせる
        duration_ms=300
    )
