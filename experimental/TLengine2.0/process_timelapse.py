import os
import re
import glob
import numpy as np
import cv2
from PIL import Image
import nd2reader

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


def get_ph_image(file_name: str, field_idx: int, time_idx: int) -> np.ndarray:
    """
    指定したファイル・Field・Timeからph (channel=0) の画像を取得し、
    process_image を通して正規化した numpy.ndarray を返す関数。
    """
    with nd2reader.ND2Reader(file_name) as images:
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"
        images.default_coords.update({"v": field_idx, "t": time_idx})
        image_data = images[0]

        if len(image_data.shape) == 3:
            # channel軸がある場合は channel=0 (ph) を取り出す
            ph_image_data = image_data[0]
        else:
            # channelが1つしかない場合
            ph_image_data = image_data

        return process_image(ph_image_data)

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

def load_image_as_array(path: str) -> np.ndarray:
    """
    TIFF ファイルなどを読み込んで np.uint8 の 2D配列として返す。
    """
    pil_img = Image.open(path)
    return np.array(pil_img, dtype=np.uint8)

def save_array_as_tiff(img_array: np.ndarray, path: str):
    """
    np.uint8 の 2D配列を TIFF として保存する。
    """
    pil_img = Image.fromarray(img_array)
    pil_img.save(path, format="TIFF")

def align_extracted_tifs_and_create_gifs(
    base_output_dir: str,
    field_idx: int,
    use_reference_first_frame: bool = True,
    duration_ms: int = 300
):
    """
    extract_nd2 ですでに出力されたディレクトリ構造を再利用し、
    指定フィールドの ph / fluo1 / fluo2 を位置合わせして
    - TIFFを座標補正後に上書き
    - ph, fluo1, fluo2 のGIFを出力

    Parameters
    ----------
    base_output_dir : str
        extract_nd2 で指定したベース出力ディレクトリ（例: experimental/TLengine2.0/output/sk450gen120min-tl）
    field_idx : int
        位置合わせしたい Field のインデックス (0始まり)
    use_reference_first_frame : bool
        True なら最初のフレームを基準に他フレームを合わせる
        False なら直前フレームに対する累積合わせ
    duration_ms : int
        GIF 1フレームあたりの表示時間 (ミリ秒)
    """
    # フィールドディレクトリ (例: Field_1, Field_2, ...)
    field_dir = os.path.join(base_output_dir, f"Field_{field_idx + 1}")

    # ph, fluo1, fluo2 のサブフォルダ
    ph_dir = os.path.join(field_dir, "ph")
    fluo1_dir = os.path.join(field_dir, "fluo1")
    fluo2_dir = os.path.join(field_dir, "fluo2")

    # time_\d+_ph.tif, time_\d+_fluo1.tif, ... を探す
    ph_paths = sorted(glob.glob(os.path.join(ph_dir, "time_*_ph.tif")))
    fluo1_paths = sorted(glob.glob(os.path.join(fluo1_dir, "time_*_fluo1.tif")))
    fluo2_paths = sorted(glob.glob(os.path.join(fluo2_dir, "time_*_fluo2.tif")))

    # フレーム数は ph_paths の数で決める（fluo1, fluo2 が途中までしかない場合も想定）
    num_timepoints = len(ph_paths)
    if num_timepoints == 0:
        print(f"No TIFF files found in {ph_dir}. Abort.")
        return

    # ph, fluo1, fluo2 の各フレームを np.ndarray で格納
    ph_frames = [load_image_as_array(p) for p in ph_paths]
    fluo1_frames = [load_image_as_array(p) for p in fluo1_paths] if len(fluo1_paths) == num_timepoints else []
    fluo2_frames = [load_image_as_array(p) for p in fluo2_paths] if len(fluo2_paths) == num_timepoints else []

    # 補正後画像を GIF 用に保持するリスト
    aligned_ph = []
    aligned_fluo1 = []
    aligned_fluo2 = []

    # 最初のフレームを基準
    ref_img = ph_frames[0]
    aligned_ph.append(Image.fromarray(ref_img))  # GIF用にPIL化
    if fluo1_frames:
        aligned_fluo1.append(Image.fromarray(fluo1_frames[0]))
    if fluo2_frames:
        aligned_fluo2.append(Image.fromarray(fluo2_frames[0]))

    prev_img = ref_img  # 累積モード用

    # time=0 (最初のフレーム) は既に補正済みとみなし、そのまま上書き（実質変わらない）
    save_array_as_tiff(ref_img, ph_paths[0])
    if fluo1_frames:
        save_array_as_tiff(fluo1_frames[0], fluo1_paths[0])
    if fluo2_frames:
        save_array_as_tiff(fluo2_frames[0], fluo2_paths[0])

    # 2フレーム目以降
    for i in range(1, num_timepoints):
        ph_target = ph_frames[i]

        # ph間のシフトを計算
        if use_reference_first_frame:
            v_shift, h_shift = calc_shift_by_phase_correlation(ref_img, ph_target)
        else:
            v_shift, h_shift = calc_shift_by_phase_correlation(prev_img, ph_target)

        # ph をシフト
        shifted_ph = shift_image(ph_target, v_shift, h_shift)
        aligned_ph.append(Image.fromarray(shifted_ph))
        # 上書き保存
        save_array_as_tiff(shifted_ph, ph_paths[i])

        # 累積モードなら基準を更新
        if not use_reference_first_frame:
            prev_img = shifted_ph

        # fluo1, fluo2 があれば同じシフト量を適用
        if fluo1_frames:
            fluo1_target = fluo1_frames[i]
            shifted_f1 = shift_image(fluo1_target, v_shift, h_shift)
            aligned_fluo1.append(Image.fromarray(shifted_f1))
            save_array_as_tiff(shifted_f1, fluo1_paths[i])

        if fluo2_frames:
            fluo2_target = fluo2_frames[i]
            shifted_f2 = shift_image(fluo2_target, v_shift, h_shift)
            aligned_fluo2.append(Image.fromarray(shifted_f2))
            save_array_as_tiff(shifted_f2, fluo2_paths[i])

    # それぞれGIFとして保存
    # ph
    ph_gif_path = os.path.join(field_dir, "aligned_ph.gif")
    aligned_ph[0].save(
        ph_gif_path,
        save_all=True,
        append_images=aligned_ph[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"ph GIF saved to: {ph_gif_path}")

    # fluo1
    if fluo1_frames:
        fluo1_gif_path = os.path.join(field_dir, "aligned_fluo1.gif")
        aligned_fluo1[0].save(
            fluo1_gif_path,
            save_all=True,
            append_images=aligned_fluo1[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"fluo1 GIF saved to: {fluo1_gif_path}")

    # fluo2
    if fluo2_frames:
        fluo2_gif_path = os.path.join(field_dir, "aligned_fluo2.gif")
        aligned_fluo2[0].save(
            fluo2_gif_path,
            save_all=True,
            append_images=aligned_fluo2[1:],
            duration=duration_ms,
            loop=0
        )
        print(f"fluo2 GIF saved to: {fluo2_gif_path}")

# ============================================================================
# 使い方
# ============================================================================
if __name__ == "__main__":
    # 例として
    # extract_nd2("experimental/TLengine2.0/sk450gen120min-tl.nd2") が作成した
    #  "experimental/TLengine2.0/output/sk450gen120min-tl" 以下の構造を再利用
    base_dir = "experimental/TLengine2.0/output/sk450gen120min-tl"
    test_field_index = 0

    # ph, fluo1, fluo2 の TIFF を座標補正 + 上書き、
    # かつ 3つの GIF を出力
    align_extracted_tifs_and_create_gifs(
        base_output_dir=base_dir,
        field_idx=test_field_index,
        use_reference_first_frame=True,  # 最初のフレームを常に基準
        duration_ms=300
    )
