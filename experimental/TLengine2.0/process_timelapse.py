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

def shift_ph_image(file_name: str, field_idx: int, time_idx: int,
                   vertical_shift: int, horizontal_shift: int) -> np.ndarray:
    """
    指定したファイル・Field・Timeからph画像を取得し、
    指定ピクセル分だけ縦横にずらした画像を返す。
    """
    ph_img = get_ph_image(file_name, field_idx, time_idx)
    shifted_ph = shift_image(ph_img, vertical_shift, horizontal_shift)
    return shifted_ph

# =============================================================================
# ここから自動位置合わせ & GIF作成用の追加関数
# =============================================================================

def calc_shift_by_phase_correlation(ref_img: np.ndarray, target_img: np.ndarray) -> tuple[int, int]:
    """
    phaseCorrelateを使ってref_imgに対するtarget_imgのx,yズレを推定する。
    返すのは (vertical_shift, horizontal_shift) のタプル。
    """
    # OpenCVのphaseCorrelateを使うには、FFTしたスペクトルを与える場合が多い
    # (ただし直接phaseCorrelateにimgを与えるやり方もあるが、ドキュメントにより推奨が異なる)
    ref_float = ref_img.astype(np.float32)
    tgt_float = target_img.astype(np.float32)

    # 2D-FFTを計算
    ref_fft = np.fft.fft2(ref_float)
    tgt_fft = np.fft.fft2(tgt_float)

    (shift_x, shift_y), _ = cv2.phaseCorrelate(ref_fft, tgt_fft)
    # phaseCorrelateが返すshift_x, shift_yは
    # 「targetを(x, y)だけ動かせばrefに重なる」という向き
    # shift_imageはshift_image(img, vertical, horizontal)なので順番と符号に注意
    vertical_shift = -int(round(shift_y))
    horizontal_shift = -int(round(shift_x))
    return (vertical_shift, horizontal_shift)

def align_ph_images_and_create_gif(
    file_name: str,
    field_idx: int,
    output_gif_path: str,
    use_reference_first_frame: bool = True,
    duration_ms: int = 300
):
    """
    指定したnd2から指定したfieldの全timeのph画像を取得し、
    自動で位置合わせを行ったうえでGIFを出力する。

    Parameters
    ----------
    file_name : str
        nd2ファイルのパス
    field_idx : int
        取得したい視野(Index) 0始まり
    output_gif_path : str
        出力するGIFのパス
    use_reference_first_frame : bool
        Trueなら最初のフレームを常に基準にして他のフレームを合わせる
        Falseなら一つ前のフレームに対して逐次合わせる（累積的）
    duration_ms : int
        GIFの1フレームあたりの表示時間(ミリ秒)
    """

    with nd2reader.ND2Reader(file_name) as images:
        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"
        num_timepoints = images.sizes.get("t", 1)

    # まず全フレームのph画像を読み込む
    ph_frames = [get_ph_image(file_name, field_idx, t) for t in range(num_timepoints)]

    # align後の画像を格納するリスト
    aligned_images: list[Image.Image] = []

    # 基準となるフレームを画像として保持
    ref_img = ph_frames[0]
    aligned_images.append(Image.fromarray(ref_img))

    prev_img = ref_img  # 累積モード用

    for t in range(1, num_timepoints):
        target_img = ph_frames[t]

        if use_reference_first_frame:
            # 常に最初のフレーム(ref_img)に対してズレを計算
            v_shift, h_shift = calc_shift_by_phase_correlation(ref_img, target_img)
        else:
            # 一つ前のフレーム(prev_img)に対してズレを計算
            v_shift, h_shift = calc_shift_by_phase_correlation(prev_img, target_img)

        shifted = shift_image(target_img, v_shift, h_shift)
        aligned_images.append(Image.fromarray(shifted))
        if not use_reference_first_frame:
            # 累積モードなら、今回シフトしたものを「次の基準」に更新
            prev_img = shifted

    # GIFとして保存
    aligned_images[0].save(
        output_gif_path,
        save_all=True,
        append_images=aligned_images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF saved to: {output_gif_path}")

# =============================================================================
# 動作テスト (必要に応じて実行)
# =============================================================================
if __name__ == "__main__":
    filename = "experimental/TLengine2.0/sk450gen120min-tl.nd2"
    test_field = 0       # 例: 最初のField
    output_gif = "aligned_field0_ph.gif"

    # 位置合わせしてGIF出力
    align_ph_images_and_create_gif(
        file_name=filename,
        field_idx=test_field,
        output_gif_path=output_gif,
        use_reference_first_frame=True,  # 最初のフレームを常に基準
        duration_ms=300
    )
