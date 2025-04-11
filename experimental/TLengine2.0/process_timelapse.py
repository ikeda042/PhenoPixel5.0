import numpy as np
import os
import nd2reader
from PIL import Image
import cv2


def process_image(array):
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
    """
    base_output_dir = "timelapsen2dtotiff"

    os.makedirs(base_output_dir, exist_ok=True)

    with nd2reader.ND2Reader(file_name) as images:
        print(f"Available axes: {images.axes}")
        print(f"Sizes: {images.sizes}")

        images.bundle_axes = "cyx" if "c" in images.axes else "yx"
        images.iter_axes = "v"

        num_fields = images.sizes.get("v", 1)
        num_channels = images.sizes.get("c", 1)
        num_timepoints = images.sizes.get("t", 1)

        for field_idx in range(num_fields):
            field_folder = os.path.join(base_output_dir, f"Field_{field_idx + 1}")
            os.makedirs(field_folder, exist_ok=True)
            base_output_subdir_ph = field_folder + "/ph"
            base_output_subdir_fluo = field_folder + "/fluo"
            os.makedirs(base_output_subdir_ph, exist_ok=True)
            os.makedirs(base_output_subdir_fluo, exist_ok=True)

            for channel_idx in range(num_channels):
                for time_idx in range(num_timepoints):
                    images.default_coords.update(
                        {"v": field_idx, "c": channel_idx, "t": time_idx}
                    )
                    image_data = images[0]

                    if len(image_data.shape) == 3:
                        for i in range(image_data.shape[0]):
                            channel_image = process_image(image_data[i])
                            tiff_filename = os.path.join(
                                (
                                    base_output_subdir_ph
                                    if i == 1
                                    else base_output_subdir_fluo
                                ),
                                f"time_{time_idx + 1}_channel_{i}.tif",
                            )
                            img = Image.fromarray(channel_image)
                            img.save(tiff_filename)
                            print(f"Saved: {tiff_filename}")
                    else:
                        image_data = process_image(image_data)
                        tiff_filename = os.path.join(
                            field_folder, f"time_{time_idx + 1}.tif"
                        )
                        img = Image.fromarray(image_data)
                        img.save(tiff_filename)
                        print(f"Saved: {tiff_filename}")

filename = "experimental/TLengine2.0/sk450gen120min-tl.nd2"
extract_nd2(filename)
