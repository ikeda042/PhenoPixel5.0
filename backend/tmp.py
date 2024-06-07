import os
from PIL import Image


def extract_tiff(
    tiff_file, fluo_dual_layer: bool = False, singe_layer_mode: bool = True
) -> int:
    folders = [
        folder
        for folder in os.listdir("TempData")
        if os.path.isdir(os.path.join(".", folder))
    ]

    if fluo_dual_layer:
        for i in [i for i in ["Fluo1", "Fluo2", "PH"] if i not in folders]:
            try:
                os.mkdir(f"TempData/{i}")
            except:
                continue
    elif singe_layer_mode:
        for i in [i for i in ["PH"] if i not in folders]:
            try:
                os.mkdir(f"TempData/{i}")
            except:
                continue
    else:
        for i in [i for i in ["Fluo1", "PH"] if i not in folders]:
            try:
                os.mkdir(f"TempData/{i}")
            except:
                continue

    with Image.open(tiff_file) as tiff:
        num_pages = tiff.n_frames
        print("###############################################")
        print(num_pages)
        print("###############################################")
        img_num = 0
        if fluo_dual_layer:
            for i in range(num_pages):
                tiff.seek(i)
                if (i + 2) % 3 == 0:
                    filename = f"TempData/Fluo1/{img_num}.tif"
                elif (i + 2) % 3 == 1:
                    filename = f"TempData/Fluo2/{img_num}.tif"
                    img_num += 1
                else:
                    filename = f"TempData/PH/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")
        elif singe_layer_mode:
            for i in range(num_pages):
                tiff.seek(i)
                filename = f"TempData/PH/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")
                img_num += 1
        else:
            for i in range(num_pages):
                tiff.seek(i)
                if (i + 1) % 2 == 0:
                    filename = f"TempData/Fluo1/{img_num}.tif"
                    img_num += 1
                else:
                    filename = f"TempData/PH/{img_num}.tif"
                print(filename)
                tiff.save(filename, format="TIFF")

    return num_pages
