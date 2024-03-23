from typing import Literal
from app.nd2extract import extract_nd2
from app.pyfiles.image_process import image_process
import os
from .pyfiles.database import Base, Cell
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
import shutil
from app.pyfiles.SelectGUI import app
import time
import matplotlib


def delete_all():
    dirs = [
        "app_data",
        "Fluo",
        "frames",
        "manual_detection_data",
        "manual_detection_data_raw",
        "PH",
        "ph_contours",
        "Fluo1",
        "Fluo2",
    ]
    files = ["cell.db", "image_labels.db"]
    temp_dirs = [f"TempData/{i}" for i in dirs if i in os.listdir("TempData")] + [
        "ph_contours"
    ]
    all_dirs = temp_dirs + ["Cell", "nd2totiff"]

    if "TempData" in os.listdir():
        for dir_path in all_dirs:
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(e)

    for file_path in files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(e)


def main(
    file_name: str,
    param1: int,
    param2: int,
    img_size: int,
    mode: Literal[
        "all", "data_analysis", "data_analysis_all", "delete_all", "", "load stackfile"
    ] = "all",
    layer_mode: Literal[
        "Triple Channel", "Dual Channel", "Single Channel"
    ] = "Dual Channel",
) -> None:
    delete_all()
    if layer_mode == "Triple Channel":
        dual_layer_mode = True
        single_layer_mode = False
    elif layer_mode == "Single Channel":
        dual_layer_mode = False
        single_layer_mode = True
    else:
        dual_layer_mode = False
        single_layer_mode = False

    if mode == "all":
        if file_name.split(".")[-1] == "nd2":
            extract_nd2(file_name)
            file_name = file_name.split("/")[-1].split(".")[0] + ".tif"
        image_process(
            input_filename=file_name,
            param1=param1,
            param2=param2,
            image_size=img_size,
            fluo_dual_layer_mode=dual_layer_mode,
            single_layer_mode=single_layer_mode,
        )
        app()
