import os
import sys
import h5py
import cv2
import glob
import pandas as pd
import numpy as np
import parmap
from dotenv import find_dotenv, load_dotenv


def resize_VGG(img):
    """
    Resize img to a 224 x 224 image
    """

    return cv2.resize(img,
                      (224, 224),
                      interpolation=cv2.INTER_AREA)


if __name__ == '__main__':

    load_dotenv(find_dotenv())

    # Check the env variables exist
    raw_msg = "Set your raw data absolute path in the .env file at project root"
    data_msg = "Set your processed data absolute path in the .env file at project root"
    assert "RAW_DIR" in os.environ, raw_msg
    assert "DATA_DIR" in os.environ, data_msg
