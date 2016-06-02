"""
Collection of utils for data processing
"""

import botocore
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
import math
from dotenv import load_dotenv, find_dotenv


def download_s3(s3, bucket_name, fname, fout):
    """
    Download file from s3

    details: download from specified bucket and output to fout
             if the file was not found (ClientError) return
             a False flag. (True otherwise)

    args: s3 (boto3 s3 client)
             bucket_name (str) s3 bucket name
             fname (str) full path to file on s3
             fout (str) full path to where the file will be stored locally

    returns: bool that states whether the download worked
    """
    try:
        s3.download_file(bucket_name, fname, fout)
        return True
    except botocore.exceptions.ClientError:
        return False
