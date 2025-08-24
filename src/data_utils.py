import os.path
from pathlib import Path
import urllib.request

def download_dataset():
    # чтобы не закачивать на github этот датасет, скачаем его (один раз):
    RAW_DATASET_URL = "https://code.s3.yandex.net/deep-learning/tweets.txt"
    RAW_DATASET_PATH = 'data/tweets.txt'
    if not os.path.exists(RAW_DATASET_PATH):
        urllib.request.urlretrieve(RAW_DATASET_URL, RAW_DATASET_PATH)
    return RAW_DATASET_PATH
