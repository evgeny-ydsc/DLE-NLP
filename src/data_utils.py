import os.path
from pathlib import Path
import urllib.request

def download_dataset(raw_dataset_url:str, raw_dataset_path:str):
    # чтобы не закачивать на github этот датасет, скачаем его (один раз):

    if not os.path.exists(raw_dataset_path):
        urllib.request.urlretrieve(raw_dataset_url, raw_dataset_path)
