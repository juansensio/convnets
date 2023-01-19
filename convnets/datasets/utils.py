import requests
import os
import pandas as pd
import zipfile
from tqdm import tqdm

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unzip_file(source, path, msg="Extracting"):
    with zipfile.ZipFile(source) as zf:
        for member in tqdm(zf.infolist(), desc=msg):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                pass

def download_data(path, compressed_data_filename, data_folder, url, verbose):
    compressed_data_path = path / compressed_data_filename
    uncompressed_data_path = path / data_folder
    # create data folder
    os.makedirs(path, exist_ok=True)
    # extract
    if not os.path.isdir(uncompressed_data_path):
        # check data is not already downloaded
        if not os.path.isfile(compressed_data_path):
            print("downloading data ...")
            download_url(url, compressed_data_path)
        else:
            print("data already downloaded !")
        unzip_file(compressed_data_path, path, msg="extracting data ...")
    else:
        if verbose:
            print("data already downloaded and extracted !")
    return uncompressed_data_path


# retrieve classes from folder structure	
def generate_classes_list(uncompressed_data_path):
    return sorted(os.listdir(uncompressed_data_path))


def generate_df(classes, uncompressed_data_path, verbose):
    images, labels = [], []
    for ix, label in enumerate(classes):
        _images = os.listdir(uncompressed_data_path / label)
        images += [str(uncompressed_data_path /
                       label / img) for img in _images]
        labels += [ix]*len(_images)
    assert len(images) == len(labels)
    if verbose:
        print(f'Number of images: {len(images)}')
    return pd.DataFrame({'image': images, 'label': labels})

