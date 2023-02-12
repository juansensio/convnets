import os 
from glob import glob 
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import argparse
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import numpy as np
from einops import rearrange
import shutil

# resize image by keeping the aspect ratio and cropping the center
def resize_and_crop(path, size=256):
    img = Image.open(path).convert('RGB') # some images are RGBA
    w, h = img.size
    ar = w / h
    if ar > 1:
        img = img.resize((int(size * ar), size), Image.BILINEAR)
        w, h = img.size
        return img.crop((w//2 - size//2, 0, w//2 + size//2, size))
    img = img.resize((size, int(size / ar)), Image.BILINEAR)
    w, h = img.size
    return img.crop((0, h//2 - size//2, size, h//2 + size//2))	

# preprocess a single image and save the result
def preprocess(args):
    dst_path, mode, path, size, cls = args
    img_name = path.split('/')[-1]
    if cls is None:
        dst_folder = f'{dst_path}/{mode}'
    else:
        dst_folder = f'{dst_path}/{mode}/{cls}'
    os.makedirs(dst_folder, exist_ok=True)
    new_path = f'{dst_folder}/{img_name}'
    if size is None: 
        # just copy the original image
        # shutil.copyfile(path, new_path) # cannot copy because some image are not RGB (we could convert them afterwards but better here for performance)
        img = Image.open(path).convert('RGB') # some images are RGBA
    else:
        img = resize_and_crop(path, size)
    img.save(new_path)
    return new_path

# this function will process the entire dataset, resizing and croping the images and generating the output folders for training
# the datasets is supposed to be downloaded from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

def process(base_path = 'data/ILSVRC', dst_path = 'data/imagenet256', size=None, workers=None):
    # this is where the dataset is supposed to be
    path = f'{base_path}/Data/CLS-LOC/'
    classes = sorted(os.listdir(f'{path}/train'))
    assert len(classes) == 1000
    # create df with all the training images and their labels
    print("Generating training set...")
    images, labels, cls = [], [], []
    for i, c in enumerate(tqdm(classes)):
        this_images = glob(f'{path}/train/{c}/*.JPEG')
        images += this_images
        labels += [i] * len(this_images)
        cls += [c] * len(this_images)
    train = pd.DataFrame({'image': images, 'label': labels, 'class': cls})
    assert len(train) == 1281167
    # repeat for validation
    print("Generating validation set...")
    val_annotations = glob(f'{base_path}/Annotations/CLS-LOC/val/*.xml')
    val_images, val_labels, val_cls = [], [], []
    for ann in tqdm(val_annotations):
        # read xml
        tree = ET.parse(ann)
        root = tree.getroot()
        cls = root.findall('object')[0].find('name').text
        val_cls.append(cls)
        val_labels.append(classes.index(cls))
        val_images.append(f'{path}/val/{ann.split("/")[-1].split(".")[0]}.JPEG')
    val = pd.DataFrame({'image': val_images, 'label': val_labels, 'class': val_cls})
    assert len(val.label.unique()) == len(classes)
    assert len(val) == 50000
    # repeat for test
    test = glob(f'{path}/test/*.JPEG')
    assert len(test) == 100000
    # process images in parallel
    args = [(dst_path, 'train', img, size, cls) for img, cls in zip(train.image.values, train['class'].values)] + \
        [(dst_path, 'val', img, size, cls) for img, cls in zip(val.image.values, val['class'].values)] + \
        [(dst_path, 'test', img, size, None) for img in test]
    print("Processing images...")
    num_cores = multiprocessing.cpu_count() if workers is None else workers
    with ThreadPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(args)) as progress:
            futures = []
            for arg in args:
                future = pool.submit(preprocess, arg) # enviamos la tupla de argumentos
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
    assert len(results) == len(train) + len(val) + len(test)

def compute_mean_std(image):
    image = io.imread(image).astype(np.float32) / 255
    return image.mean(axis=(0,1)), image.std(axis=(0,1))

def compute_stats(images, workers=None):
    num_cores = multiprocessing.cpu_count() if workers is None else workers
    with ThreadPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(images)) as progress:
            futures = []
            for img in images:
                future = pool.submit(compute_mean_std, img) 
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
    mean, std = zip(*results)
    return np.array(mean).mean(axis=0), np.array(std).mean(axis=0)

class ImageNet(Dataset):
    def __init__(self, path, mode, trans=None, **kwargs):
        assert mode in ['train', 'val', 'test'], 'mode must be train, val or test'
        path = Path(path)
        self.labels = None
        if mode == 'test':
            self.images = glob(str(path/'test/*.JPEG'))
            assert len(self.images) == 100000
        else:
            classes = sorted(os.listdir(path/mode))
            assert len(classes) == 1000
            images, labels = [], []
            for i, c in enumerate(classes):
                this_images = glob(str(path/mode/c/'*.JPEG'))
                images += this_images
                labels += [i] * len(this_images)
            if mode == 'train':
                assert len(images) == 1281167
            else:
                assert len(images) == 50000
            self.images, self.labels = images, labels
        self.trans = trans 

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, ix):
        im = io.imread(self.images[ix])
        if self.trans is not None:
            im = self.trans(image=im)['image']
        im = im.astype(np.float32) / 255.
        im = rearrange(im, 'h w c -> c h w')
        if self.labels is not None:
            return im, self.labels[ix]
        return im
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process imagenet.')
    parser.add_argument('--base-path', help='Folder with the original imagenet dataset')
    parser.add_argument('--dst-path', help='Folder where the processed dataset will be saved')
    parser.add_argument('--size', help='Size of the images', default=None) # 
    parser.add_argument('--workers', help='Number of workers to use', default=None)
    args = parser.parse_args()
    process(
        base_path=args.base_path,
        dst_path=args.dst_path,
        size=args.size,
        workers=args.workers
    )