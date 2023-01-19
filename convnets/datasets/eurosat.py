from pathlib import Path
import numpy as np
from .utils import download_data, generate_classes_list, generate_df
from sklearn.model_selection import train_test_split
import pandas as pd

class EuroSAT():
    def __init__(self, path="./data", val_size=0, test_size=0.2, random_seed=42, verbose=True, label_ratio=1):
        super().__init__()
        self.path = Path(path)
        self.num_classes = 10
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
        self.compressed_data_filename = 'EuroSAT.zip'
        self.data_folder = '2750'
        self.in_chans = 3
        self.verbose = verbose
        self.label_ratio = label_ratio
        # download and uncompress data
        uncompressed_data_path = download_data(
            self.path,
            self.compressed_data_filename,
            self.data_folder,
            self.url,
            self.verbose
        )
        self.classes = generate_classes_list(uncompressed_data_path)
        assert len(self.classes) == self.num_classes
        self.data = generate_df(self.classes, uncompressed_data_path, self.verbose)
        # make splits
        if self.test_size > 0:
            train_df, self.test = train_test_split(
                self.data,
                test_size=int(len(self.data)*self.test_size),
                random_state=self.random_seed,
                stratify=self.data.label.values
            )
        else: 
            train_df, self.test = self.data, None
        if self.val_size > 0:
            self.train, self.val = train_test_split(
                train_df,
                test_size=int(len(self.data)*self.val_size),
                random_state=self.random_seed,
                stratify=train_df.label.values
            )
        else: 
            self.train, self.val = train_df, None
        if self.verbose:
            print("Training samples", len(self.train))
            if self.val is not None:
                print("Validation samples", len(self.val))
            if self.test is not None:
                print("Test samples", len(self.test))
        # filter by label ratio (used for ssl validation experiments)
        if self.label_ratio < 1:
            train_labels = self.train.label.values
            train_images = self.train.image.values
            train_images_ratio, train_labels_ratio = [], []
            unique_labels = np.unique(train_labels)
            for label in unique_labels:
                filter = np.array(train_labels) == label
                ixs = filter.nonzero()[0]
                num_samples = filter.sum()
                ratio_ixs = np.random.choice(
                    ixs, int(self.label_ratio*num_samples), replace=False)
                train_images_ratio += (np.array(train_images)
                                       [ratio_ixs]).tolist()
                train_labels_ratio += (np.array(train_labels)
                                       [ratio_ixs]).tolist()
            self.train = pd.DataFrame(
                {'image': train_images_ratio, 'label': train_labels_ratio})
            if self.verbose:
                print("training samples after label ratio filtering:",
                      len(self.train))