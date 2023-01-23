from pathlib import Path
import pandas as pd
import os
from glob import glob

# https://github.com/ServiceNow/seasonal-contrast

class SeCo100k():
    def __init__(self, path="/fastdata/seasonal_contrast_100k"):
        super().__init__()
        self.path = Path(path)
        self.in_chans = 3
        images = glob(str(self.path) + '/*/*.tif')
        self.data = pd.DataFrame({'image': images})


class SeCo():
    def __init__(self, path="/fastdata/seasonal_contrast_1m", bands=['B4', 'B3', 'B2']):
        super().__init__()
        self.path = Path(path)
        self.in_chans = len(bands)
        self.bands = bands
        aois = os.listdir(str(self.path))
        images = []
        for aoi in aois:
            images += [f'{self.path}/{aoi}/{f}' for f in os.listdir(str(self.path / aoi))]
        self.data = pd.DataFrame({'image': images})