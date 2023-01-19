from pathlib import Path
import pandas as pd
from glob import glob

# https://github.com/ServiceNow/seasonal-contrast

class SeCo():
    def __init__(self, path="/fastdata/seasonal_contrast_100k", verbose=True):
        super().__init__()
        self.path = Path(path)
        self.in_chans = 3
        self.verbose = verbose
        images = glob(str(self.path) + '/*/*.tif')
        self.data = pd.DataFrame({'image': images})