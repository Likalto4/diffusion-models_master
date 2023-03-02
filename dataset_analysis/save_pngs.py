# Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None


# Libraries
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2 as cv
from tqdm import tqdm

def main():

    metadata_path = 'metadata/metadata_pixelspacing.csv' # path to the metadata file
    metadata = pd.read_csv(metadata_path)
    paths = metadata['path'].values # get paths

    for i in tqdm(range(len(paths)//10)):
        path = paths[i]
        # load the dicom image
        ds = pydicom.dcmread(path)
        # apply the voi lut and convert to uint16
        image_array = apply_voi_lut(ds.pixel_array, ds)
        image_array = np.uint16(image_array)
        # turn right side images to left side
        if ds.ImageLaterality == 'R':
            image_array = np.fliplr(image_array)
        # get image id from metadata
        image_id = metadata.iloc[i]['image_id']
        # save the image
        cv.imwrite(f'images/{image_id}.png', image_array)

if __name__ == '__main__':
    main()