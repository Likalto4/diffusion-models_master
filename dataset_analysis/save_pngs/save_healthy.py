from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 as cv
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    InterpolationMode,
)
import pandas as pd

# code to save only the healthy images using the metadata information

def main():
    source_folder = 'breast40k_RGB'
    saving_folder = 'breast40k_RGB_healthy'
    # create saving folder if needed
    saving_folder_path = repo_path / 'data/images' / f'{saving_folder}'
    saving_folder_path.mkdir(parents=False, exist_ok=True)
    
    # read the metadata
    metadata_path = repo_path / 'data/metadata' / f'{source_folder}.csv'
    metadata = pd.read_csv(metadata_path)
    
    # filter metadata to only keep healthy images, meaning without marks
    metadata = metadata[metadata['marks'] == False]
    metadata.reset_index(inplace=True, drop=True)
    # go through all images in the metadata
    break_i = None  # early break for debugging
    for i, row in tqdm(metadata.iterrows(), desc="Preprocessing healthy", total=len(metadata)):
        # get image path
        im_name = row['image_id'] + '.png'
        im_path = repo_path / 'data/images' / f'{source_folder}' / f'{im_name}'
        im = Image.open(im_path)
        im = np.asarray(im)
        # save as png
        saving_path = saving_folder_path / f'{im_name}'
        cv.imwrite(str(saving_path), im)
        if i == break_i: # early break for debugging
            break
        
if __name__ == "__main__":
    main()
        