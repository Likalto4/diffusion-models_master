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
import yaml

def main():
    # load the config file
    config_path = "png_saving.yaml"
    with open(config_path) as file: # expects the config file to be in the same directory
            config = yaml.load(file, Loader=yaml.FullLoader)

    folder_dir = repo_path / config['processing']['dataset_path']
    # create the save directory if needed
    saving_dir = folder_dir.parent / config['processing']['saving_name']
    saving_dir.mkdir(parents=False, exist_ok=True)
    
    preprocess = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(config['processing']['resolution'], interpolation= InterpolationMode.BILINEAR),
            CenterCrop(config['processing']['resolution']), 
        ]
    )
    break_i = None # early break for debugging
    for i, filename in enumerate(tqdm(os.listdir(folder_dir), desc="Preprocessing images")):
        if filename.endswith(f".png"):
            path_im = folder_dir / filename
            saving_path = saving_dir / filename
            # read the image using PIL
            im = Image.open(path_im)
            im_preprocessed = preprocess(im)
            im_preprocessed = np.array(im_preprocessed)
            # scale to 0-255 and convert to uint8
            im_uint8 = (im_preprocessed/ 4095.0)*255.0
            im_uint8 = im_uint8.astype(np.uint8)
            # convert to RGB
            im_RGB = np.stack((im_uint8,)*3, axis=-1)
            # save as png
            cv.imwrite(str(saving_path), im_RGB)
            if i == break_i: # early break for debugging
                break
            
if __name__ == "__main__":
    main()
            