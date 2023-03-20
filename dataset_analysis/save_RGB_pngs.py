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
    config_path = repo_path / "dataset_analysis/config_latent.yaml"
    with open(config_path) as file: # expects the config file to be in the same directory
            config = yaml.load(file, Loader=yaml.FullLoader)

    folder_dir = repo_path / config['processing']['dataset']
    # create the save directory if needed
    saving_dir = folder_dir.parent / "breast10p_RGB"
    saving_dir.mkdir(parents=False, exist_ok=True)
    
    preprocess = Compose(
        [
            Resize(config['processing']['resolution'], interpolation= InterpolationMode.BILINEAR), #getattr(InterpolationMode, config['processing']['interpolation'])),  # Smaller edge is resized to 256 preserving aspect ratio
            CenterCrop(config['processing']['resolution']),  # Center crop to the desired squared resolution
        ]
    )
    break_i = None
    for i, filename in enumerate(tqdm(os.listdir(folder_dir), desc="Preprocessing images")):
        if filename.endswith(f".png"):
            path_ex = folder_dir / filename
            saving_path = saving_dir / filename
            # read the image using PIL
            im = Image.open(path_ex)
            # center crop to be square
            im_preprocessed = preprocess(im)
            # convert to numpy array
            im_preprocessed = np.array(im_preprocessed)
            # scale to 0-255 and convert to uint8
            im_uint8 = (im_preprocessed/ 4095.0)*255.0
            im_uint8 = im_uint8.astype(np.uint8)
            # convert to RGB
            im_RGB = np.stack((im_uint8,)*3, axis=-1)
            # save as png using cv
            cv.imwrite(str(saving_path), im_RGB)
            if i == break_i:
                break
            
if __name__ == "__main__":
    main()
            