# Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import numpy as np
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    InterpolationMode,
)

from tqdm import tqdm
import cv2 as cv


def main():
    # get all images from original folder
    images_directory = repo_path / 'data/vindr-mammo/images/siemens15k'
    images = [x for x in images_directory.iterdir() if x.is_file() and x.suffix == '.png'] # only pngs
    
    # HP
    resolution = 512
    max_value = 3500.0 # max value float
    break_i = None # early break for debugging
    images_dir = repo_path / 'data/vindr-mammo/images' / f'{images_directory.name}_RGB'

    # create dir if not exists
    images_dir.mkdir(parents=False, exist_ok=True)
    # preprocess pipeline
    preprocess = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(resolution, interpolation= InterpolationMode.BILINEAR),
            CenterCrop(resolution), 
        ]
    )

    bar = tqdm(range(0,len(images)), desc='Converting into uint8')
    for i, image in enumerate(images):
        # compute histogram
        im = Image.open(image)
        im_preprocessed = preprocess(im)
        im_preprocessed_array = np.asarray(im_preprocessed)
        # scale to 0-255 and convert to uint8
        im_scaled = (im_preprocessed_array / max_value)*255.0
        # saturate the values above 255
        im_scaled[im_scaled > 255] = 255
        im_uint8 = im_scaled.astype(np.uint8) # convert to uint8
        # convert to RGB
        im_RGB = np.stack((im_uint8,)*3, axis=-1)
        # save as png
        saving_path = images_dir / f'{image.stem}.png'
        cv.imwrite(str(saving_path), im_RGB)

        bar.update(1)
        if i == break_i: # early break for debugging
                break

if __name__ == "__main__":
    main()
            
        