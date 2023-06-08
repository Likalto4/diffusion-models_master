from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve() # experiment path
# visible GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pandas as pd
import numpy as np
from PIL import Image
import re

def main():
    """In case the dataset needs the mammograms cropped on the breast area, this script will do it.
    """
    # HP
    image_folder = 'siemens_healthy-inpainted'
    stop_at = None
    
    # get the original images file
    original_images_path = repo_path /'generation/inpainting/data/images'/f'{image_folder}'
    # define cropped images directory
    cropped_images_path = repo_path /'generation/inpainting/data/images'/f'{image_folder}-cropped'
    cropped_images_path.mkdir(parents=True, exist_ok=True)

    # read metadata file
    metadata_path = repo_path / 'generation/inpainting/data/metadata'/ f'{image_folder}.csv'
    metadata = pd.read_csv(metadata_path)

    # go trhough all the rows in the metadata file
    for index, row in metadata.iterrows():
        # get image name
        image_name = row['filename']
        image_path = original_images_path / f'{image_name}'
        im = np.asarray(Image.open(image_path))
        # get the coordinates of the bounding box text prompt
        bbox = row['bbox']
        matches = re.findall(r"\d+", bbox)
        # remove odd positions in matches, as they are not coordinates
        matches = [int(i) for i in matches[1::2]]
        coordinates = np.array(matches, dtype=int)
        x1, y1, x2, y2 = coordinates

        # crop the image
        im = im[y1:y2, x1:x2]
        # save the image
        im = Image.fromarray(im)
        im.save(cropped_images_path / f'{image_name}')
        if index == stop_at:
            break

if __name__ == '__main__':
    main()
