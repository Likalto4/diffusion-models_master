from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import pandas as pd
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

# HP
folder_name = 'breast40k'
resolution = 512

# paths
metadata_path = repo_path / 'data/metadata'/f'{folder_name}_RGB.csv'
imageFolder_path = repo_path / 'data/images'/folder_name
lesion_folder_path = repo_path / 'data/images'/f'lesions_{folder_name}'
lesion_folder_path.mkdir(parents=False, exist_ok=True) # create if needed
# Read metadata file
metadata = pd.read_csv(metadata_path)
# filter only images with marks
metadata = metadata[metadata['marks'] == True]
metadata = metadata.reset_index(drop=True)

# preprocessing steps
preprocess = Compose(
    [ # classic squared aspect-preserved centered image
        Resize(resolution, interpolation= InterpolationMode.BILINEAR),
        CenterCrop(resolution), 
    ]
)
tqdm_bar = tqdm(total=len(metadata), desc="Saving lesions") # tqdm progress bar
# loop over the rows of the metadata file
bad_image_count = 0
for i, row in metadata.iterrows():
    # get the path to the image and laterality
    image_path = imageFolder_path / str(row['image_id'] + '.png')
    side = row['image_laterality']
    # read PIL image and convert to numpy array
    image = Image.open(image_path)
    image = np.asarray(image)
    # get the coordinates of the lesion
    bbox = row['bbox']
    bbox = bbox.replace('(', '').replace(')', '').split(',')
    x1, y1, x2, y2 = [int(i) for i in bbox]
    coord = np.asarray([x1, y1, x2, y2])
    if side == 'R':
        coord[0] =  image.shape[1] - coord[0]# -1
        coord[2] =  image.shape[1] - coord[2]# -1
        # switch x coordinates
        coord[[0, 2]] = coord[[2, 0]]
    
    # if any of the coordinates is negative, skip
    if np.any(coord < 0) or (coord[0]-coord[2]) == 0:
        tqdm_bar.update()
        bad_image_count += 1
        print(f'bad image count is now {bad_image_count}')
        continue

    # get lesion
    lesion = image[coord[1]:coord[3], coord[0]:coord[2]]

    # transform lesion to pil image and back to numpy array
    lesion = Image.fromarray(lesion)
    preprocessed_lesion = preprocess(lesion)
    preprocessed_lesion = np.asarray(preprocessed_lesion)

    # scale to 0-255 and convert to uint8
    im_uint8 = (preprocessed_lesion/ 4095.0)*255.0
    im_uint8 = im_uint8.astype(np.uint8)
    # convert to RGB
    im_RGB = np.stack((im_uint8,)*3, axis=-1)
    # save as png
    saving_path = lesion_folder_path / str(row['image_id'] + '.png')
    cv.imwrite(str(saving_path), im_RGB)
    # update progress bar
    tqdm_bar.update()