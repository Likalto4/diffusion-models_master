from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2 as cv
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    InterpolationMode,
)
from matplotlib import pyplot as plt

# HP
folder_name = 'breast40k'
resolution = 512
emergency_stop = None

# read metadata file
metadata_path = repo_path / 'data/metadata'/f'lesions_{folder_name}.csv' 
metadata = pd.read_csv(metadata_path)

# paths
new_images_folder_path = repo_path / 'data/images'/f'{folder_name}_RGB_wlesions' # where the images are going to
masks_folder_path = repo_path / 'data/masks'/f'{folder_name}_RGB_wlesions' # where the masks are going to

# create folders if needed
new_images_folder_path.mkdir(parents=False, exist_ok=True)
masks_folder_path.mkdir(parents=False, exist_ok=True)

# loop starts
tqdm_bar = tqdm(total=len(metadata), desc="Saving images and masks") # tqdm progress bar
bad_image_count = 0 # count images that are not saved
for j, row in metadata.iterrows(): # go over all images with lesion
    bbox = row['bbox']
    side = row['image_laterality']
    # get x1, y1, x2, y2 coordinates form string in that order
    bbox = bbox.replace('(', '').replace(')', '').split(',')
    x1, y1, x2, y2 = [int(i) for i in bbox]
    coord = np.asarray([x1, y1, x2, y2])
    file_name = row.image_id + '.png'
    image_path = repo_path / 'data/images' / folder_name / file_name
    image_path_RGB = repo_path / 'data/images' / f'{folder_name}_RGB' / file_name
    
    # read images
    im = np.asarray(Image.open(image_path))
    im_preprocessed = np.asarray(Image.open(image_path_RGB))
    
    # images sizes
    original_size = im.shape
    reshape_size = im_preprocessed.shape
    
    # if side is right, flip x coordinate
    if side == 'R':
        coord[0] =  original_size[1] - coord[0]# -1
        coord[2] =  original_size[1] - coord[2]# -1
    # if any of the coordinates is negative, or the coordinates are zero skip    
    if np.any(coord < 0) or (coord[0]-coord[2]) == 0:
        tqdm_bar.update()
        bad_image_count += 1
        print(f'bad image count is now {bad_image_count}')
        continue
    
    # get resized bounding box coordinates
    orig_height = original_size[0]
    orig_width = original_size[1]
    # vector of coordinates
    coord_r = (coord * (resolution / orig_width))
    coord_r = np.round(coord_r).astype(int) # round to nearest int
    
    # get the resized height and width difference beffofe cropping to remove from y coordinates (the larger, height)
    resize_diff = int((orig_height * resolution)/orig_width) - resolution
    resize_diff = int(resize_diff/2)
    # substract from y1 and y2
    coord_r[1] = coord_r[1] - resize_diff
    coord_r[3] = coord_r[3] - resize_diff
    
    # create mask using pil
    mask = Image.new('L', (resolution, resolution), 0) # creates empty 8-bit grayscale image
    draw = ImageDraw.Draw(mask) # prepare drawing
    # draw the rectangle using the coord_r
    draw.rectangle([coord_r[0], coord_r[1], coord_r[2], coord_r[3]], fill=255) # fill with white
    
    # save image and mask
    new_image_path = new_images_folder_path / file_name
    new_mask_path = masks_folder_path / file_name
    
    im_preprocessed = Image.fromarray(im_preprocessed)
    im_preprocessed.save(new_image_path)
    mask.save(new_mask_path)
    tqdm_bar.update()
    
    # emergency stop
    if j == emergency_stop:
        break