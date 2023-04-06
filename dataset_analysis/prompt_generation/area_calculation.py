#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import csv
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm

# HP
folder_name = 'breast10p_RGB'

# read paths csv
filenames_csv = repo_path / f'data/filenames/{folder_name}.csv'
filenames_pd = pd.read_csv(filenames_csv, header=None)[0].values

#create pandas for the results
results_pd = pd.DataFrame(columns=['id', 'breast_percentage'])
for im_num in tqdm(range(len(filenames_pd)), desc='images'):
    im_id = filenames_pd[im_num] # image id
    im_path = repo_path / f'data/images/{folder_name}' / f'{im_id}'
    # read image as numpy array (remember image has 3 identical gray channels)
    img = cv.imread(str(im_path))[:,:,0]
    # get img histogram
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    # get zero pixels
    zero_pixels = np.where(img == 0)
    # create binary mask of the zero pixels and invert it
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[zero_pixels] = 1
    mask = 1 - mask # invert mask

    # total area
    total_area = img.shape[0] * img.shape[1]
    # breast area
    breast_area = np.sum(mask)
    # breast percentage
    breast_percentage = breast_area / total_area
    
    # add to results using concat
    results_pd = pd.concat([results_pd, pd.DataFrame([[im_id, breast_percentage]], columns=['id', 'breast_percentage'])], ignore_index=True)

# add new column with small if area<0.5 and big if bigger than 0.5
results_pd['size'] = results_pd['breast_percentage'].apply(lambda x: 'small' if x<0.5 else 'big')
# save results
csv_path = filenames_csv.parent.parent / 'metadata' / f'area_{folder_name}.csv'
results_pd.to_csv(csv_path, index=False)