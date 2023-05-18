# Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2 as cv

def main():
    # read metadata file
    vindr_original_directory = Path('/home/habtamu/physionet.org/files/vindr-mammo/1.0.0')
    # go through all files in the dicom files in the subdirevtories of all levels in the images directory
    images_directory = vindr_original_directory / 'images'
    # get all subdirectories
    subdirectories = [x for x in images_directory.iterdir() if x.is_dir()]
    # get all files in the subdirectories hat are dicom files
    files = []
    for subdirectory in subdirectories:
        files += [x for x in subdirectory.iterdir() if x.is_file() and x.suffix == '.dicom']
    
    not_uint16 = 0 # number of files that are not uint16
    # define tqdm bar and show current not uint16 files
    bar = tqdm(total=len(files), desc=f'Saving pngs')
    for path in files:
        # read dicom
        ds = pydicom.dcmread(path)
        # apply the voi lut and convert to uint16
        image_array = apply_voi_lut(ds.pixel_array, ds)
        if image_array.dtype != 'uint16':
            # skip
            not_uint16 += 1
            bar.update(1)
            continue

        if ds.ImageLaterality == 'R': # turn right side images to left side
            image_array = np.fliplr(image_array)
        
        # saving path
        saving_path = repo_path / 'data/vindr-mammo/images/siemens15k' / f'{path.stem}.png'
        # save the image
        cv.imwrite(str(saving_path), image_array)
        bar.update(1)
    bar.close()
    print(f'Number of files that are not uint16: {not_uint16}')

if __name__ == '__main__':
    main()