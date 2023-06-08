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
from tqdm import tqdm
import omidb
import cv2 as cv

def get_normal_BBox(image):
    """This function returns the mask of the breast, as well as the boundig box that encopasses it.

    Args:
        image (np.array): image as array

    Returns:
        omidb.bbox, np.array: returns omidb.box and mask image as np.arrays
    """

    mask = cv.threshold(image, 0, 255, cv.THRESH_BINARY)[1]
    nb_components, output, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape,dtype=np.uint8)
    img2[output == max_label] = 255
    contours, _ = cv.findContours(img2,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    cnt = contours[0]
    aux_im = img2
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(aux_im,(x,y),(x+w,y+h),(255,0,0),5)
    out_bbox = omidb.mark.BoundingBox(x, y, x+w, y+h)
    
    return out_bbox, img2, mask # returns bounding box and mask image.

def main():
    # HP
    folder_name = 'siemens15k'
    resolution = 512
    emergency_stop = None

    # read metadata file
    metadata_path = repo_path / 'data/vindr-mammo/metadata/finding_annotations.csv'
    # read metadata file
    metadata = pd.read_csv(metadata_path)
    # remove rows with "[No Finding]" in 'finding_categories'
    metadata = metadata[metadata['finding_categories'] != "['No Finding']"]
    # get only images of the training split
    metadata = metadata[metadata['split'] == 'training']
    
    # paths
    name_experiment = f'{folder_name}_real-training_{resolution}'
    new_images_folder_path = repo_path / 'generation/inpainting/data/images'/ name_experiment # where the images are going to
    new_metadata_dir = repo_path  / 'generation/inpainting/data/metadata'
    # create folders if needed
    new_images_folder_path.mkdir(parents=False, exist_ok=True)

    # loop starts
    tqdm_bar = tqdm(total=len(metadata), desc=f"Saving {resolution} images") # tqdm progress bar
    bad_image_count = 0 # count images that are not saved
    previous_file_name = None # to check if the image id is repeated
    differer = 0

    new_metadata = pd.DataFrame(columns=['filename','bbox', 'bbox_roi'])
    for j, row in metadata.iterrows(): # go over all images with lesion
       # get lesion bbox
        x1, y1, x2, y2 = row.xmin, row.ymin, row.xmax, row.ymax
        coord = np.asarray([x1, y1, x2, y2], dtype=int)
        side = row.laterality

        # add png extension
        file_name = row.image_id + '.png'
        image_path = repo_path / 'data/vindr-mammo/images/' f'{folder_name}' / file_name
        image_path_RGB = repo_path / 'data/vindr-mammo/images/' f'{folder_name}_RGB' / file_name
        
        # check if the image id is repeated
        if file_name == previous_file_name:
            differer += 1
            file_name_save = row.image_id + f'_{differer}.png'
        else:
            differer = 0
            file_name_save = file_name

        # check if the image exist in the folder (only SIEMENS will appear)
        if not image_path.exists():
            tqdm_bar.update()
            bad_image_count += 1
            print(f'bad image {file_name} count is now {bad_image_count}')
            continue

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
        # clip to 0 if negative
        coord = np.clip(coord, 0, None) 
        # if the coordinates are zero skip    
        if np.any(coord < 0) or (coord[0]-coord[2]) == 0:
            tqdm_bar.update()
            bad_image_count += 1
            print(f'bad image {file_name} count is now {bad_image_count}')
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
        # lesion bbox in "original size"
        x1_roi = min(coord_r[0], coord_r[2])
        y1_roi = min(coord_r[1], coord_r[3])
        x2_roi = max(coord_r[0], coord_r[2])
        y2_roi = max(coord_r[1], coord_r[3])
        bbox_roi = omidb.mark.BoundingBox(x1_roi, y1_roi, x2_roi, y2_roi)
        # take just one channel
        image_gray = cv.cvtColor(im_preprocessed, cv.COLOR_RGB2GRAY)
        # get the mask and the bbox
        bbox, _, _ = get_normal_BBox(image_gray)

        # Save important information
        cv.imwrite(str(new_images_folder_path / f'{file_name_save}'), image_gray[bbox.y1:bbox.y2, bbox.x1:bbox.x2])
        # save metadata using concat
        new_metadata = pd.concat(
            [
                new_metadata,
                pd.DataFrame([[file_name_save, bbox, bbox_roi]], columns=['filename','bbox', 'bbox_roi'])
            ],
            ignore_index=True,
        )
        tqdm_bar.update()
        # update previous_file_name
        previous_file_name = file_name
        # emergency stop
        if j == emergency_stop:
            break
    # save metadata
    new_metadata.to_csv( new_metadata_dir / f'{name_experiment}.csv',index=False)
    



if __name__ == '__main__':
    main()