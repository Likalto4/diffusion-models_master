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


def main():
    # get all images in folder
    images_directory = repo_path / 'data/vindr-mammo/images/siemens15k'
    images = [x for x in images_directory.iterdir() if x.is_file() and x.suffix == '.png']
    # HP
    resolution = 512

    preprocess = Compose(
        [ # classic squared aspect-preserved centered image
            Resize(resolution, interpolation= InterpolationMode.BILINEAR),
            CenterCrop(resolution), 
        ]
    )

    max_val = 0 # max value in the dataset
    bar = tqdm(range(0,len(images)))
    for i, image in enumerate(images):
        # compute histogram
        im = Image.open(image)
        im_preprocessed = preprocess(im)
        im_preprocessed_array = np.asarray(im_preprocessed)
        # get max value
        image_max = im_preprocessed_array.max()
        if image_max > max_val:
            max_val = image_max
            # postfix in bar set to max value
            bar.set_postfix({'max_val': max_val, 'image_num':i})
        bar.update(1)

if __name__ == "__main__":
    main()
            
        