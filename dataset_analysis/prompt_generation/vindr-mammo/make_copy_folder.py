
#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import shutil
import pandas as pd
from tqdm import tqdm


def main():
    # read metadata
    metadata_path = repo_path / 'data/vindr-mammo/metadata/training_metadata.csv'
    metadata = pd.read_csv(metadata_path, header=0)

    # create a copy of the files in metadata folder
    old_images_dir = repo_path / 'data/vindr-mammo/images/siemens15k_RGB'
    new_images_dir = repo_path / 'data/vindr-mammo/images/SIEMENS_8bits_512_training'
    new_images_dir.mkdir(parents=False, exist_ok=True)
       
    for file_name in tqdm(metadata['image_id']):
        file_path = old_images_dir / f'{file_name}.png'
        shutil.copy(file_path, new_images_dir)

if __name__ == "__main__":
    main()