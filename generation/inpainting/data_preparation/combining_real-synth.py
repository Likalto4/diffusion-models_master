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
import shutil

def main():
    name_dataset_1 = 'siemens_healthy-inpainted'
    name_dataset_2 = 'siemens15k_real-training_512'
    images_dir = 'generation/inpainting/data/images'
    metadata_dir = 'generation/inpainting/data/metadata'
    name_experiment = 'siemens-combined'
    
    # new paths
    new_images_dir = repo_path / images_dir / name_experiment
    new_images_dir.mkdir(parents=False, exist_ok=True)
    new_metadata_path = repo_path / metadata_dir / f'{name_experiment}.csv'


    # copy all files from dataset 1 to new folder using shutil
    shutil.copytree(repo_path / images_dir / f'{name_dataset_1}-cropped', new_images_dir, dirs_exist_ok=True)
    # copy all files from dataset 2 to new folder using shutil
    shutil.copytree(repo_path / images_dir / name_dataset_2, new_images_dir, dirs_exist_ok=True)

    print(f'Number of elements in new_images_dir: {len(os.listdir(new_images_dir))}')
    # open metadata files
    metadata_1 = pd.read_csv(repo_path / metadata_dir / f'{name_dataset_1}.csv')
    metadata_2 = pd.read_csv(repo_path / metadata_dir / f'{name_dataset_2}.csv')
    # stack metadata files
    metadata = pd.concat([metadata_1, metadata_2], ignore_index=True)
    # save metadata file
    metadata.to_csv(new_metadata_path, index=False)

if __name__ == '__main__':
    main()
