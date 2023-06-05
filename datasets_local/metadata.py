#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import csv
import pandas as pd

def create_folder_csv(folder_dir:Path, image_extension: str):
    """Creates a csv file with the name of the files in the folder with specific extension.

    Args:
        folder_dir (Path): images folder
        image_extension (str): png, jpg, etc.
    """
    # get folder name from directory
    folder_name = folder_dir.name
    # check if the csv file with the filenames already exists
    csv_path = folder_dir.parent.parent / 'filenames' / f'{folder_name}.csv'
    if not csv_path.exists(): # if not, create it
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for filename in os.listdir(folder_dir):
                if filename.endswith(f".{image_extension}"):
                    writer.writerow([filename])
    return csv_path

def subset_csv(files_folder:Path, reference_folder:Path):
    """creates subset coming from the ids of the files in the files folder with reference to the referencer forlder

    Args:
        files_folder (Path): files folder (absolute)
        reference_folder (Path): reference folder
    """
    csv_path = create_folder_csv(files_folder, 'png') # create name csv if it does not exist
    # open csv file
    name_csv = pd.read_csv(csv_path, header=None)
    # set column name as filename
    name_csv.columns = ['filename']
    # remove extension in all filenames in name_csv (for comperison with general metadata)
    name_csv['filename'] = name_csv['filename'].str.replace('.png', '', regex=True)
    # open general metadata csv file
    general_csv = pd.read_csv(reference_folder, header=0)
    # create new csv only with the filenames in the folder
    new_csv = general_csv[general_csv['image_id'].isin(name_csv['filename'])]
    # save new csv
    save_path = files_folder.parent.parent / 'metadata' / f'{files_folder.name}.csv'
    new_csv.to_csv(save_path, index=False)
    
    return save_path

