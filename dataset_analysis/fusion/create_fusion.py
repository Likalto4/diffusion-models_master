from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import shutil
import pandas as pd
from tqdm import tqdm

# HP
make_copy_of_files = True

def main():
    # directories
    omid_directory = repo_path / 'data/images/breast40k_RGB_healthy'
    vindr_directory = repo_path / 'data/vindr-mammo/images/siemens15k_RGB_healthy'
    fusion_directory = repo_path / 'data/fusion/images/RGB_healthy'

    if make_copy_of_files:
        # create fusion directory if needed
        fusion_directory.mkdir(parents=False, exist_ok=True)
        # copy all the images from omid and vindr to fusion, only pngs
        for filename in tqdm(os.listdir(omid_directory), desc='copying omid images'):
            if filename.endswith(f".png"):
                path_im = omid_directory / filename
                saving_path = fusion_directory / filename
                shutil.copy(path_im, saving_path)

        for filename in tqdm(os.listdir(vindr_directory), desc='copying vindr images'):
            if filename.endswith(f".png"):
                path_im = vindr_directory / filename
                saving_path = fusion_directory / filename
                shutil.copy(path_im, saving_path)

    # get jsonl files in omid and vindr
    omid_jsonl = omid_directory / 'metadata.jsonl'
    vindr_jsonl = vindr_directory / 'metadata.jsonl'

    # change omid "mammogram" for "hologic mammogram"
    omid_metadata = pd.read_json(omid_jsonl, lines=True)
    omid_metadata['prompt'] = omid_metadata['prompt'].str.replace('mammogram', 'hologic mammogram')
    # do the same for vindr but for siemens
    vindr_metadata = pd.read_json(vindr_jsonl, lines=True)
    vindr_metadata['prompt'] = vindr_metadata['prompt'].str.replace('mammogram', 'siemens mammogram')
    # stack both dataframes
    metadata = pd.concat([omid_metadata, vindr_metadata], ignore_index=True)
    # save a jsonl file in fusion directory
    jsonl_path = fusion_directory / 'metadata.jsonl'
    metadata.to_json(jsonl_path, orient='records', lines=True)

if __name__ == '__main__':
    main()