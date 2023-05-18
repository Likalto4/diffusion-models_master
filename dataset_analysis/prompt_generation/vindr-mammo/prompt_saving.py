#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from datasets_local.metadata import subset_csv
import pandas as pd

def main():
    # metadata folder
    metadata_path = repo_path  /'data/vindr-mammo/metadata' / 'finding_annotations.csv'
    metadata = pd.read_csv(metadata_path, header=0)
    # filter to images with no findings
    metadata = metadata[metadata['finding_categories'] == "['No Finding']"]

    # files found in the folder
    directory_path = repo_path / 'data/vindr-mammo/images' / 'siemens15k_RGB'
    # get only the file stem
    file_names = [file.stem for file in directory_path.glob('*.png')]
    # make dataframe with names
    file_names = pd.DataFrame(file_names, columns=['image_id'])

    # filter the metadata with the file names
    metadata_actual = metadata.merge(file_names, on='image_id')
    # add column called breast_density_prompt with diferent values depending the desity
    metadata_actual['breast_density_prompt'] = metadata_actual['breast_density'].apply(lambda x: 'very low' if x == 'DENSITY A' else 'low' if x == 'DENSITY B' else 'high' if x == 'DENSITY C' else 'very high')
    metadata_actual['prompt'] = 'a mammogram in ' + metadata_actual['view_position'] + ' view ' + 'with ' + metadata_actual['breast_density_prompt'] + ' density'

    # drop all columns that are not prompt or image_id
    metadata_actual = metadata_actual[['image_id', 'prompt']]
    # add exension to the end of the image_id
    metadata_actual['image_id'] = metadata_actual['image_id'].astype(str) + '.png'
    # change image_id to file_name
    metadata_actual = metadata_actual.rename(columns={'image_id': 'file_name'})
    # transform to json
    metadata_actual = metadata_actual.to_json(orient='records', lines=True)
    

    # use same folder as files folder
    json_path = directory_path / 'metadata.jsonl'
    # save json
    with open(json_path, 'w') as f:
        f.write(metadata_actual)
    
if __name__ == "__main__":
    main()