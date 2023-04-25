# this code serves to get the bb coordinates from the omid dataset and save it in an INDIVIDUAL csv file
# The information can be later joined to the promt information in the promt_saving.py file

#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

from datasets_local.metadata import subset_csv
import pandas as pd
import omidb

# STEP 0. load omid dataset
reading_path = '/mnt/mia_images/breast/omi-db/image_db/sharing/omi-db/'
db = omidb.DB(reading_path)
clients = [client for client in db]

# STEP 1: get the marks column from the metadata file

# HP
folder_name = 'breast40k_RGB'
reference_file = 'metadata_Hologic.csv'

# in case the metadata file does not exist, it will be created
files_folder = repo_path / 'data/images' / f'{folder_name}' # images are stored here
reference_folder = repo_path / 'data/metadata/' / f'{reference_file}' # presentation metadata is stored here
metadata_path = subset_csv(files_folder, reference_folder) # subset metadata csv
# read csv file
metadata = pd.read_csv(metadata_path, header=0)
# we go through all the rows
for index, row in metadata.iterrows():
    if row['marks']==True: # access image only if the marks column is True
        
