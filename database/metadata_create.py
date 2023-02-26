#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import omidb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import pydicom
import yaml

class stats:
    """Class for keeping track of the statistics of the dataset
    """
    def __init__(self, N:int, B:int, M:int, IC:int):
        """Initialize the statistics object

        Args:
            N (int): Normal client status
            B (int): Bening client status
            M (int): Malignant client status
            IC (int): Interval Cancer client status
        """
        self.N = N
        self.M = M
        self.B = B
        self.IC = IC # not present
        self.image_CC = 0
        self.image_MLO = 0
        self.image_R = 0
        self.image_L = 0
        self.subtype = np.zeros(8, dtype=np.int32)

    def __repr__(self):
        """when printing the object, print the following:

        Returns:
            str: description of the statistics
        """
        return \
            f'Stats [N: {self.N}, M {self.M}, B: {self.B}, IC {self.IC},'\
            f'CC: {self.image_CC}, MLO: {self.image_MLO}, '\
            f'R: {self.image_R}, L:{self.image_L}, ' \
            f'Subtype: {np.array2string(self.subtype)} ]'

def update_stats_patient(overall:stats, client):

    # Save the general pathological status of the patient:
    #   M: malignant, N:Normal, CI: Interval Cancer
    # In the API, the label is given in the following order of importance:
    # CI > M > B > N
    if client.status.value == 'Interval Cancer': # not present
        overall.IC += 1
    elif client.status.value == 'Malignant':
        overall.M += 1
    elif client.status.value == 'Benign':
        overall.B += 1
    elif client.status.value == 'Normal':
        overall.N += 1
    return overall

def check_tag_and_value(json_data:dict, tag:str):
    """returns true if the tag is in the json file and has value

    Args:
        json_data (dict): json object with the data
        tag (str): DICOM tag

    Returns:
        bool: true if the tag is in the json file and has value
    """
    return (tag in json_data) and ('Value' in json_data[tag])

def check_presence(json_data:dict):
    """checks if the json file has the required fields

    Args:
        json_data (dict): jason object with the data

    Returns:
        bool: true if the json file has the required fields
    """
    # criteria for selecting images
    for_pres = json_data['00080068']['Value'][0] == 'FOR PRESENTATION' # only images for presentation
    series_descrip = check_tag_and_value(json_data, '0008103E') # check if the series description (side view or other) is defined and has value
    manufacturer = check_tag_and_value(json_data, '00080070') # check if the manufacturer is defined and has value
    fov_type = check_tag_and_value(json_data, '00191039') # check if the fov_type is defined and has value

    if for_pres: #and (not fov_type): #and series_descrip and manufacturer: # if all the criteria are met, return True
        return True
    else:
        return False

def json_tag2value(json_data:dict, tag:str):
    """returns the value of the tag in the json file

    Args:
        json_data (dict): json object with the data
        tag (str): DICOM tag

    Returns:
        str: value of the tag or None
    """
    if check_tag_and_value(json_data, tag):
        return json_data[tag]['Value'][0]
    else:
        return None
    
def main():
    #define the path to the dataset
    reading_path = '/mnt/mia_images/breast/omi-db/image_db/sharing/omi-db/'
    # db = omidb.DB(reading_path, clients=['demd100018', 'demd128247', 'demd843','demd94678'])
    db = omidb.DB(reading_path)

    # create dataframe storing path, presentation type, side, manufacturer, view
    df = None
    empty_json = 0

    # read dicom tags yaml
    with open('metadata/dicom_tags.yaml', 'r') as file:
        tags = yaml.load(file, Loader=yaml.FullLoader)

    # Initialize the statistics object
    overall = stats(0, 0, 0, 0)
    for client in tqdm(db, total=len(db.clients)):          #@ client level
        overall = update_stats_patient(overall, client) # extract the client status for statistics
        #nbss_data = db._nbss(client.id) # store nbss data of the client
        for episode in client.episodes:                     #@ episode level
            if episode.studies is not None: # only if the episode has studies
                for study in episode.studies:               #@ study level
                    for series in study.series:             #@ series level
                        for image in series.images:         #@ image level
                            with open(image.json_path, 'r') as file: # open the image json file
                                if file.read() != '': # check the file is not empty
                                    file.seek(0) # go back to the beginning of the file
                                    json_data = json.load(file)
                                else:
                                    empty_json += 1
                            # check if the json file has the relevant information
                            if check_presence(json_data): # if the image is for presentation
                                # object oriented information storage
                                df_sub = pd.DataFrame(
                                    {
                                        'client_id': client.id,
                                        'episode_id': episode.id,
                                        'study_id': study.id,
                                        'series_id': series.id,
                                        'image_id': image.id,
                                        'path': image.dcm_path,
                                        'json_path': image.json_path,
                                        'client_status': client.status.value,
                                        'marks': True if image.marks else False,
                                    }, index=[0])
                                # go through the tags and save the json-contained information
                                for tag in tags:
                                    df_sub = pd.concat([df_sub, pd.DataFrame({tag: json_tag2value(json_data, tags[tag])}, index=[0])], axis=1)
                                # concat in general df
                                df = pd.concat(
                                    [df, df_sub], axis=0)                        

    # save the dataframe
    df.to_csv('metadata/metadata_FP.csv', index=False)

    # print the statistics
    subclients_num =  len(df['client_id'].unique())
    print(f'Number of valid clients: {subclients_num}')
    print(f'Images with no valid json file: {empty_json}')\

if __name__ == '__main__':
    main()
