#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None


import torch
import pandas as pd
from PIL import Image
import csv

# create a dataset class for our breast images
class breast_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: Path, images_dir: Path, transform=None):
        """_summary_

        Args:
            csv_path (Path): path to the csv file with the filenames
            images_dir (Path): path to the folder with the images
            transform (function, optional): transformation function. Usually pytorch.Transform. Defaults to None.
        """
        self.names = pd.read_csv(csv_path, header=None) # read csv file
        self.images_dir = images_dir # path to image folder
        self.transform = transform # transform to apply to images
    
    def __len__(self):
        """returns the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.names)
    
    def __repr__(self) -> str:
        """printing the dataset will return the length of the dataset

        Returns:
            str: length of the dataset
        """
        return f"({len(self)} images)"
    
    def __getitem__(self, idx: int):
        """returns the image at index idx

        Args:
            idx (int): index in the csv file

        Returns:
            PIL.Image: PIL image
        """
        img_path = self.images_dir / self.names.iloc[idx, 0] # get image path
        image = Image.open(img_path) # open image
        # image = np.array(image, dtype=np.float32) # convert to numpy array
        if self.transform: # apply transform if it exists
            image = self.transform(image)
        return image
    
    def set_transform(self, transform):
        """set the transform to apply to the images

        Args:
            transform (function): transform to apply to the images
        """
        self.transform = transform

class breast_dataset_latents(breast_dataset):
    def __getitem__(self, idx: int):
        img_path = self.images_dir / self.names.iloc[idx, 0] # get image path
        # load torch tensor
        image = torch.load(img_path)
        if self.transform: # apply transform if it exists
            image = self.transform(image)
        return image

def load_breast_dataset(folder_dir:Path, image_type='png'):
    """given a folder with images, create a breast_dataset object

    Args:
        folder_dir (Path): path of the folder with the images

    Returns:
        dataset: dataset object
    """
    # get directory name
    folder_name = folder_dir.name
    # check if the csv file with the filenames already exists
    csv_path = folder_dir.parent.parent / 'filenames' / f'{folder_name}.csv'
    if not csv_path.exists(): # if not, create it
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for filename in os.listdir(folder_dir):
                if filename.endswith(f".{image_type}"):
                    writer.writerow([filename])
    # now we can create the dataset
    if image_type == 'png':
        dataset = breast_dataset(csv_path, images_dir= folder_dir)
    elif image_type == 'pt':
        dataset = breast_dataset_latents(csv_path, images_dir= folder_dir)
    
    return dataset    
