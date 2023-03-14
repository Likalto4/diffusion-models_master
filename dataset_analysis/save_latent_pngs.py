from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

#Libraries
import yaml
import torch
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from diffusers import AutoencoderKL
from datasets_local.datasets import load_breast_dataset
import pandas as pd

 

def main():

    # GPU and config
    selected_gpu = 0 #select the GPU to use
    device = torch.device("cuda:" + str(selected_gpu) if torch.cuda.is_available() else "cpu")
    print(f'The device is: {device}\n')
    # load the config file
    with open('config_latent.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    ### 1. Dataset loading and preprocessing
    # Dataset loading
    data_dir = repo_path / config['processing']['dataset']
    # dataset name
    dataset_name = data_dir.name
    dataset = load_breast_dataset(data_dir)
    # Define data augmentations
    class ToFloat32Tensor(object):
        """
        Converts a PIL Image to a PyTorch tensor with dtype float32, and normalises it.
        """
        def __call__(self, image):
            # Convert PIL Image to PyTorch tensor with dtype float32
            tensor = ToTensor()(image).float()/config['processing']['normalisation_value']
            return tensor

    preprocess = Compose(
        [
            Resize(config['processing']['resolution'], interpolation= InterpolationMode.BILINEAR), #getattr(InterpolationMode, config['processing']['interpolation'])),  # Smaller edge is resized to 256 preserving aspect ratio
            CenterCrop(config['processing']['resolution']),  # Center crop to the desired squared resolution
            ToFloat32Tensor(),  # Convert to tensor (0, 1)
            Normalize(mean=[0.5], std=[0.5]),  # Map to (-1, 1) as a way to make data more similar to a Gaussian distribution
        ]
    )
    #set the transform function to the dataset
    dataset.set_transform(preprocess)
    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers= config['processing']['num_workers'], shuffle=False # need order
    )
    # VAE
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device)

    # create the saving folder if it does not exist
    saving_directory = repo_path / 'data/images'/f'{dataset_name}_latents'
    if not saving_directory.exists():
        saving_directory.mkdir(parents=False)
    for i, batch in enumerate(tqdm(train_dataloader)):
        batch = batch.to(device)
        # read csv file with the filenames
        csv_path = repo_path / 'data/filenames'/f'{dataset_name}.csv'
        names = pd.read_csv(csv_path, header=None)
        name = names.iloc[i].item() # name of the image

        # extract vae version
        batch = batch.expand(-1, 3, -1, -1) # expand the batch to have three channels
        latents = vae.encode(batch).latent_dist.sample() # sample from the latent distribution
        latents = latents * vae.config.scaling_factor # scale the latents so they are around -1 and 1 (but not exactly)

        # save latents as torch tensor withoput gradients
        latents = latents.detach().cpu()[0]
        torch.save(latents, saving_directory / f'{Path(name).stem}.pt')

if __name__ == '__main__':
    main()