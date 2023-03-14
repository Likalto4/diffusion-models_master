#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None
exp_path = Path.cwd().resolve() # path to the experiment folder

from datasets_local.datasets import load_breast_dataset
#Libraries
import yaml
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import datasets, diffusers
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
)    
from diffusers import DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
import logging
from accelerate.logging import get_logger
from accelerate import Accelerator

# extra
from packaging import version


# Check the diffusers version
check_min_version("0.15.0.dev0")

# set the logger
logger = get_logger(__name__, log_level="INFO") # allow from info level and above

def main():
    ### 0. Initial setups
    # load the config file
    config_path = exp_path / 'config_files/latent_32.yaml' # configuration file path (beter to call it from the args parser)
    with open(config_path) as file: # expects the config file to be in the same directory
        config = yaml.load(file, Loader=yaml.FullLoader)
    # define logging directory
    pipeline_dir = repo_path / config['saving']['local']['outputs_dir'] / config['saving']['local']['pipeline_name']
    logging_dir = pipeline_dir / config['logging']['dir_name']

    # start the accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation']['steps'],
        mixed_precision=config['training']['mixed_precision']['type'],
        log_with= config['logging']['logger_name'],
        logging_dir= logging_dir,
    )
    # define basic logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", # format of the log message. # name is the logger name.
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # show the accelerator state as first log message
    logger.info(accelerator.state, main_process_only=False)
    # set the level of verbosity for the datasets and diffusers libraries, depending on the process type
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    ### 1. Dataset loading and preprocessing
    # Dataset loading
    data_dir = repo_path / config['processing']['dataset']
    dataset = load_breast_dataset(data_dir, image_type='pt')
    logger.info(f"Dataset loaded with {len(dataset)} images") # show info about the dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['processing']['batch_size'], num_workers= config['processing']['num_workers'], shuffle=True
    )

    ### 2. Model definition
    model = UNet2DModel(
        sample_size=config['processing']['resolution'],  # the target image resolution
        in_channels=config['model']['in_channels'],  # the number of input channels, 3 for RGB images
        out_channels=config['model']['out_channels'],  # the number of output channels
        layers_per_block=config['model']['layers_per_block'],  # how many ResNet layers to use per UNet block
        block_out_channels=config['model']['block_out_channels'],  # More channels -> more parameters
        down_block_types= config['model']['down_block_types'],
        up_block_types=config['model']['up_block_types'],
    )
    # memory efficient attention for model
    if config['training']['enable_xformers_memory_efficient_attention']:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    ### 3. Training
    num_epochs = config['training']['num_epochs']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr= config['training']['optimizer']['learning_rate'], # learning rate of the optimizer
        betas= (config['training']['optimizer']['beta_1'], config['training']['optimizer']['beta_2']), # betas according to the AdamW paper
        weight_decay= config['training']['optimizer']['weight_decay'], # weight decay according to the AdamW paper
        eps= config['training']['optimizer']['eps'] # epsilon according to the AdamW paper
    )
    lr_scheduler = get_scheduler(
        name= config['training']['lr_scheduler']['name'], # name of the scheduler
        optimizer= optimizer, # optimizer to use
        num_warmup_steps= config['training']['lr_scheduler']['num_warmup_steps'] * config['training']['gradient_accumulation']['steps'],
        num_training_steps= (len(train_dataloader) * num_epochs), #* config['training']['gradient_accumulation']['steps']? # no because this changes every step
    )
    noise_scheduler = DDPMScheduler(
        beta_start=config['training']['noise_scheduler']['beta_start'],
        beta_end=config['training']['noise_scheduler']['beta_end'],
        num_train_timesteps=config['training']['noise_scheduler']['num_train_timesteps'],
        beta_schedule=config['training']['noise_scheduler']['beta_schedule'],
    )

    # prepare with the accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # init tracker (wand or TB) and save config file
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0] # get the name of the script
        accelerator.init_trackers(project_name=run) # intialize a run for all trackers
        accelerator.get_tracker('wandb').save(str(config_path)) if config['logging']['logger_name'] == 'wandb' else None # save the config file in the wandb run

    # global trackers
    total_batch_size = config['processing']['batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation']['steps'] # considering accumulated and distributed training
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['training']['gradient_accumulation']['steps']) # take into account the gradient accumulation (divide)
    max_train_steps = num_epochs * num_update_steps_per_epoch # total number of training steps

    logger.info('The training is starting...\n')
    logger.info(f'The number of examples is: {len(dataset)}\n')
    logger.info(f'The number of epochs is: {num_epochs}\n')
    logger.info(f'The number of batches is: {len(train_dataloader)}\n')
    logger.info(f'The batch size is: {config["processing"]["batch_size"]}\n')
    logger.info(f'The number of update steps per epoch is: {num_update_steps_per_epoch}\n')
    logger.info(f'The gradient accumulation steps is: {config["training"]["gradient_accumulation"]["steps"]}\n')
    logger.info(f'The total batch size (accumulated, multiprocess) is: {total_batch_size}\n')
    logger.info(f'Total optimization steps: {max_train_steps}\n')

    # global variables
    global_step = 0

    #### Training loop
    for epoch in range(num_epochs): # Loop over the epochs
        model.train()
        train_loss = [] # accumulated loss list
        pbar = tqdm(total=num_update_steps_per_epoch)
        pbar.set_description(f"Epoch {epoch}")
        for latents in train_dataloader: # Loop over the batches
            with accelerator.accumulate(model): # start gradient accumulation
                noise = torch.randn_like(latents) # Sample noise to add to the images
                bs = latents.shape[0] # batch size variable for later use
                # Sample a random timestep for each image
                timesteps = torch.randint( #create bs random integers from init=0 to end=timesteps, and send them to device (3rd thing in device)
                    low= 0,
                    high= noise_scheduler.num_train_timesteps,
                    size= (bs,),
                    device=latents.device ,
                ).long() #int64
                # Forward diffusion process: add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
                # Get the model prediction, #### This part changes according to the prediction type (e.g. epsilon, sample, etc.)
                noise_pred = model(noisy_images, timesteps).sample # sample tensor
                # Calculate the loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config['processing']['batch_size'])).mean()
                # append the loss to the train loss
                train_loss.append(avg_loss.item())
                
                # Backpropagate the loss
                accelerator.backward(loss) #loss is used as a gradient, coming from the accumulation of the gradients of the loss function
                if accelerator.sync_gradients: # gradient clipping
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['gradient_clip']['max_norm'])
                # Update
                optimizer.step() # update the weights
                lr_scheduler.step() # Update the learning rate
                optimizer.zero_grad() # reset the gradients
            #### gradient accumulation ends here
            
            # logging and checkpoint saving happens only if the gradients are synced
            if accelerator.sync_gradients:
                # Update the progress bar
                pbar.update(1)
                global_step += 1
                # take the mean of the accumulated loss
                train_loss = np.mean(train_loss)
                accelerator.log({"loss": train_loss, "log-loss": np.log(train_loss)}, step=global_step) #accumulated loss
                train_loss = [] # reset the train for next accumulation
                # Save the checkpoint
                if global_step % config['saving']['local']['checkpoint_frequency'] == 0: # if saving time
                    if accelerator.is_main_process: # only if in main process
                        save_path = pipeline_dir / f"checkpoint-{global_step}" # create the path
                        accelerator.save_state(save_path) # save the state
                        logger.info(f"Saving checkpoint to {save_path}") # let the user know
            # step logging
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(values=logs, step=global_step)
            pbar.set_postfix(**logs) # add to the end of the progress bar
            # Close the progress bar at the end of the epoch
        pbar.close()
        accelerator.wait_for_everyone() # wait for all processes to finish the epoch

        ##### 4. Saving the model and visual samples
        if accelerator.is_main_process: # only main process saves the model
            if epoch % config['logging']['images']['freq_epochs'] == 0 or epoch == num_epochs - 1: # if in image saving epoch or last one
                # create random noise
                latent_inf = torch.randn( # Use seed to denoise always the same images
                    config['logging']['images']['batch_size'], config['model']['in_channels'],
                    config['processing']['resolution'], config['processing']['resolution'],
                    generator=torch.manual_seed(4)
                ).to(accelerator.device)
                latent_inf *= noise_scheduler.init_noise_sigma # init noise is 1.0 in vanilla case
                # denoise images
                for t in tqdm(noise_scheduler.timesteps): # markov chain
                    latent_inf = noise_scheduler.scale_model_input(latent_inf, t) # # Apply scaling, no change in vanilla case
                    with torch.no_grad(): # predict the noise residual with the unet
                        noise_pred = model(latent_inf, t).sample
                    latent_inf = noise_scheduler.step(noise_pred, t, latent_inf).prev_sample # compute the previous noisy sample x_t -> x_t-1
                # log images
                if config['logging']['logger_name'] == 'tensorboard':
                    for i in range (4):
                        accelerator.get_tracker('tensorboard').add_images(
                            f"latent{i}", latent_inf[:,i:i+1], epoch
                        )
                elif config['logging']['logger_name'] == 'wandb':
                    for i in range (4): # log the 4 latent channels
                        accelerator.get_tracker('wandb').log(
                            {f"latent_{i}": [wandb.Image(latent_inf[b,i], mode='F') for b in range(config['logging']['images']['batch_size'])]},
                            step=global_step,
                        )
            # save model
            if epoch % config['saving']['local']['saving_frequency'] == 0 or epoch == num_epochs - 1: # if in model saving epoch or last one
                # create pipeline # unwrap the model
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline.save_pretrained(str(pipeline_dir))
                logger.info(f"Saving model to {pipeline_dir}")
    logger.info("Finished training!\n")
    # stop tracking
    accelerator.end_training()
    
if __name__ == "__main__":
    main()