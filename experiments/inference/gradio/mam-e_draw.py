#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
import torch
from datasets_local.metadata import subset_csv
import pandas as pd
from PIL import Image
import gradio as gr
from gradio import Interface

def main():
    # define model and load weights
    model_dir='Likalto4/mammo_lesion-inpainting'
    pipe = DiffusionPipeline.from_pretrained(
        model_dir,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()


    ### Metadata and paths preparation ###
    # define folder of input images
    input_images_name = 'breast40k_RGB_healthy'
    input_images_folder_path = repo_path / 'data/images' / input_images_name
    reference_folder_path = repo_path / 'data/metadata/metadata_Hologic.csv'
    # define matadata path and create file if needed
    metadata_path = subset_csv(input_images_folder_path, reference_folder_path) # get metadata csv of the subset
    # read metadata
    metadata = pd.read_csv(metadata_path, header=0)

    ### get input image ###
    im_num = 5
    # get one image example
    image_id = metadata.iloc[im_num]['image_id']
    image_file = image_id + '.png'
    image_path = input_images_folder_path / image_file
    # read image using pil
    input_im = Image.open(image_path)


    ### fn function ###

    # sketchpad with breast image as background
    sketch = gr.Image(value=input_im, type='pil', interactive=True, tool='sketch', invert_colors=True, image_mode='L').style(height=700, width=700)
    guidance_slider = gr.Slider(
        minimum=0,
        maximum=10,
        step=1,
        value=3,
        label="Guidance scale",
        info="The guidance scale controls how much the model should pay attention to the text promt.",
    )
    diffusion_slider = gr.Slider(
        minimum=10,
        maximum=50,
        step=1,
        value=24,
        label="inference diffusion steps",
        info="The number of steps in the denoising diffusion process.",
    )
    seed_checkbox = gr.Checkbox(
        label="Use seed",
        value=True,
        info="Use seed for reproducibility",
    )

    def fn(sketch, guidance_scale, diffusion_steps, seed_checkbox):
        # get mask from input sketch
        mask = sketch['mask']
        ### generate images ###
        #internal HP
        prompt = "a mammogram with a lesion"
        negative_prompt = ""
        num_samples = 1
        if seed_checkbox:
            generator = torch.Generator(device='cuda')
            seed = 1337 # for reproducibility
            generator.manual_seed(seed)    
        else:
            generator = None
        

        with torch.autocast("cuda"), torch.inference_mode():
            image = pipe(
                prompt=prompt,
                image=input_im,
                mask_image=mask, # mask coming from the sketchpad
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                generator=generator,
            ).images

        return image[0]

    # define the interface
    iface = Interface(
        fn=fn,
        inputs=[sketch, guidance_slider, diffusion_slider, seed_checkbox],
        outputs="image",
        title="MAM-E drawing tool",
        description="Draw a lesion",
    )

    iface.launch(debug=False, share=True)

if __name__ == '__main__':
    main()