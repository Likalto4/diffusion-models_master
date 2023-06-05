#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch
import gradio as gr
from gradio import Interface
from PIL import ImageOps

def main():
    # define model and load weights
    model_dir = 'Likalto4/fusion_healthy'
    pipe = StableDiffusionPipeline.from_pretrained( # sable diffusion pipe?
        model_dir,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # set gradio inputs
    vendor_dropdown = gr.Dropdown(
        choices=['Siemens', 'Hologic'],
        label="Vendor",
        value='Siemens',
        info="Choose the mammographic unit vendor",
    )
    view_dropdown = gr.Dropdown(
        choices=['CC', 'MLO'],
        label="View",
        value='MLO',
        info="Choose the mammogram view",
    )
    density_dropdown = gr.Dropdown(
        choices=['very low', 'low', 'high', 'very high'],
        label='Breast density',
        value='very high',
        info='Choose the breast density'
    )
    area_dropdown = gr.Dropdown(
        choices=['small', 'medium', 'big'],
        label='Breast area',
        value='small',
        info='Choose the breast area size'
    )
    priority_dropdown = gr.Dropdown(
        choices=['area', 'density', 'only area', 'only density'],
        label='Priority',
        value='area',
        info='Choose which feature to prioritize'
    )
    laterality_dropdown = gr.Dropdown(
        choices=['L', 'R'],
        label="Laterality",
        value='L',
        info="Choose the laterality",
    )
    negative_prompt_box = gr.Textbox(
        value='',
        label='Negative prompt',
        info='Describe which features to avoid in the generated mammogram'
    )
    guidance_slider = gr.Slider(
        minimum=0,
        maximum=20,
        step=1,
        value=6,
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
    seed_value = gr.Number(
        value=1338,
        label="Seed",
        info="Seed value",
        precision=0, # integer
)

    inputs = [vendor_dropdown,view_dropdown, density_dropdown, area_dropdown, laterality_dropdown,
              priority_dropdown,
              negative_prompt_box,
              guidance_slider, diffusion_slider,
              seed_checkbox, seed_value]

    def fn(vendor, view, density, area, laterality, priority, negative_prompt, guidance_scale, diffusion_steps, seed_checkbox, seed):
        """gradio inference function

        Args:
            prompt (str): input priompt for diffusion
            guidance_scale (int|float): guidance scale for text prompt
            diffusion_steps (int): number of steps in the Markov chain
            seed_checkbox (bool): whether to use seed for reproducibility

        Returns:
            PIL.Image: output of diffusion process
        """
        #internal HP
        num_samples = 1

        # prompt
        if priority == 'area':
            prompt = f'a {vendor} mammogram in {view} view with {area} area and {density} density'
        elif priority == 'density':
            prompt = f'a {vendor} mammogram in {view} view with {density} density and {area} area'
        elif priority == 'only area':
            prompt = f'a {vendor} mammogram in {view} view with {area} area'
        elif priority == 'only density':
            prompt = f'a {vendor} mammogram in {view} view with {density} density'
        else:
            raise ValueError('Invalid priority value')

        # seed
        if seed_checkbox:
            generator = torch.Generator(device='cuda')
            generator.manual_seed(seed)     # for reproducibility
        else:
            generator = None
        
        with torch.autocast("cuda"), torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512,
                generator=generator,
            ).images
        
        pil_output = image[0] # extract only one image
        if laterality=='R':
            # flip image
            pil_output = ImageOps.mirror(pil_output)

        return pil_output
    
    # define the interface
    iface = Interface(
        fn=fn,
        inputs=inputs,
        outputs="image",
        title="Fusion MAM-E: Generate mammograms",
        description="Generate mammograms using the available options.<br>"  + \
            "The generated images are healthy mammograms (No lesions).",
    )

    iface.queue()
    iface.launch(debug=False, share=False)

if __name__ == '__main__':
    main()