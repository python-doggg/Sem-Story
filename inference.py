import os
from typing import Optional
from torchvision import transforms
from PIL import Image
from typing import List, Optional, Union

import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import get_time_string
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline_simcse import StableDiffusionPipeline # yuan model.pipeline

logger = get_logger(__name__)

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: Union[str, List[str]], ### yuan str
    ref_prompt: Union[str, List[str]],
    ref_image: Union[str, List[str]],
    num_inference_steps: int = 40,
    guidance_scale: float = 7.0,
    image_guidance_scale: float = 3.5,
    num_sample_per_prompt: int = 10,
    stage: str = "multi-image-condition", # ["multi-image-condition", "auto-regressive", "no"]
    mixed_precision: Optional[str] = "fp16" ,
):
    time_string = get_time_string()
    logdir += f"{time_string}"
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("StoryGen")

    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    ref_images= []
    for id in ref_image:
        r_image = Image.open(id).convert('RGB').resize((512, 512))
        r_image = transforms.ToTensor()(r_image)
        ref_images.append(np.ascontiguousarray(r_image))
    ref_images = torch.from_numpy(np.ascontiguousarray(ref_images)).float()
    for ref_image in ref_images:
        ref_image = ref_image * 2. - 1.
    ref_images = ref_images.unsqueeze(0)

    sample_seeds = torch.randint(0, 100000, (num_sample_per_prompt,)) # yuan num_sample_per_prompt # *len(prompt)
    sample_seeds = sorted(sample_seeds.numpy().tolist())
    
    generator = []
    for seed in sample_seeds:
        generator_temp = torch.Generator(device=accelerator.device)
        generator_temp.manual_seed(seed)
        generator.append(generator_temp)

    with torch.no_grad():
        output = pipeline(
            stage = stage,
            prompt = prompt,
            image_prompt = ref_images,
            prev_prompt = ref_prompt,
            height = 512,
            width = 512,
            generator = generator,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            image_guidance_scale = image_guidance_scale,
            num_images_per_prompt=num_sample_per_prompt,
        ).images
    
    images = []
    print("output len",len(output))
    print("sample_seeds len",len(sample_seeds))
    for i, image in enumerate(output):
        images.append(image[0])
        images[i].save(os.path.join(logdir, f"_{sample_seeds[i]}_output.png")) # yuan sample_seeds[i]


if __name__ == "__main__":

    pretrained_model_path = "/home/pengjie/StoryGen_c/stage2_log/eeee0.2_/checkpoint_50000/" #"/home/pengjie/StoryGen_c/stage2_log/_240702-213515/checkpoint_20000/" # '/home/pengjie/StoryGen_c/checkpoint_StorySalon/'
    logdir = "/home/pengjie/StoryGen_c/drawing/zero_pic/"
    num_inference_steps = 40
    guidance_scale = 7
    image_guidance_scale = 3.5
    num_sample_per_prompt = 10
    mixed_precision = "fp16"
    stage = 'auto-regressive' # ["multi-image-condition", "auto-regressive", "no"]
    
    #prompt = "The white cat is running after the black-haired man."
    '''
    prompt = "A red dragon is playing on the grass"
    prev_p = ["The red dragon"]#, "The red dragon"]
    ref_image = ["./data/image/00002.png"]#, "./data/image/00002.png"]
    '''
    #prompt = "The mischievous gray cat is sprinting full speed after the clever little mouse, its paws barely touching the ground as it stretches forward with determination, while the tiny mouse scurries ahead with a sly grin, effortlessly staying just out of reach. The scene is full of playful chaos, as the chase unfolds with exaggerated expressions and comical energy, capturing the timeless rivalry between the two iconic characters."
    #prev_p = ["This image features a joyful, anthropomorphic cartoon cat with sleek blue-gray fur and a lighter gray belly, characteristic of the classic Tom from the iconic animated series 'Tom and Jerry'. Tom is seen striking a dramatic, dance-like pose with one arm playfully stretched upward and his face expressing gleeful excitement. His mouth is wide open with a big smile, and his eyes are shut, as if he’s lost in a moment of celebration or triumph. The scene’s backdrop features a cozy, vintage indoor setting with muted green walls and simple wooden furniture, reflecting the hand-drawn animation style of the mid-20th century. This lighthearted image perfectly captures Tom’s lively personality and the timeless charm of the 'Tom and Jerry' cartoon series.", "This image features a cheerful cartoon mouse with a wide smile and an excited expression, standing in a dimly lit indoor space. The character has large, rounded ears, expressive eyes, and a small brown body with lighter beige tones on the chest and inner ears. The mouse's playful pose suggests movement, with its arms slightly extended and its tail curved behind. The scene is styled in a classic hand-drawn animation art typical of mid-20th-century cartoons. In the bottom right corner, the 'Tom and Jerry' logo is visible, indicating that this is a character from the famous animated series known for its timeless humor and slapstick comedy."]
    prompt = "A joyful, anthropomorphic cartoon cat with sleek blue-gray fur and a lighter gray belly is running after the mouse."
    prev_p = ["A joyful, anthropomorphic cartoon cat with sleek blue-gray fur and a lighter gray belly.", "A cheerful cartoon mouse with a wide smile and an excited expression, standing in a dimly lit indoor space."]
    ref_image = ["/home/pengjie/StoryGen_c/drawing/zero_pic/ref_im/tom_1.jpg", "/home/pengjie/StoryGen_c/drawing/zero_pic/ref_im/jerry_1.jpg"]#["./data/boy.jpg", ".data/whitecat1.jpg"]


    test(pretrained_model_path, 
         logdir, 
         prompt, 
         prev_p, 
         ref_image, 
         num_inference_steps, 
         guidance_scale, 
         image_guidance_scale, 
         num_sample_per_prompt,
         stage, 
         mixed_precision)

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py