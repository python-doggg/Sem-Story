import os
import cv2
import random
from typing import Optional, Dict

from omegaconf import OmegaConf

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.cuda.amp import autocast

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import get_time_string, get_function_args
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline
from dataset import StorySalonDataset

from simcse import SimCSE # add


simcse_model = SimCSE("/home/share/models/sup-simcse-bert-base-uncased")
train_dataset = StorySalonDataset(root="/mnt/lustre/pengjie/data/StorySalon/", dataset_name='train') # root="./StorySalon/"
val_dataset = StorySalonDataset(root="/mnt/lustre/pengjie/data/StorySalon/", dataset_name='test') # root="./StorySalon/"

a = 0
b = 0

for batch in val_dataset:
    #print(batch)
    prompt = batch["prompt"]
    prev_prompts = batch["ref_prompt"]
    prev_prompt_ids = []
    similarities = []
    for prev_prompt in prev_prompts:
        similarity = simcse_model.similarity(prompt, prev_prompt)
        similarities.append(similarity)
    print(prompt)
    a = max(similarities) if a < max(similarities) else a
    b = min(similarities) if b > min(similarities) else b
    with open("/home/pengjie/StoryGen_c/datasets_sim/storysalon_val.txt", mode="a", encoding="utf-8") as f:
        #f.write(prompt + "\n")
        #f.write("\n")

        for i in range(len(prev_prompts)):
            #f.write(prev_prompts[i])
            #f.write("\t")
            f.write(str(similarities[i]))
            f.write("\t")

        #f.write(prev_prompts)
        #f.write("\n")
        #f.write(similarities)
        f.write("\n")
    #break
print("max", a)
print("min", b)