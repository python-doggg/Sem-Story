import os
import cv2
import random
import numpy as np
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
from model.pipeline_simcse import StableDiffusionPipeline # _simcse
from dataset import StorySalonDataset
from transformers import AutoProcessor, AutoModel

logger = get_logger(__name__)


def calc_probs(processor, model, prompt, images):
    # preprocess
    image_inputs = processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt").to(
        'cuda')
    text_inputs = processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to('cuda')

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()


class SampleLogger:
    def __init__(
            self,
            logdir: str,
            subdir: str = "sample",
            stage: str = 'auto-regressive',
            num_samples_per_prompt: int = 10, # yuan 1，论文里是10，代码里也是10
            num_inference_steps: int = 40,
            guidance_scale: float = 7.0,
            image_guidance_scale: float = 3.5,
    ) -> None:
        self.stage = stage
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = logdir#self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir, exist_ok=True)

    def log_sample_images(
            self, batch, pipeline: StableDiffusionPipeline, device: torch.device, processor, model, i, alpha
    ):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        self.prev_prompts = batch["ref_prompt"]

        txtdir = os.path.join(self.logdir, "txt")
        os.makedirs(txtdir, exist_ok=True)

        gtdir = os.path.join(self.logdir, "gt")
        os.makedirs(gtdir, exist_ok=True)

        outputdir = os.path.join(self.logdir, "output")
        os.makedirs(outputdir, exist_ok=True)

        ridir = os.path.join(self.logdir, "ref_i")
        os.makedirs(ridir, exist_ok=True)

        rpdir = os.path.join(self.logdir, "ref_p")
        os.makedirs(rpdir, exist_ok=True)

        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            #print("self.prompts", self.prompts)
            #print("idx", idx)
            #continue
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            ref_images = batch["ref_image"][idx, :, :, :, :].unsqueeze(0)
            image = image.to(device=device)
            ref_images = ref_images.to(device=device)
            generator = []
            for seed in self.sample_seeds:
                generator_temp = torch.Generator(device=device)
                generator_temp.manual_seed(seed)
                generator.append(generator_temp)
            sequence = pipeline(
                stage=self.stage,
                prompt=prompt,
                image_prompt=ref_images,
                prev_prompt=self.prev_prompts,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                image_guidance_scale=self.image_guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
                #alpha=alpha,
            ).images
            #print("len sequence", len(sequence))


            images = []
            #print(len(sequence))
            for j, img in enumerate(sequence):
                images.append(img[0])
            scores = calc_probs(processor, model, prompt, images)
            index = np.argmax(scores)
            images[index].save(os.path.join(outputdir, f"{i}.png")) # _{sample_seeds[j]}

            image = (image + 1.) / 2.  # for visualization
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(gtdir, f"{i}.png"), image[:, :, ::-1] * 255) # _{seed}
            v_refs = []
            ref_images = ref_images.squeeze(0)
            for ref_image in ref_images:
                # v_ref = (ref_image + 1.) / 2. # for visualization
                v_ref = ref_image.permute(1, 2, 0).detach().cpu().numpy()
                v_refs.append(v_ref)
            for j in range(len(v_refs)):
                cv2.imwrite(os.path.join(ridir, f"{i}_ref_{j}.png"), v_refs[j][:, :, ::-1] * 255) # _{seed}

            with open(os.path.join(txtdir, f"{i}" + '.txt'), 'a') as f: # _{seed}
                f.write(batch['prompt'][idx])
                f.write('\n')

            with open(os.path.join(rpdir, f"{i}" + '.txt'), 'a') as f: # _{seed}
                for prev_prompt in self.prev_prompts:
                    f.write(prev_prompt[0])
                    f.write('\n')


            #for i, img in enumerate(sequence):
                #img[0].save(os.path.join(outputdir, f"{idx}_{sample_seeds[i]}.png"))


def test(
        pretrained_model_path: str,
        logdir: str,
        train_steps: int = 300,
        validation_steps: int = 1000,
        validation_sample_logger: Optional[Dict] = None,
        gradient_accumulation_steps: int = 30,  # important hyper-parameter
        seed: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        train_batch_size: int = 4,
        val_batch_size: int = 1,
        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        lr_warmup_steps: int = 0,
        use_8bit_adam: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        checkpointing_steps: int = 2000,
        alpha : float = 0.3,
):
    #print("alpha", alpha)
    args = get_function_args()
    time_string = get_time_string()
    ls = pretrained_model_path.split("/")
    # logdir += f"v_{time_string}" ###
    l = ls[-3] + "_" + ls[-2]
    #l = ls[-2]
    logdir += l
    #logdir = "./test_all/0.3_stage2_log0.3_240712-151307_checkpoint_50000/" # 20240714 add
    #logdir = "/home/pengjie/StoryGen_c/test_all/stage2_log_0.1_240716-191622_checkpoint_10000/"
    #logdir = "/home/pengjie/StoryGen_c/test_all/stage2_log_fun1_240719-195542_checkpoint_50000/"
    #t = logdir
    #v_s_l = validation_sample_logger

    #for k in range(100):
        #l = "ablation/"+"1222more" + str(k+1) +"_" + ls[-3] + "_" + ls[-2]
        # l = ls[-2]
        #logdir += l
        #logdir = t + l

    accelerator = Accelerator( # 从此处开始全部\tab
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)

    if seed is not None:
        set_seed(seed)

    processor_name_or_path = "/home/share/models/CLIP-ViT-H-14-laion2B-s32B-b79K/"
    model_pretrained_name_or_path = "/home/share/models/PickScore_v1/"
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    val_dataset = StorySalonDataset(root="/mnt/lustre/pengjie/data/StorySalon/", dataset_name='test')
    print(val_dataset.__len__())
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

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

    #if validation_sample_logger is not None and accelerator.is_main_process: # yuan
        #validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir) # yuan

    if validation_sample_logger is not None and accelerator.is_main_process: # yuan
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir) # yuan v_l_s

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    val_data_yielder = make_data_yielder(val_dataloader)

    text_encoder.eval()
    vae.eval()
    unet.eval()

    i = 0 # add 从此处开始全部\tab
    while True:
        try:
            # 获取下一批数据
            val_batch = next(val_data_yielder)
            with autocast():
                if i>=4673:
                    i += 1
                    break
                elif 3507<i<4673: # 1565 3125 4673; 1172 2341 3508 4673
                    validation_sample_logger.log_sample_images(
                        batch=val_batch,
                        pipeline=pipeline,
                        device=accelerator.device,
                        processor=processor,
                        model=model,
                        i=i,
                        alpha=alpha,
                    )
                    i += 1

                else:
                    i += 1
                    continue


        except StopIteration:
            # 当没有更多数据时，生成器会抛出 StopIteration 异常
            # 这时可以结束循环
            break





if __name__ == "__main__":
    config = "./config/test_all.yml"
    test(**OmegaConf.load(config))