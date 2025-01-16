import torch
import h5py
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from dataset import StorySalonDataset
from model.pipeline_simcse import StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from model.unet_2d_condition import UNet2DConditionModel

# 动态追加保存Tensor数据的函数
def append_tensor_to_h5(tensor, filename, dataset_name='tensor_data'):
    # 以读写模式打开文件，如果文件不存在则创建
    with h5py.File(filename, 'a') as f:
        # 如果数据集不存在，则创建数据集
        if dataset_name not in f:
            max_shape = (None,) + tensor.shape[1:]  # None 表示可扩展维度
            chunks = (1,) + tensor.shape[1:]  # 每次追加的数据块形状
            dset = f.create_dataset(dataset_name, data=tensor, maxshape=max_shape, chunks=chunks)
        else:
            # 获取数据集
            dset = f[dataset_name]
            # 扩展数据集的形状以容纳新的数据
            dset.resize(dset.shape[0] + tensor.shape[0], axis=0)
            # 追加数据
            dset[-tensor.shape[0]:] = tensor

"""
# 创建Tensor
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])

# 追加保存Tensor
append_tensor_to_h5(tensor1, 'tensor_data.h5')
append_tensor_to_h5(tensor2, 'tensor_data.h5')

# 读取HDF5文件中的Tensor数据
with h5py.File('tensor_data.h5', 'r') as f:
    loaded_tensor = torch.tensor(f['tensor_data'][:])
"""

# 打印加载的Tensor以验证
#print(loaded_tensor)

"""
def save_data():
    val_dataset = StorySalonDataset(root="/mnt/lustre/pengjie/data/StorySalon/", dataset_name='test')
    print(val_dataset.__len__())
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    pretrained_model_path = "/home/pengjie/StoryGen_c/stage2_log/eeee0.2_/checkpoint_50000/" #
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
    pipeline.set_progress_bar_config(disable=True)

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    val_data_yielder = make_data_yielder(val_dataloader)
    i = 0 # add
    while True:
        try:
            # 获取下一批数据
            val_batch = next(val_data_yielder)
            with autocast():
                if i>=4673:
                    i += 1
                    break
                elif -1<i<4673:
                    #self.prompts = batch["prompt"]
                    sample_seeds = torch.randint(0, 100000, (1,))
                    sample_seeds = sorted(sample_seeds.numpy().tolist())
                    prev_prompts = val_batch["ref_prompt"]
                    for idx, prompt in enumerate(tqdm(val_batch["prompt"], desc="idx 0 or 1")):
                        image = val_batch["image"][idx, :, :, :].unsqueeze(0)
                        ref_images = val_batch["ref_image"][idx, :, :, :, :].unsqueeze(0)
                        image = image.to(device='cuda')
                        ref_images = ref_images.to(device='cuda')
                        generator = []
                        for seed in sample_seeds:
                            generator_temp = torch.Generator(device='cuda')
                            generator_temp.manual_seed(seed)
                            generator.append(generator_temp)

                        sequence = pipeline(
                            stage='auto-regressive',
                            prompt=prompt,
                            image_prompt=ref_images,
                            prev_prompt=prev_prompts,
                            height=image.shape[2],
                            width=image.shape[3],
                            generator=generator,
                            num_inference_steps=40,
                            guidance_scale=7.0,
                            image_guidance_scale=3.5,
                            num_images_per_prompt=1,
                            alpha=0.2,
                        ).images
                    i += 1
                    break # add
                else:
                    i += 1
                    continue


        except StopIteration:
            # 当没有更多数据时，生成器会抛出 StopIteration 异常
            # 这时可以结束循环
            break

    #with h5py.File("myfile.hdf5", "w") as f:
        #f["data"] = data

if __name__ == "__main__":
    save_data()
"""

with h5py.File('picture_data/ours.h5', 'r') as f:
    loaded_tensor = torch.tensor(f['ours_val_text'][:])

print(loaded_tensor)
print(loaded_tensor.shape)