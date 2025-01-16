import os
import torch
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def draw_text(prompt):
    # 创建一个与图像同宽但高度任意的白色画布
    canvas = torch.ones(3, 512, 512) # (batch_size, height, width)，1是白色

    # 使用PIL将文本绘制到画布上
    # 首先将PyTorch张量转换为PIL图像
    canvas_pil = transforms.ToPILImage()(canvas).convert("RGB") # 将张量canvas转换为PIL图像
    draw = ImageDraw.Draw(canvas_pil)

    # 设置字体和大小
    # 如果没有以下字体文件，请替换为系统上可用的字体文件路径
    font = ImageFont.truetype("/usr/share/fonts/smc/Meera.ttf", 36)
    # font = ImageFont.truetype(size=36)

    # 写入文本
    # text_width, text_height = draw.textlength(prompt, font=font), 36
    # text_x = (canvas_pil.width - text_width) // 2
    # text_y = (canvas_pil.height - text_height) // 2
    #print("prompt:",prompt)
    t = len(prompt) // 30
    for i in range(t):
        draw.text((0, 50+i*40), prompt[30*i:30*(i+1)], font=font, fill="black")
    draw.text((0, 50+t*40), prompt[30*t:], font=font, fill="black")

    #prompt = list(prompt.split(","))
    # print("change prompt:",prompt)
    '''
    draw.text((0, 467), prompt[0], font=font, fill="black") ####已改
    draw.text((0, 497), prompt[1], font=font, fill="black")
    draw.text((0, 527), prompt[2], font=font, fill="black")
    '''

    # 将PIL图像转换回PyTorch张量
    canvas = transforms.ToTensor()(canvas_pil)

    return canvas


if __name__ == "__main__":
    main_path = "/home/pengjie/StoryGen_c/test_all/1207more2_eeee0.2__checkpoint_50000/"
    save_path = main_path + "visual/"

    gt_path = main_path + "gt/"
    ot_path = main_path + "output/"
    ri_path = main_path + "ref_i/"
    rp_path = main_path + "ref_p/"
    txt_path = main_path + "txt/"


    name_all = sorted(os.listdir(txt_path))
    for name in name_all:
        print(name)
        txt = open(txt_path + name, "r").read()
        txt_canvas = draw_text(txt)

        with open(rp_path + name, "r") as f:
            ls = f.readlines()
            for i in range(len(ls)):
                if i == 0:
                    ref0_canvas = draw_text(ls[i])
                elif i == 1:
                    ref1_canvas = draw_text(ls[i])
                else:
                    ref2_canvas = draw_text(ls[i])

        result1 = torch.cat((ref0_canvas, ref1_canvas, ref2_canvas, txt_canvas, txt_canvas), dim=2)

        ref0 = Image.open(ri_path + name.split(".")[0] + "_ref_0.png")
        ref1 = Image.open(ri_path + name.split(".")[0] + "_ref_1.png")
        ref2 = Image.open(ri_path + name.split(".")[0] + "_ref_2.png")
        gt = Image.open(gt_path + name.split(".")[0] + ".png")
        ot = Image.open(ot_path + name.split(".")[0] + ".png")

        '''
        ref0 = torch.from_numpy(np.array(ref0)).permute(2, 0, 1)
        ref1 = torch.from_numpy(np.array(ref1)).permute(2, 0, 1)
        ref2 = torch.from_numpy(np.array(ref2)).permute(2, 0, 1)
        gt = torch.from_numpy(np.array(gt)).permute(2, 0, 1)
        ot = torch.from_numpy(np.array(ot)).permute(2, 0, 1)
        '''

        ref0 = transforms.ToTensor()(ref0)
        ref1 = transforms.ToTensor()(ref1)
        ref2 = transforms.ToTensor()(ref2)
        gt = transforms.ToTensor()(gt)
        ot = transforms.ToTensor()(ot)

        result2 = torch.cat((ref0, ref1, ref2, gt, ot), dim=2)

        #print(result1.shape)
        #print(result2.shape)
        result = torch.cat((result1, result2), dim=1)
        if os.path.isfile(save_path + name.split(".")[0] + ".png"):
            continue
        else:
            save_image(result, save_path + name.split(".")[0] + ".png", nrow=1, normalize=True, value_range=(-1, 1))



