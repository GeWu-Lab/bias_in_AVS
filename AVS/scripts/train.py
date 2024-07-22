import os.path

import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from utils import pyutils
from utils import utility 
from PIL import Image
import random

avg_meter_miou = pyutils.AverageMeter('miou')
avg_meter_F = pyutils.AverageMeter('F_score')
random.seed(3407)

def train(model, train_loader, optimizer, idx_ep, args, ddpm):
    print('>>> Train.')
    model.train()
    n_steps = ddpm.n_steps
    losses = []
    for batch_idx, batch_data in enumerate(train_loader):
        img_recs, audio_recs, mask_recs, image_sizes, vid = batch_data
        current_batch_size = 1
        # print(len(mask_recs), len(img_recs), audio_recs.shape)  # [5, 5, (1, 5, 128)]
        t = torch.randint(0, n_steps, (current_batch_size, )).to(args.device)
        loss_vid, _ = model(batch_data,idx_ep,t,ddpm)
        # print(img_recs.shape, audio_recs.shape, mask_recs.shape)si
        loss_vid = torch.mean(torch.stack(loss_vid))
        optimizer.zero_grad()
        loss_vid.backward()
        optimizer.step()

        print(f'loss_{idx_ep}_{batch_idx}: {loss_vid.item()} | ', end='\r')

    losses.append(loss_vid.item())
    return np.mean(losses)

def save2img(bool_tensors, name):
    # 定义图像尺寸
    bool_tensors = torch.sigmoid(bool_tensors)
    bool_tensors = (bool_tensors > 0.4)
    bool_tensors = bool_tensors.cpu().detach()
    image_width = bool_tensors[0].shape[1]
    image_height = bool_tensors[0].shape[0]

    # 创建图像对象，并将每个布尔张量转换为对应的黑白像素值
    images = [Image.new('L', (image_width, image_height)) for _ in range(5)]
    for i in range(5):
        image = images[i]
        bool_tensor = bool_tensors[i]
        for y in range(image_height):
            for x in range(image_width):
                pixel_value = 255 if bool_tensor[y][x] else 0
                image.putpixel((x, y), pixel_value)

    # 保存图像
    for i in range(5):
        image = images[i]
        if not os.path.exists(f'../place_holder/more_frame_AVS_video/output_10/{name[0]}'):
            os.mkdir(f'../place_holder/more_frame_AVS_video/output_10/{name[0]}')
        image.save(f'../place_holder/more_frame_AVS_video/output_10/{name[0]}/{name[0]}_{i+5*3}.png')


def test(model, test_loader, optimizer, idx_ep, args, ddpm):
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # if batch_idx >= 64*5:
            #     break
            t = ddpm.n_steps-1
            img_recs, audio_recs, mask_recs, image_sizes, vid = batch_data
            # print(len(mask_recs), len(img_recs), audio_recs.shape)  # [5, 5, (1, 5, 128)]
            _, vid_preds = model(batch_data,idx_ep,t,ddpm)

            vid_preds_t = torch.stack(vid_preds, dim=0).squeeze().cuda()  # [5, 720, 1280] = [1*frames, H, W]
            vid_masks_t = torch.stack(mask_recs, dim=0).squeeze().cuda()
            # print(vid_preds_t)
            # save2img(vid_preds_t,vid)
            # print(vid_preds_t.shape,vid_masks_t.shape)
            # vid_preds_t = vid_preds_t.unsqueeze(1)
            miou = utility.mask_iou(vid_preds_t.unsqueeze(1), vid_masks_t.unsqueeze(1))  # mIoU
            avg_meter_miou.add({'miou': miou})

            F_score = utility.Eval_Fmeasure(vid_preds_t, vid_masks_t, './logger', device=args.device)  # F_score
            avg_meter_F.add({'F_score': F_score})
            
            print(f'[te] loss_{idx_ep}_{batch_idx}: miou={miou:.03f} | F={F_score:.03f} |  ', end='\r')
    
    miou_epoch = (avg_meter_miou.pop('miou'))
    F_epoch = (avg_meter_F.pop('F_score'))
    
    return miou_epoch.item(), F_epoch

def save_fig(_p, _pred, _img):
    # print(_pred.shape)
    plt.tight_layout(pad=0)
    plt.imshow(_img)
    # plt.colorbar()
    plt.imshow(_pred.squeeze().cpu(), alpha=0.4, cmap='rainbow')
    # plt.imshow(upscaled_imemb.squeeze().cpu(), alpha=0.3, cmap='rainbow')
    plt.axis('off')
    plt.savefig(_p, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    # input('.')
    