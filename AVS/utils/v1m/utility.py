import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

import os
import shutil
# import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
import pdb
from torchvision import transforms

# logger = logging.getLogger(__name__)


def save_checkpoint(state, epoch, is_best, checkpoint_dir='./models', filename='checkpoint', thres=100):
    """
    - state
    - epoch
    - is_best
    - checkpoint_dir: default, ./models
    - filename: default, checkpoint
    - freq: default, 10
    - thres: default, 100
    """
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if epoch >= thres:
        file_path = os.path.join(checkpoint_dir, filename + '_{}'.format(str(epoch)) + '.pth.tar')
    else:
        file_path = os.path.join(checkpoint_dir, filename + '.pth.tar')
    torch.save(state, file_path)
    # logger.info('==> save model at {}'.format(file_path))

    if is_best:
        cpy_file = os.path.join(checkpoint_dir, filename + '_model_best.pth.tar')
        shutil.copyfile(file_path, cpy_file)
        # logger.info('==> save best model at {}'.format(cpy_file))


def mask_iou_224(pred, target, eps=1e-7):
    assert len(pred.shape) == 3 and pred.shape == target.shape

    temp_pred = torch.sigmoid(pred)
    # temp_pred = F.interpolate(temp_pred.unsqueeze(0), size=(224, 224), mode='nearest', align_corners=False).squeeze(0)
    # temp_pred = F.interpolate(temp_pred.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    # target = F.interpolate(target.unsqueeze(0).float(), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0).long()

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    # temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.4).int()
    # plt.imshow(temp_pred[0].cpu())
    # plt.savefig('temp_pred_224.png')
    # plt.imshow(target[0].cpu())
    # plt.savefig('target_224.png')
    # input('224.')
    # print(pred.shape, target.shape, ' <<< file: utility')  # [224, 224]

    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou

def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param:
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # return mask_iou_224(pred, target, eps=1e-7)
    NF, bsz, H, W = pred.shape
    pred = pred.view(NF*bsz, H, W)
    target = target.view(NF*bsz, H, W)
    # print(pred.shape, target.shape)
    assert len(pred.shape) == 3 and pred.shape == target.shape

    # print(torch.max(pred), torch.min(pred))
    # print(torch.max(target), torch.min(target))
    # input('max-min')

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.4).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, device='cuda:0'):
    if device.startswith('cuda'):
        prec, recall = torch.zeros(num).to(y_pred.device), torch.zeros(num).to(y_pred.device)
        thlist = torch.linspace(0, 1 - 1e-10, num).to(y_pred.device)
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, measure_path, pr_num=255, device='cuda:0'):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # print('=> eval [FMeasure]..')
    pred = torch.sigmoid(pred)  # =======================================[important]
    # print(pred)
    # input()
    N = pred.size(0)
    beta2 = 0.1
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    # fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
    # print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num, device=device)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)
    # fLog.close()

    return score.max().item()


def save_mask(pred_masks, save_base_path, video_name_list):
    # pred_mask: [bs*5, 1, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()

    pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx]  # [5, 1, 224, 224]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png" % (video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')


def save_raw_img_mask(anno_file_path, raw_img_base_path, mask_base_path, split='test', r=0.5):
    df = pd.read_csv(anno_file_path, sep=',')
    df_test = df[df['split'] == split]
    count = 0
    for video_id in range(len(df_test)):
        video_name = df_test.iloc[video_id][0]
        raw_img_path = os.path.join(raw_img_base_path, video_name)
        for img_id in range(5):
            img_name = "%s.mp4_%d.png" % (video_name, img_id + 1)
            raw_img = cv2.imread(os.path.join(raw_img_path, img_name))
            mask = cv2.imread(os.path.join(mask_base_path, 'pred_masks', video_name, "%s_%d.png" % (video_name, img_id)))
            # pdb.set_trace()
            raw_img_mask = cv2.addWeighted(raw_img, 1, mask, r, 0)
            save_img_path = os.path.join(mask_base_path, 'img_add_masks', video_name)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_img_path, img_name), raw_img_mask)
        count += 1
    print(f'count: {count} videos')


if __name__ == "__main__":
    pred1 = torch.ones(4, 5, 5)
    target1 = torch.ones(4, 5, 5)
    iou1 = mask_iou(pred1, target1)

    pred2 = torch.zeros(4, 5, 5)
    target2 = torch.zeros(4, 5, 5)
    iou2 = mask_iou(pred2, target2)

    pred3 = torch.zeros(4, 5, 5)
    target3 = torch.ones(4, 5, 5)
    iou3 = mask_iou(pred3, target3)

    pred4 = torch.ones(4, 5, 5)
    target4 = torch.zeros(4, 5, 5)
    iou4 = mask_iou(pred4, target4)

    pred5 = torch.ones(4, 5, 5)
    target5 = torch.ones(4, 5, 5)
    target5[:2] = torch.zeros(5, 5)
    iou5 = mask_iou(pred5, target5)

    pred6 = torch.zeros(4, 5, 5)
    pred6[:2] = torch.ones(5, 5)
    target6 = torch.ones(4, 5, 5)
    iou6 = mask_iou(pred6, target6)

    one_mask = torch.randn(224, 224)
    one_mask = (torch.sigmoid(one_mask) > 0.5).numpy().astype(np.uint8)
    one_real_mask = one_mask * 255

    pdb.set_trace()
