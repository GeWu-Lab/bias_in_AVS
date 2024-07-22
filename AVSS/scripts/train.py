import os.path

import torch
import numpy as np
from utils import pyutils
from utils import utility
from utils import miou_fscore
import sys
import random

sys.path.append('..')
from scripts.save_mask import save_batch_raw_mask

import torch.nn.functional as F

avg_meter_miou = pyutils.AverageMeter('miou')
avg_meter_F = pyutils.AverageMeter('F_score')
random.seed(3407)

def train(model, train_loader, optimizer, idx_ep, args, ddpm):
    print('>>> Train.')
    model.train()
    
    losses = []
    n_steps = ddpm.n_steps

    for batch_idx, batch_data in enumerate(train_loader):
        # print(len(mask_recs), len(img_recs), audio_recs.shape)  # [5, 5, (1, 5, 128)]
        # print(len(batch_data))
        current_batch_size = len(batch_data["mask_recs"])
        t = torch.randint(0, n_steps, (current_batch_size, )).to(args.device)
        loss_vid, vid_preds = model(batch_data, idx_ep,time_step=t,ddpm=ddpm)
        # print(vid_preds[0])
        # print(img_recs.shape, audio_recs.shape, mask_recs.shape)
        
        loss_vid = torch.mean(torch.stack(loss_vid))
        optimizer.zero_grad()
        loss_vid.backward()
        # for name, param in model.named_parameters():
        #     if "class_proj" in name and param.requires_grad and param.grad is not None:
        #         print(name,param.grad)
        optimizer.step()
        
        print(f'loss_{idx_ep}_{batch_idx}: {loss_vid.item()}', end='\r')
    
    losses.append(loss_vid.item())
    return np.mean(losses)
    
def test(model, test_loader, optimizer, idx_ep, args, ddpm, save_mask=None):
    # print("testing with ",args.data_ver)
    if args.data_ver == "v2":
        N_CLASSES = 71
    else:
        N_CLASSES = 22
    model.eval()
    miou_pc = torch.zeros((N_CLASSES)) # miou value per class (total sum)
    Fs_pc = torch.zeros((N_CLASSES)) # f-score per class (total sum)
    cls_pc = torch.zeros((N_CLASSES)) # count per class
    t = ddpm.n_steps-1
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            _, vid_preds = model(batch_data,time_step=t,ddpm=ddpm)
            if not vid_preds:
                continue
            vid_preds_t = torch.stack(vid_preds, dim=0).squeeze().cuda()  # [5, 720, 1280] = [1*frames, H, W]
            vid_masks_t = torch.stack(batch_data["mask_recs"], dim=0).squeeze().cuda().float()
            vid_preds_t = F.interpolate(vid_preds_t.unsqueeze(1).float(), size=(224,224), mode='nearest').squeeze(1)
            vid_masks_t = F.interpolate(vid_masks_t.unsqueeze(1).float(), size=(224,224), mode='nearest').squeeze(1)
            # print(vid_preds_t.shape, vid_masks_t.shape)
            miou = utility.mask_iou(vid_preds_t, vid_masks_t)  # mIoU
            if save_mask:
                from datetime import datetime
                current_time = datetime.now()
                if not os.path.exists(save_mask):
                    os.mkdir(save_mask)
                # print(torch.unique(vid_preds_t))
                if len(torch.unique(vid_preds_t))>=2:
                    save_batch_raw_mask(save_mask+batch_data["vid"][0], vid_preds_t)
            # print(vid_preds_t.shape, torch.max(vid_preds_t), torch.min(vid_preds_t))
            _miou_pc, _fscore_pc, _cls_pc, _ = miou_fscore(vid_preds_t, vid_masks_t, N_CLASSES,len(vid_masks_t))
            miou_pc += _miou_pc
            cls_pc += _cls_pc
            # compute f-score, F-measure
            Fs_pc += _fscore_pc

            avg_meter_miou.add({'miou': miou})

            F_score = utility.Eval_Fmeasure(vid_preds_t, vid_masks_t, './logger', device=args.device)  # F_score
            avg_meter_F.add({'F_score': F_score})
            
            print(f'[te] loss_{idx_ep}_{batch_idx}: miou={miou:.03f} | F={F_score:.03f}', end='\r')

    miou_pc = miou_pc / cls_pc
    print(f"[miou] {torch.sum(torch.isnan(miou_pc)).item()} classes ({torch.where(cls_pc==0)}) are not predicted in this batch")
    miou_pc[torch.isnan(miou_pc)] = 0
    miou = torch.mean(miou_pc).item()
    miou_noBg = torch.mean(miou_pc[1:]).item()
    f_score_pc = Fs_pc / cls_pc
    print(f"[fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes ({torch.where(cls_pc==0)}) are not predicted in this batch")
    f_score_pc[torch.isnan(f_score_pc)] = 0
    f_score = torch.mean(f_score_pc).item()
    f_score_noBg = torch.mean(f_score_pc[1:]).item()

    try:
        miou_epoch = (avg_meter_miou.pop('miou'))
        F_epoch = (avg_meter_F.pop('F_score'))
    except:
        miou_epoch = torch.tensor([0])
        F_epoch = 0
    res = {
        'miou_ori': miou_epoch.item(),
        'fscore_ori': F_epoch,
        'miou': miou,
        'miou_nb': miou_noBg,
        'fscore': f_score,
        'fscore_nb': f_score_noBg,
    }

    return res
