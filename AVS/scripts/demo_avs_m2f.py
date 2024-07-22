'''
# -*- encoding: utf-8 -*-
@File    :   demo_avs_m2f.py
@Time    :   2024/03
@Author  :   Peiwen Sun
@description   :   For bias in AVS
'''

import sys
import os
from datetime import datetime

# print(sys.path)

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from configs import args
from dataset import AVS
from models import AVS_BASE

from train import train, test
from logs.write_log import write_log
# print(args)
import wandb
import random
import numpy as np
data_ver = 'v1m'
log_name = f'{data_ver}.txt'
os.environ["WANDB_DISABLED"] = "False"


class DDPM():
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(model, device, ddpm):
    # ---- datasets
    train_dataset = AVS('train', data_ver)  # TODO: v1s, change its func name.
    val_dataset = AVS('val', data_ver)  # val
    test_dataset = AVS('test', data_ver)  # test

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    if args.val == 'val':
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    elif args.val == 'test' or args.val == 'test_in':
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    # ---- fine tune config. 
    params_names = [
        # 'decoder.mask_predictor',
        # 'queries_embedder',
        # 'queries_features',
        'audio_proj'
        'avs_adapt',
        'swin_adapt',
        'cross_a2t',
    ]
    tuned_num = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        for _n in params_names:
            if _n in name:
                param.requires_grad = True
                tuned_num += 1
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            write_log(f'Requires_grad: {name}', log_name)
            ...

    params1 = [{'params': [p for name, p in model.named_parameters() if p.requires_grad], 'lr': args.lr}]
    params = params1 # + params2
    optimizer = torch.optim.AdamW(params)

    
    train_losses = []
    m_s, f_s = [], []
    
    message = f'All: {sum(p.numel() for p in model.parameters()) / 1e6}\n'
    message += f'Tr: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}\n'
    write_log(message, log_name, tag='params.')

    wandb.init(
        project="m2f_v1m_class",
        config=args,
    )

    # model
    model = model.cuda()
    currentDateAndTime = datetime.now().strftime("%y%m%d_%H_%M_%S_%f")
    miou_max = 0
    for idx_ep in range(args.epochs):
        print(f'[Epoch] {idx_ep}')
        
        if args.train:
            model.train()
            loss_train = train(model, train_loader, optimizer, idx_ep, args,ddpm)
            wandb.log({'loss_train': loss_train})
            train_losses.append(loss_train)
        
        if args.val:
            model.eval()
            m, f = test(model, val_loader, optimizer, idx_ep, args,ddpm)
            m_s.append(m)
            f_s.append(f)
            wandb.log({'miou': m,
                      'f-score':f})
            tag = 'warm'
            folder = f"../place_holder/AVS_class/ckpt/{currentDateAndTime}/"
            quit()
            if not os.path.exists(folder):
                os.mkdir(folder)
            if m>miou_max:
                miou_max = m
                print("saved:",folder+f"{idx_ep}.pth")
                torch.save(model.state_dict(), folder+f"{idx_ep}.pth")  # TODO: ...
                ...
        print(f'train-losses: {train_losses} | miou: {m_s} | f-score{f_s}')


if __name__ == '__main__':

    print(vars(args))
    write_log(f'{vars(args)}', log_name, once=False)
    m2f_avs = AVS_BASE()
    write_log(m2f_avs, log_name, once=False)
    set_seed(3407)
    ckpt = "../preprocess/best_avs.ckpt"
    m2f_avs.load_state_dict(torch.load(ckpt), strict=False)

    ddpm = DDPM(device=args.device, n_steps=100)
    run(m2f_avs, device=args.device,ddpm=ddpm)
