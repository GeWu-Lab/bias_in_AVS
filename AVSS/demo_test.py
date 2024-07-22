import sys
import os
from datetime import datetime
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.config import args
from datasets_avss import AVS
from models import PmpAVS
from itertools import chain
import copy
from scripts.train import train, test
# print(args)
import wandb
import random
import numpy as np
os.environ['WANDB_DISABLED'] = 'true'

data_ver = 'v2'

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

def custom_collate(batch):

    mask_recs = [item['mask_recs'] for item in batch]
    image_size = [item['image_size'] for item in batch]
    vid = [item['vid'] for item in batch]
    feat_aud = [item['feat_aud'] for item in batch]
    feat_aud = torch.stack(feat_aud)
    pixel_values = [item['pixel_values'] for item in batch]
    pixel_values = torch.stack(pixel_values)
    pixel_mask = [item['pixel_mask'] for item in batch]
    pixel_mask = torch.stack(pixel_mask)
    class_labels = [item['class_labels'] for item in batch]
    mask_labels = [item['mask_labels'] for item in batch]
    res = {
        "mask_recs":mask_recs,
        "image_size":image_size,
        "vid":vid,
        "feat_aud":feat_aud,
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
    }
    return res

def run(model, device, ddpm=None):
    # ---- datasets
    num_workers_all=1
    if data_ver == "v2":
        train_dataset_v1s = AVS('train', "v1s")
        train_dataset_v1m = AVS('train', "v1m")
        train_dataset_v2 = AVS('train', "v2")
        train_loader_v1s = DataLoader(train_dataset_v1s, batch_size=1, shuffle=True, num_workers=num_workers_all, pin_memory=True,collate_fn=custom_collate)
        train_loader_v1m = DataLoader(train_dataset_v1m, batch_size=1, shuffle=True, num_workers=num_workers_all, pin_memory=True,collate_fn=custom_collate)
        train_loader_v2 = DataLoader(train_dataset_v2, batch_size=1, shuffle=True, num_workers=num_workers_all, pin_memory=True,collate_fn=custom_collate)
        
        train_loader = chain(train_loader_v1s,train_loader_v1m,train_loader_v2)
        if args.val == 'val':
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False,collate_fn=custom_collate)
        elif args.val == 'test' or args.val == 'test_in':
            test_dataset_v1s = AVS('test', "v1s")
            test_dataset_v1m = AVS('test', "v1m")
            test_dataset_v2 = AVS('test', "v2")
            val_loader_v1s = DataLoader(test_dataset_v1s, batch_size=1, shuffle=False, num_workers=num_workers_all, pin_memory=False,collate_fn=custom_collate)
            val_loader_v1m = DataLoader(test_dataset_v1m, batch_size=1, shuffle=False, num_workers=num_workers_all, pin_memory=False,collate_fn=custom_collate)
            val_loader_v2 = DataLoader(test_dataset_v2, batch_size=1, shuffle=False, num_workers=num_workers_all, pin_memory=False,collate_fn=custom_collate)
            val_loader = chain(val_loader_v2,val_loader_v1m,val_loader_v1s)
            # val_loader = chain(val_loader_v1m)
    else:
        train_dataset = AVS('train', data_ver)  # TODO: v1s, change its func name.
        # train_dataset = V1S('test', data_ver, feature_dir)
        # val_dataset = AVS('val', data_ver)  # val
        test_dataset = AVS('test', data_ver)  # test
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers_all, pin_memory=True,collate_fn=custom_collate)
        if args.val == 'val':
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers_all, pin_memory=False,collate_fn=custom_collate)
        elif args.val == 'test' or args.val == 'test_in':
            val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers_all, pin_memory=False,collate_fn=custom_collate)
    # print(model)
    # ---- fine tune config.
    params_names = [
        # 'decoder.mask_predictor',
        # 'queries_embedder',
        # 'queries_features',
        'class_proj',
        'audio_proj',
        'avs_adapt',
        'cross_a2t',
    ]
    wandb.init(
        project="mask2former_avss_ba",
        config=args,
    )

    tuned_num = 0

    for name, param in model.named_parameters():
        param.requires_grad = False
        # print('# >>> ', name)
        for _n in params_names:
            if _n in name:  # 如果该参数位于adapter_1或adapter_2中
                # print('yes:', _n, name)
                param.requires_grad = True  # 设置 requires_grad=True 微调该参数
                tuned_num += 1
    # print(model)
        if param.requires_grad:
            print("Requires_grad:", name)
    # optimizer
    params1 = [{'params': [p for name, p in model.named_parameters() if p.requires_grad], 'lr': 1e-4}]
    params = params1 #+ params2



    optimizer = torch.optim.AdamW(params)

    train_losses = []
    m_s, f_s = [], []
    
    # print(sum(p.numel() for p in model.parameters()) / 1e6)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    # input('params in M.')
    
    # train_loader = chain(train_loader_v1m)
    # val_loader = chain(val_loader_v1m)
    
    # model
    model = model.cuda()

    best_miou=0
    current_time = datetime.now()
    folder_name = current_time.strftime("%y%m%d_%H_%M_%S_%f")
    save_bool = True
    max_miou=0
    for idx_ep in range(args.epochs):
        print(f'[Epoch] {idx_ep}')
        # if idx_ep == -1:
        #     train_loader = chain(train_loader_v1s,train_loader_v1m)
        #     val_loader = chain(val_loader_v1s,val_loader_v1m)
        train_loader_epoch = copy.deepcopy(train_loader)
        val_loader_epoch = copy.deepcopy(val_loader)
        # currentDateAndTime = datetime.now().strftime("%y%m%d_%H_%M_%S_%f")
        if idx_ep==0:
            for name, param in model.named_parameters():
                # printta('# >>> ', name)
                if params_names[0] not in name and params_names[1] not in name:
                    if "mtl_loss" in name: # 如果该参数位于adapter_1或adapter_2中
                        param.requires_grad = True  # 设置 requires_grad=True 微调该参数
                        optimizer.add_param_group({'params': param,'weight_decay': 0})
                        # print(name)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print("Epoch:",idx_ep,"Requires_grad:", name)

        # if args.train:
        #     # model.model_v.mask_decoder.train()
        #     model.train()
        #     loss_train = train(model, train_loader_epoch, optimizer, idx_ep, args, ddpm)
        #     wandb.log({'loss_train': loss_train})
        #     train_losses.append(loss_train)
        
        if args.val:
            model.eval()
            # res = test(model, val_loader_epoch, optimizer, idx_ep, args, ddpm,save_mask="../place_holder/project/AVSS_m2f/test_all_mask/horse/")
            res = test(model, val_loader_epoch, optimizer, idx_ep, args, ddpm
                       ,save_mask="../place_holder/project/AVSS_m2f/test_all_mask/queries_0/")
            wandb.log(res)
            m_s.append(res["miou"])
            f_s.append(res["fscore"])
            print("Model saved!!!!")
            quit()
            if save_bool==True and res["miou"]>max_miou:
                max_miou = res["miou"]
                if not os.path.exists("../place_holder/ckpt/"+folder_name):
                    os.mkdir("../place_holder/ckpt/"+folder_name)
                torch.save(model,"../place_holder/ckpt/"+folder_name+"/"+str(idx_ep)+".ckpt")
        # print(f'train-losses: {train_losses} | test-losses: {val_losses} | test-miou: {miou_list} | test-F: {F_list}')

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(3407)
    args.data_ver = data_ver
    torch.multiprocessing.set_start_method('spawn', force=True)
    m2f_avs = PmpAVS()
    m2f_avs.load_state_dict(torch.load(
        "../preprocess/best_avss.ckpt").state_dict(),
                            strict=True)
    ddpm = DDPM(device=args.device, n_steps=100)
    # print(m2f_avs)
    run(m2f_avs, device=args.device, ddpm=ddpm)
