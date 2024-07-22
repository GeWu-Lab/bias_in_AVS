'''
# -*- encoding: utf-8 -*-
@File    :   avs_m2f_base.py
@Time    :   2024/03
@Author  :   Peiwen Sun
@description   :   For bias in AVS
'''


from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, logging
from PIL import Image
import requests
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
import re
import matplotlib.pyplot as plt
from .crossmodal import CrossmodalTransformer

logging.set_verbosity_error()


# Choose coco or ade
# image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",cache_dir=r"../place_holder/.models",local_files_only=True)
model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
    # "facebook/mask2former-swin-small-coco-instance"
    "facebook/mask2former-swin-base-ade-semantic",
    cache_dir=r"../place_holder/.models",
    local_files_only=True
)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, input, target):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(input, target, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class PmpAVS(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_v = model_m2f.cuda()


        # direct projected
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )


        # audio gated fusion
        # self.audio_gated = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid(),
        # )

        # audio cross attention
        # self.cross_a2t = CrossmodalTransformer(256, 256, 4, 3, 0.1, 100, 100)


        # Choose focal loss over simple bce
        self.loss_fn = FocalLoss()
        # self.loss_fn = F.binary_cross_entropy_with_logits  # 'bce'

    def one_hot_encoding(self, class_labels):
        def encoding(input_label):
            max_value = 99
            if len(input_label)==1:
                one_hot = torch.zeros(max_value + 1)
                one_hot[0] = 1
                one_hot = torch.clamp(one_hot, min=1e-6, max=1 - 1e-6)
                return one_hot
            one_hot = torch.eye(max_value + 1).cuda()[input_label]
            one_hot = torch.sum(one_hot, dim=0)
            # one_hot[0]=0
            one_hot = torch.clamp(one_hot, min=1e-6, max=1 - 1e-6)
            return one_hot
        one_hots = []
        for i in class_labels:
            one_hot = encoding(i).cuda()
            one_hots.append(one_hot)
        one_hots = torch.stack(one_hots)
        return one_hots


    def random_zero_tensor(self, tensor, probability):
        mask = torch.rand(tensor.size()) < probability
        random_zero_tensor = tensor * (~mask).float().cuda()
        return random_zero_tensor

    def add_gaussian_noise(self, tensor, snr_db):
        snr = 10 ** (snr_db / 10)
        noise = torch.randn_like(tensor)
        signal_power = torch.mean(tensor ** 2)
        noise_power = signal_power / snr
        noise_std = torch.sqrt(noise_power)
        noise *= noise_std
        return noise

    def forward(self, batch_data, idx_ep = None,time_step=0,ddpm=None):
        img_recs, audio_recs, mask_recs, image_sizes, vid = batch_data
        bsz = len(vid)
        loss_vid = []
        vid_preds = []
        image_sizes = torch.stack(image_sizes[0], dim=1).view(bsz, 2).tolist()
        
        # load the pretrained clustering and classification
        class_emb = self.one_hot_encoding(torch.load(
                "../preprocess/classification_avs/threshold_0.4/" + vid[0] + ".pth"))

        # not used. This is used to the other 2 methosd in paper
        # bias_vid_preds = torch.load("../place_holder/AVS_class/debias_ckpt/0.5989/"+vid[0]+".pth")
        for idx, _ in enumerate(img_recs):
            img_input = img_recs[idx]
            gt_label = mask_recs[idx]
            
            audio_emb = audio_recs[:, idx].cuda().view(bsz, 128)
            # gate = self.audio_gated(audio_emb) * 2
            audio_emb = self.audio_proj(audio_emb)
            audio_emb = audio_emb.repeat(100, 1, 1)
            if False and self.training == True:
                audio_emb = self.random_zero_tensor(audio_emb,0.05)
            audio_emb = torch.einsum('ij,ijk->ijk', class_emb.permute(1, 0)[:,idx].unsqueeze(-1), audio_emb)

            # whether use cross_a2t or not
            # audio_emb = self.cross_a2t(audio_emb.squeeze().unsqueeze(0),audio_emb.squeeze().unsqueeze(0), mask=None, tag='a2t')[0].squeeze().unsqueeze(1)
            # audio_emb = self.cross_a2t(self.model_v.model.transformer_module.queries_features.weight.unsqueeze(0),audio_emb.squeeze().unsqueeze(0), mask=None, tag='a2t')[0].squeeze().unsqueeze(1)
            
            img_input['prompt_features_projected'] = audio_emb
            # no 
            # img_input['prompt_features_projected'] = audio_emb.expand(100, -1, -1)
            img_input['pixel_values'] = img_input['pixel_values'].squeeze().view(bsz, 3, 384, 384).cuda()
            img_input['pixel_mask'] = img_input['pixel_mask'].squeeze().view(bsz, 384, 384).cuda()
            IH, IW = gt_label.shape[-2], gt_label.shape[-1]
            img_input['mask_labels'] = gt_label.view(bsz, IH, IW).cuda()
            
            outputs = self.model_v(**img_input)
            
            # Perform post-processing to get instance segmentation map
            pred_instance_map, kl_loss = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=image_sizes, class_emb=class_emb[idx], training=self.training
            )

            # Just like AVSBench and AVSegformer
            pred_instance_map = [F.interpolate(single_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False) for single_map in pred_instance_map]
            pred_instance_map = torch.stack(pred_instance_map, dim=0).view(bsz, 256, 256)            
            if True:
                eps = torch.randn_like(audio_emb[0]).cuda()
                audio_emb_distorted = ddpm.sample_forward(audio_emb[0], time_step, eps).squeeze(0)
                audio_emb_distorted = audio_emb_distorted.repeat(100, 1, 1)
                img_input['prompt_features_projected'] = audio_emb_distorted
                outputs_distorted = self.model_v(**img_input)
                pred_instance_map_distorted, kl_loss = image_processor.post_process_semantic_segmentation(
                  outputs_distorted, target_sizes=image_sizes, class_emb=class_emb[idx], training=self.training)

                pred_instance_map_distorted = [F.interpolate(single_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False) for single_map in pred_instance_map_distorted]
                pred_instance_map_distorted = torch.stack(pred_instance_map_distorted, dim=0).view(bsz, 256, 256)
                pred_instance_map = (2*pred_instance_map - 1*pred_instance_map_distorted)
            
            # Use other method stated in the paper
            if False and self.training == True and (type(idx_ep) == int and idx_ep>=0):
                pred_instance_map = 2*pred_instance_map - 1*bias_vid_preds[idx]
            loss_frame = self.loss_fn(input=pred_instance_map.squeeze(), target=mask_recs[idx].squeeze().cuda())
            loss_vid.append(loss_frame-0*kl_loss)
            vid_preds.append(pred_instance_map.squeeze())
            
        return loss_vid, vid_preds
