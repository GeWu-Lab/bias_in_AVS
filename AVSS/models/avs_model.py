'''
# -*- encoding: utf-8 -*-
@File    :   avs_model.py
@Time    :   2024/03
@Author  :   Peiwen Sun
@description   :   For bias in AVS
'''

# from transformers import AutoImageProcessor
import sys
sys.path.append('..')
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
import re
import matplotlib.pyplot as plt
from .crossmodal import CrossmodalTransformer

# Load Mask2Former trained on COCO instance segmentation dataset
ckpt = "facebook/mask2former-swin-base-ade-semantic"
print("using:", ckpt)
image_processor = Mask2FormerImageProcessor.from_pretrained(ckpt,
                                                            cache_dir=".models/",local_files_only=True)
model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(ckpt,cache_dir=".models/", local_files_only=True)


class PmpAVS(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_v = model_m2f.cuda()
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.m2f_criterion = self.model_v.criterion
        # self.cross_a2t = CrossmodalTransformer(256, 256, 4, 3, 0.1, 100, 100)
        # self.class_proj = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(256, 100),
        #     # nn.ReLU(),
        #     nn.Softmax(dim=1),
        #     # nn.Sigmoid(),
        # )
        # self.sig = nn.Sigmoid()

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
            one_hot = torch.clamp(one_hot, min=1e-6, max=1 - 1e-6)
            return one_hot
        one_hots = []
        for i in class_labels:
            one_hot = encoding(i).cuda()
            one_hots.append(one_hot)
        one_hots = torch.stack(one_hots)
        return one_hots

    def one_hot_bce_loss(self, class_labels, class_embeds):
        loss_funct = torch.nn.BCELoss()

        one_hots = self.one_hot_encoding(class_labels)
        loss = loss_funct(one_hots, class_embeds)
        return 10*loss

    def add_gaussian_noise(self, tensor, snr_db):
        snr = 10 ** (snr_db / 10)
        noise = torch.randn_like(tensor)
        signal_power = torch.mean(tensor ** 2)
        noise_power = signal_power / snr
        noise_std = torch.sqrt(noise_power)
        noise *= noise_std
        noisy_tensor = tensor + noise
        return noisy_tensor

    def forward(self, batch_data, idx_ep=None,time_step=0,ddpm=None):
        for idx in range(1):
            len_img = len(batch_data["mask_recs"][idx])
            image_sizes = batch_data["image_size"][idx]
            image_sizes = len_img*image_sizes
            gt_label = batch_data["mask_recs"][idx]
            img_input = {}
            audio_emb = batch_data["feat_aud"][idx].cuda()
            # audio_emb = torch.zeros_like(audio_emb) # verify the sensitivity
            audio_emb = self.audio_proj(audio_emb)
            audio_emb = audio_emb.view(1, -1, 256).repeat(100, 1, 1)# [100, 1, 256]

            img_input['pixel_values'] = batch_data['pixel_values'][idx].squeeze().view(-1, 3, 384, 384).cuda()
            img_input['pixel_mask'] = batch_data['pixel_mask'][idx].squeeze().view(-1, 384, 384).cuda()

            img_input["mask_labels"] = [i.cuda() for i in batch_data["mask_labels"][idx]]
            img_input["class_labels"] = [i.cuda() for i in batch_data["class_labels"][idx]]
            class_emb = self.one_hot_encoding(torch.load(
                "../preprocess/classification_avs/threshold_0.4/" + batch_data["vid"][idx] + ".pth"))
            audio_emb = torch.einsum('ij,ijk->ijk', class_emb.permute(1, 0), audio_emb)
            bsz = audio_emb.shape[1]

            # use cross attn or not
            # audio_emb = self.cross_a2t(audio_emb.permute(1, 0, 2),
            #                            self.model_v.model.transformer_module.queries_features.weight.unsqueeze(0).repeat(bsz, 1, 1),
            #                            mask=None, tag='a2t')[0].permute(1, 0, 2)
            img_input['prompt_features_projected'] = audio_emb
            img_input['prompt_class_projected'] = class_emb
            img_input['vid'] = batch_data["vid"][idx]

            outputs = self.model_v(**img_input)
            outputs_distorted = outputs
            if True:
                eps = torch.randn_like(audio_emb[0]).cuda()
                audio_emb_distorted = ddpm.sample_forward(audio_emb[0], time_step, eps).squeeze(0)
                audio_emb_distorted = audio_emb_distorted.repeat(100, 1, 1)
                img_input['prompt_features_projected'] = audio_emb_distorted
                outputs_distorted = self.model_v(**img_input)

            # with all positive queries just like original
            if False and self.training == False:
                with torch.no_grad():
                    audio_emb = self.add_gaussian_noise(audio_emb,30)
                    img_input['prompt_features_projected'] = audio_emb
                    img_input['prompt_class_projected'] = torch.ones_like(class_emb)
                    outputs_distorted = self.model_v(**img_input)

            # not use. was used for other 2 debias strategy
            if False and self.training == False:
                outputs_debias = torch.load(
                    "../place_holder/project/AVSS_m2f/debias_pth/outputs/0.3317/" + batch_data["vid"][
                        idx] + ".pth")
                outputs_distorted = outputs_debias
            # Perform post-processing to get instance segmentation map
            obser = torch.zeros_like(img_input["prompt_class_projected"])
            obser[:, 0] = 1
            if torch.all(img_input["prompt_class_projected"] * obser < 0.5):
                return 0, None
            output = image_processor.post_process_semantic_segmentation(
                outputs, outputs_distorted, target_sizes=image_sizes,observe_queries=img_input["prompt_class_projected"]*obser)
            pred_instance_map = output

            mask_logits = 2 * outputs.masks_queries_logits - outputs_distorted.masks_queries_logits
            masks_queries_logits_match = torch.einsum('ij,ijhw->ijhw', img_input["prompt_class_projected"], mask_logits)
            class_logits = 2 * outputs.class_queries_logits - outputs_distorted.class_queries_logits
            indices = self.m2f_criterion.matcher(masks_queries_logits_match, class_logits, img_input["mask_labels"], img_input["class_labels"])
            num_masks = self.m2f_criterion.get_num_masks(img_input["class_labels"], device=img_input["class_labels"][0].device)
            loss_distorted = {
                **self.m2f_criterion.loss_masks(mask_logits, img_input["mask_labels"], indices, num_masks, img_input["prompt_class_projected"],False),
                **self.m2f_criterion.loss_labels(class_logits, img_input["class_labels"], indices),
            }

            loss_distorted = self.model_v.get_loss(loss_distorted)
            loss_frame = outputs.loss + loss_distorted*10
        return [loss_frame], pred_instance_map


def vis_seg(pred_instance_map):
    plt.figure()

    plt.imshow(pred_instance_map.cpu(), cmap='gray')
    plt.savefig('./demo.png')
    plt.show()
    plt.close()
    
    print("pred-shape:", pred_instance_map.shape)
    input('>')
