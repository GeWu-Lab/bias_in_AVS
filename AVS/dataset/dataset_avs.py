import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb

import sys
import os
import random

sys.path.append('../modeling/')
sys.path.append('..')
sys.path.append('../../segment_anything/')

from torchvision import transforms
from collections import defaultdict
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image


import pickle as pkl

class AVS(Dataset):  # 读取avs metadata
    def __init__(self, split='train', ver='v2', feature_dir='', model=None, device=None, audio_from=None):
        # metadata: train/test/val
        self.cooperation = False
        self.device = device
        self.model = model
        self.ver = ver
        self.feature_base_path = feature_dir
        self.data_base_path = f'/home/data/AVS'
        self.data_path = f'{self.data_base_path}/{ver}'
        self.audio_from = audio_from
        meta_path = f'{self.data_base_path}/metadata.csv'
        metadata = pd.read_csv(meta_path, header=0)
        sub_data = metadata[metadata['label'] == ver]  # v1s set

        self.split = split
        self.metadata = sub_data[sub_data['split'] == split]  # split= train,test,val.

        if self.cooperation:
            cooperation = pd.read_csv("../place_holder/dataset/cooperation/meta_cooperation_full.csv", header=0)
            self.metadata = pd.merge(self.metadata, cooperation, on='vid', how='right')
 
        self.audio = None
        self.images = None

        self.frame_num = 10 if ver == 'v2' else 5  
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

        self.work_dir = '../place_holder/sam'
        self.feat_path = f'{self.work_dir}/segment_anything/feature_extract'

        # save the preprocessed feat and then load them to save time, change it
        self.load = True

        if not self.load:
            self.img_process = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",
                                                                  cache_dir=r"../place_holder/.models",
                                                                  local_files_only=True)
    def __len__(self):
        if self.cooperation:
            return len(self.metadata)*5
        else:
            return len(self.metadata)

    def __getitem__(self, idx):
        if self.cooperation:
            df_one_video = self.metadata.iloc[idx//5]
        else:
            df_one_video = self.metadata.iloc[idx]
        vid, category = df_one_video['uid'], df_one_video['a_obj']  # uid for vid.
        if not self.load:
            img_recs = []
            mask_recs = []
            images = []

            feat_aud_p = f'{self.work_dir}/segment_anything/feature_extract/{self.ver}_vggish_embs/{vid}.npy'

            feat_aud = torch.from_numpy(np.load(feat_aud_p)).to(self.device).squeeze().detach()[:self.frame_num]
            feat_text = feat_aud

            # ----------
            if feat_aud.shape[0] != 5 and self.ver == 'v1s':
                while feat_aud.shape[0] != 5:
                    feat_aud = torch.concat([feat_aud, feat_aud[-1].view(1, -1)], dim=0)
            assert feat_aud.shape[0] == self.frame_num
            # ----------

            FN = 5

            for _idx in range(FN):  # set frame_num as the batch_size
                if _idx >= 5 and self.split == 'train':
                    break

                if self.ver == 'v1s' and self.split == 'train':
                    path_frame = f'{self.data_path}/{vid}/frames/0.jpg'  # image
                else:
                    path_frame = f'../place_holder/more_frame_AVS_video/v1m_frame_10/{vid}/frames/{_idx+5*3}.jpg'  # image


                image = Image.open(path_frame)
                image_inputs = self.img_process(image, return_tensors="pt")
                # print(image_inputs.pixel_value)

                # if self.ver == 'v1s':
                #     feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f{_idx}.npy'  # image feature
                #     image_embed = torch.from_numpy(np.load(feat_img_p)).squeeze().to(self.device)
                # else:
                #     feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f{_idx}.pth'  # image feature
                #     image_embed = torch.load(feat_img_p).squeeze().to(self.device)

                # path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'  # image

                # if self.ver == 'v1s' and self.split == 'train':
                #     path_frame = f'{self.data_path}/{vid}/frames/0.jpg'  # image
                #     feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f0.npy'  # image feature
                #     image_embed = torch.from_numpy(np.load(feat_img_p)).squeeze().to(self.device)

                # data
                # transformed_data = defaultdict(dict)
                # image = cv2.imread(path_frame)
                # # image = cv2.resize(image, (720, 1280))

                # input_image = self.transform.apply_image(image)

                # input_image_torch = torch.as_tensor(input_image, device=self.device)
                # transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

                # # prepare for input
                # input_image = self.model.preprocess(transformed_image)
                # # print(image.shape)
                # original_image_size = (image.shape[0], image.shape[1])  # H x W
                # input_size = tuple(transformed_image.shape[-2:])
                # # print(input_size, original_image_size)
                # # input()  # (576, 1024) (720, 1280)

                # # embedding
                # audio_embed = feat_aud[_idx].squeeze().to(self.device)
                # # text_embed = feat_aud[_idx].squeeze().to(self.device)
                # text_embed = feat_text[_idx].squeeze().to(self.device)
                # # num_obj = num_objs[_idx]

                # # print(image_embed)
                # # print("audio_emb:", audio_embed)
                # # print('[ds] image:', image_embed.shape)  # [256, 64, 64])

                # if self.ver == 'v1s' and self.split == 'train':
                #     audio_embed = feat_aud[0].squeeze().to(self.device)
                #     text_embed = feat_text[0].squeeze().to(self.device)
                #     path_frame = f'{self.data_path}/{vid}/frames/0.jpg'  # image

                # # dict input
                # transformed_data['image'] = input_image.squeeze()
                # transformed_data['image_path'] = path_frame
                # transformed_data['input_size'] = input_size
                # transformed_data['original_size'] = original_image_size
                # transformed_data['image_embed'] = image_embed
                # transformed_data['audio'] = audio_embed
                # transformed_data['text'] = text_embed
                # # transformed_data['num_obj'] = num_obj
                # transformed_data['num_obj'] = 2
                # # transformed_data['engine_input'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # mask label
                _idx_mask = 0 if self.split == 'train' and self.ver == 'v1s' else _idx
                mask_cv2 = np.ones((256,256))
                mask = mask_cv2
                ground_truth_mask = (mask > 0)  # turn to T/F mask.
                gt_mask_resized = ground_truth_mask
                gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

                image_sizes = [image.size[::-1]]

                # singe rec
                img_recs.append(image_inputs)
                mask_recs.append(gt_binary_mask)


            # print(img_recs.shape)
            # img_recs = torch.stack(img_recs, dim=0)
            # mask_recs = torch.stack(mask_recs, dim=0)

            # # save the preprocessed feat and then load them to save time

            # return img_recs, mask_recs, vid, category, feat_aud, feat_text
            # torch.save((img_recs, feat_aud, mask_recs, image_sizes, vid),
            #            "../place_holder/dataset/AVS_m2f_processed/" + self.ver + "" + "/" + vid + ".pth")
            return img_recs, feat_aud, mask_recs, image_sizes, vid
        else:
            if self.cooperation == True:
                res = torch.load("../place_holder/dataset/AVS_m2f_processed/" + self.ver + "" + "/" + vid + ".pth")
                feat_aud_p = "../place_holder/dataset/cooperation/cooperation_vggish_embeds/" + df_one_video["new_vid"] + "/" + df_one_video["new_vid"]+"_"+str(idx%5+2)+"-"+ str(6-idx%5) +".npy"
                feat_aud = torch.zeros_like(torch.from_numpy(np.load(feat_aud_p)).to(self.device).squeeze().detach()[:self.frame_num])
                res = (res[0],feat_aud,res[2],res[3],res[4])
                return res
            else:
                return torch.load("../place_holder/dataset/AVS_m2f_processed/" + self.ver + "" + "/" + vid + ".pth")


if __name__ == "__main__":
    train_dataset = V1S('train', 'v2', '../feature_extract')
    train_dataset = V1S('test', 'v2', '../feature_extract')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask, category, vid = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
