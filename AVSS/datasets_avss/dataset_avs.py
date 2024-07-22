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

# sys.path.append('..')
sys.path.append('..')
# sys.path.append('../../segment_anything/')
# sys.path.append('../place_holder/utils')
# print(sys.path)
# from utils.utils import log_agent
# from utils.transforms import ResizeLongestSide
from torchvision import transforms
from collections import defaultdict
import cv2
from transformers import Mask2FormerImageProcessor
import json
from PIL import Image

# logger = log_agent('audio_recs.log')

import pickle as pkl


# def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
#     img_PIL = Image.open(path).convert(mode)
#     if transform:
#         img_tensor = transform(img_PIL)
#         return img_tensor
#     return img_PIL

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete

def get_v1m_pallete(label_to_idx_path, num_cls=22):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open("./datasets_avss/v1m_idx.json", 'r') as fr:
        v1m_json = json.load(fr)
    # with open(label_to_idx_path, 'r') as fr:
    #     label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls=71) # list
    result = []
    for i in v1m_json.values():
        result.append(v2_pallete[(int(i)-1)*3:(int(i)-1)*3+3])
    # print(result)
    result = np.array(result).reshape(-1, 3)
    assert len(result) == 22
    return result



def crop_resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not img_is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    # print("crop for train. set")
    return img

def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    # if cfg.DATA.CROP_IMG_AND_MASK:
    #     if split == 'train':
    #         color_mask_PIL = crop_resize_img(cfg.DATA.CROP_SIZE, color_mask_PIL, img_is_mask=True)
    #     else:
    #         color_mask_PIL = resize_img(cfg.DATA.CROP_SIZE, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label) # [H, W]
    color_label = color_label.unsqueeze(0)
    # binary_mask = (color_label != (cfg.NUM_CLASSES-1)).float()
    # return color_label, binary_mask # both [1, H, W]
    return color_label # both [1, H, W]


class AVS(Dataset):  # 读取avs metadata
    def __init__(self, split='train', ver='v2', feature_dir='', model=None, device=None, audio_from=None):
        # metadata: train/test/val
        self.device = device
        self.model = model
        # self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.ver = ver
        self.feature_base_path = feature_dir
        self.data_base_path = f'../place_holder/data/AVS'
        self.data_path = f'{self.data_base_path}/{ver}'
        self.audio_from = audio_from
        meta_path = f'{self.data_base_path}/metadata.csv'
        metadata = pd.read_csv(meta_path, header=0)
        sub_data = metadata[metadata['label'] == ver]  # v1s set
        # sub_data_train = sub_data[sub_data['split'] == 'train']
        # sub_data_test = sub_data[sub_data['split'] == 'test']
        # sub_data_val = sub_data[sub_data['split'] == 'val']
        self.split = split
        self.metadata = sub_data[sub_data['split'] == split]  # split= train,test,val.
        # print(self.metadata)
        # input()
 
        self.audio = None
        self.images = None

        self.frame_num = 10 if ver == 'v2' else 5  # 每个video切分5个frame - 训练集只有1个
        # self.frame_num = 5 if split == 'train' else 10 # 每个video切分5个frame - 训练集只有1个
        if self.ver == 'v1s' and self.split=='train':
            self.frame_num = 5
        # print(f'[dataset] self.frame_num: {self.frame_num}')
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        if self.ver == "v1m":
            self.v2_pallete = get_v1m_pallete("") # Change v1m
            # self.v2_pallete = get_v2_pallete("../place_holder/dataset/label2idx.json")
        elif self.ver == "v2":
            self.v2_pallete = get_v2_pallete("../place_holder/dataset/label2idx.json")
        elif self.ver == "v1s":
            self.v2_pallete = get_v2_pallete("../place_holder/dataset/label2idx.json")

        # self.logger = log_agent('dataset.log')

        self.work_dir = '../place_holder/segment-anything-main/'
        self.feat_path = f'{self.work_dir}/segment_anything/feature_extract'
        self.process_bool = False
        if self.process_bool:
            self.img_process = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",
                                                            cache_dir=".models/",local_files_only=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid, category = df_one_video['uid'], df_one_video['a_obj']  # uid for vid.
        if self.process_bool:
            img_recs = []
            mask_recs = []
            images = []

            feat_aud_p = f'{self.work_dir}/segment_anything/feature_extract/{self.ver}_vggish_embs/{vid}.npy'

            feat_aud = torch.from_numpy(np.load(feat_aud_p)).to(self.device).squeeze().detach()[:self.frame_num]
            # print(f'{vid} ', feat_aud_p.shape)
            # feat_aud = torch.zeros_like(feat_aud)
            # feat_text = torch.from_numpy(np.load(feat_text_p)).to(self.device).squeeze().detach()[:self.frame_num]  # [5, 768]
            # num_objs = torch.from_numpy(np.load(num_objs_p))  # [5, 768]
            # print(feat_aud.shape, feat_text.shape)

            # ----------
            if feat_aud.shape[0] != 5 and self.ver == 'v1s':
                while feat_aud.shape[0] != 5:
                    # print(f'> warning: find {feat_aud.shape[0]}/5 audio clips in {vid}, repeat the last clip.')
                    feat_aud = torch.concat([feat_aud, feat_aud[-1].view(1, -1)], dim=0)
            # print(f'Now: {feat_aud.shape, self.frame_num}')
            if not feat_aud.shape[0] == self.frame_num:
                print(feat_aud.shape[0], self.frame_num)
            assert feat_aud.shape[0] == self.frame_num
            # ----------

            # image feature
            # "../place_holder/data/AVS/v2/--iSerV5DbY_119000_129000/frames/0.jpg"
            # _frame_num = feat_aud.shape[0] if self.ver == 'v1s' else self.frame_num
            # FN = self.frame_num
            # FN = 1 if self.ver == 'v1s' and self.split == 'train' else self.frame_num
            # FN = 5 if self.ver == 'v1s' and self.split == 'train' else self.frame_num
            # print(self.frame_num)
            FN = self.frame_num
            # print(self.frame_num)
            image_list, color_label_list = [],[]
            for _idx in range(FN):  # set frame_num as the batch_size
                # if _idx >= 5 and self.split == 'train':
                #     break

                if self.ver == 'v1s' and self.split == 'train':
                    path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'
                    # path_frame = f'{self.data_path}/{vid}/frames/0.jpg'  # image
                    # feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f0.npy'  # image feature
                    # image_embed = torch.from_numpy(np.load(feat_img_p)).squeeze().to(self.device)
                else:
                    path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'  # image


                # image = Image.open(path_frame)
                # print([image.size[::-1]])
                image = cv2.imread(path_frame)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.transpose(2, 0, 1)

                _idx_mask = 0 if self.split == 'train' and self.ver == 'v1s' else _idx
                path_mask = f'{self.data_path}/{vid}/labels_rgb/{_idx_mask}.png'
                color_label = load_color_mask_in_PIL_to_Tensor(path_mask, v_pallete=self.v2_pallete, split=self.split).squeeze(0)
                color_label_list.append(color_label)
                image_list.append(image)


            # print(img_recs.shape)
            # img_recs = torch.stack(img_recs, dim=0)
            # mask_recs = torch.stack(mask_recs, dim=0)

            image_inputs = self.img_process.preprocess(image_list, color_label_list,return_tensors="pt")
            # print(image_inputs["pixel_values"].shape,image_inputs["pixel_mask"].shape,len(image_inputs["class_labels"]),len(image_inputs["mask_labels"]))
            image_sizes = [image.shape[1:]]
            image_inputs["mask_recs"] = torch.stack(color_label_list)
            image_inputs["image_size"] = image_sizes
            image_inputs["vid"] = vid
            image_inputs["feat_aud"] = feat_aud
            # print(image_inputs["pixel_values"].shape,
            #       image_inputs["pixel_mask"].shape, len(image_inputs["mask_labels"]), image_inputs["class_labels"])
            torch.save(image_inputs,"../place_holder/dataset/AVS_m2f_processed/"+self.ver+"_22"+"/"+vid+".pth")
            # return img_recs, mask_recs, vid, category, feat_aud, feat_text
        else:
            image_inputs=torch.load("../place_holder/dataset/AVS_m2f_processed/"+self.ver+"/"+vid+".pth")
        return image_inputs


"""
# save record for each audio file.
audios = torch.load('../AudioCLIP/demo/audio_features_v1s/audio_features_v1s_with_name.pth')
for a_idx, audio in enumerate(audios):
    embed = audio['embed'].numpy()
    name = audio['name'].split('_a_audio.wav')[0]

    np.save(f'../feature_extract/audio_v1s_feature/{name}.npy', embed)
    logger.info(f'process: {a_idx} | name: {name}')
"""

if __name__ == "__main__":
    train_dataset = V1S('train', 'v2', '../feature_extract')
    train_dataset = V1S('test', 'v2', '../feature_extract')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask, category, vid = batch_data  # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # pdb.set_trace()
        print(mask.shape)
