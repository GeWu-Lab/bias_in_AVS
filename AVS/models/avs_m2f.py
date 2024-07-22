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
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic")
model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
    # "facebook/mask2former-swin-small-coco-instance"
    "facebook/mask2former-swin-base-ade-semantic"
)


class PmpAVS(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_v = model_m2f.cuda()
        
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        self.loss_fn = F.binary_cross_entropy_with_logits  # 'bce'

    def forward(self, batch_data):
        img_recs, audio_recs, mask_recs, image_sizes, vid = batch_data
        bsz = len(vid)
        loss_vid = []
        vid_preds = []
        image_sizes = torch.stack(image_sizes[0], dim=1).view(bsz, 2).tolist()
        for idx, _ in enumerate(img_recs):
            img_input = img_recs[idx]
            gt_label = mask_recs[idx]
            
            audio_emb = audio_recs[:, idx].cuda().view(bsz, 1, 128)
            audio_emb = self.audio_proj(audio_emb)

            img_input['prompt_features_projected'] = audio_emb
            img_input['pixel_values'] = img_input['pixel_values'].squeeze().view(bsz, 3, 384, 384).cuda()
            img_input['pixel_mask'] = img_input['pixel_mask'].squeeze().view(bsz, 384, 384).cuda()
            IH, IW = gt_label.shape[-2], gt_label.shape[-1]

            img_input['mask_labels'] = gt_label.view(bsz, IH, IW).cuda()
            
            outputs = self.model_v(**img_input)
            
            # Perform post-processing to get instance segmentation map
            pred_instance_map = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=image_sizes,
            )  # [0]
            
            pred_instance_map = [F.interpolate(single_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False) for single_map in pred_instance_map]
            pred_instance_map = torch.stack(pred_instance_map, dim=0).view(bsz, 256, 256)
            resized_pred = F.interpolate(pred_instance_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
            loss_frame = self.loss_fn(input=resized_pred.squeeze(), target=mask_recs[idx].squeeze().cuda())
            loss_vid.append(loss_frame)
            vid_preds.append(pred_instance_map.squeeze())

        return loss_vid, vid_preds

def vis_seg(pred_instance_map):
    plt.figure()

    # 使用imshow函数将张量作为图像显示
    plt.imshow(pred_instance_map.cpu(), cmap='gray')  # 如果 pred 是灰度图像，使用'gray'色彩映射；如果是彩色图像，可以使用默认的 'viridis' 或其他合适的色彩映射
    plt.savefig('./demo.png')
    # 显示图像
    plt.show()
    plt.close()
    
    print("pred-shape:", pred_instance_map.shape)
    input('>')
