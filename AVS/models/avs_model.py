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


# Load Mask2Former trained on COCO instance segmentation dataset
# image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",cache_dir=r"../place_holder/.models",local_files_only=True)
model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
    # "facebook/mask2former-swin-small-coco-instance"
    "facebook/mask2former-swin-base-ade-semantic",
    cache_dir=r"../place_holder/.models",
    local_files_only=True
)

# avs_dataset = AVS()

class PmpAVS(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model_v = model_v
        # self.model_t = model_t
        # self.config = config
        self.model_v = model_m2f.cuda()
        # self.model_v = None
        
        self.audio_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        self.loss_fn = F.binary_cross_entropy_with_logits  # 'bce'
        
        self.num_audio_components = 8
        self.acu_query = nn.Embedding(self.num_audio_components, 256)
        self.cross_a2t = CrossmodalTransformer(256, 256, 4, 3, 0.1, 8, 8)

    def forward(self, batch_data):
        img_recs, audio_recs, mask_recs, image_sizes, vid = batch_data
        bsz = len(vid)
        loss_vid = []
        vid_preds = []
        # print(image_sizes)
        image_sizes = torch.stack(image_sizes[0], dim=1).view(bsz, 2).tolist()
        # print(image_sizes)
        # print(len(img_recs))
        for idx, _ in enumerate(img_recs):
            img_input = img_recs[idx]
            gt_label = mask_recs[idx]
            
            # print('audio:', audio_recs.shape)
            audio_emb = audio_recs[:, idx].cuda().view(bsz, 1, 128)  # .repeat(100, 1, 1)
            audio_emb = self.audio_proj(audio_emb)  # [100, 1, 256]
            # print(audio_emb.shape, self.acu_query.weight.shape)
            
            audio_emb = audio_emb * self.acu_query.weight
            # print(audio_emb.shape)
            # input()
            
            text_emb = audio_emb
            # print(audio_recs.shape)
            
            audio_emb, bad_semantic = self.cross_a2t(audio_emb, audio_emb, mask=None, tag='a2t')
            # print(audio_emb.shape)  # [16, 8, 256]
            # print(bad_semantic.shape)  # [bsz, num_q]  # 作为下一次的key_mask.
            # input('---')
            # not_bad = ~bad_semantic  # 1 if not bad.
            # key_mask = 
            
            img_input['prompt_features_projected'] = audio_emb
            img_input['pixel_values'] = img_input['pixel_values'].squeeze().view(bsz, 3, 384, 384).cuda()
            img_input['pixel_mask'] = img_input['pixel_mask'].squeeze().view(bsz, 384, 384).cuda()
            IH, IW = gt_label.shape[-2], gt_label.shape[-1]
            # print(IW, IH)
            img_input['mask_labels'] = gt_label.view(bsz, IH, IW).cuda()
            
            # with torch.no_grad():
            outputs = self.model_v(**img_input)
            # print(outputs)
            
            # Perform post-processing to get instance segmentation map
            pred_instance_map = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=image_sizes,
            )  # [0]

            # print(len(pred_instance_map))  # bsz
            # print(pred_instance_map[0].shape)  # [8, 405, 720]
            pred_instance_map = [F.interpolate(single_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False) for single_map in pred_instance_map]
            # pred_instance_map = torch.stack(pred_instance_map, dim=0).view(bsz, 256, 256)
            # pred_instance_map = [F.interpolate(single_map.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False) for single_map in pred_instance_map]
            # print(pred_instance_map[0].shape)  # [8, 405, 720]
            num_q = pred_instance_map[0].shape[0]

            # dynamic mask to filte bad semntics.
            # print(not_bad)
            pred_instance_map = torch.stack(pred_instance_map, dim=0).view(bsz, num_q, 256, 256)
            # pred_instance_map = pred_instance_map * not_bad.unsqueeze(-1).unsqueeze(-1)
            # print(pred_instance_map.max(), pred_instance_map.min())
            # input()

            bsz, query, H, W = pred_instance_map.shape

            # 将张量 reshape 成 [bsz*query, H, W]
            # reshaped_tensor = pred_instance_map

            # 计算每个 query 的 mask 是否为全 0
            # print(reshaped_tensor.shape)
            # mask_sum = torch.sum(reshaped_tensor, dim=(2, 3))

            # 找到 mask 全 0 的 query 的索引
            # zero_mask_query_indices = torch.where(mask_sum == 0)
            # print(zero_mask_query_indices)
            # input()

            # pred_instance_map = pred_instance_map.mean(dim=1).squeeze()
            # print(pred_instance_map.shape)
            # print(mask_recs[idx].shape, pred_instance_map.shape)
            # print(mask_recs[idx].squeeze())
            # loss_frame = outputs.loss
            # print('loss:', loss_frame)
            # input()
            # H, W = pred_instance_map.shape[0], pred_instance_map.shape[1]
            resized_pred = F.interpolate(pred_instance_map, size=(256, 256), mode='bilinear', align_corners=False)
            loss_frame = self.loss_fn(input=resized_pred.squeeze(), target=mask_recs[idx].squeeze().cuda())
            # print('loss_frame:', loss_frame)
            loss_vid.append(loss_frame)
            vid_preds.append(pred_instance_map.squeeze())
            
            # vis_seg(pred_instance_map.squeeze().detach().cpu())
            

        return loss_vid, vid_preds

# model = PmpAVS()

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

# # # input('done.')

# img_path = '../place_holder/workplace/Mask2Former/AVS/scripts/x.jpg'
# # # image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open(img_path)
# inputs = image_processor(image, return_tensors="pt")

# prompt_features_projected = torch.randn([100, 1, 128])
# prompt_features_projected = model.audio_proj(prompt_features_projected)
# inputs['prompt_features_projected'] = prompt_features_projected
# # # print(inputs)
# # # input('xk.')
# print("pix1:", inputs['pixel_values'].shape)
# print("pix2:", inputs['pixel_mask'].shape)

# outputs = model.model_v(**inputs)
# # with torch.no_grad():
# #     outputs = model.model_v(**inputs)

# # # print(outputs)

# # # # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
# # # # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
# # # class_queries_logits = outputs.class_queries_logits  # [1, 100, 96, 96]
# # # masks_queries_logits = outputs.masks_queries_logits
# # # # mql = torch.zeros_like(masks_queries_logits)
# # # # outputs.masks_queries_logits = mql
# # # print(masks_queries_logits.shape)

# # # Perform post-processing to get instance segmentation map
# pred_instance_map = image_processor.post_process_semantic_segmentation(
#     outputs, target_sizes=[image.size[::-1]]
# )[0]

# print(pred_instance_map.shape)

# import matplotlib.pyplot as plt

# # 假设 pred 是形状为 [480, 640] 的张量

# # 使用Matplotlib创建一个新的图像对象
# plt.figure()

# # 使用imshow函数将张量作为图像显示
# plt.imshow(pred_instance_map, cmap='gray')  # 如果 pred 是灰度图像，使用'gray'色彩映射；如果是彩色图像，可以使用默认的 'viridis' 或其他合适的色彩映射
# plt.savefig('./demo.png')
# # 显示图像
# plt.show()