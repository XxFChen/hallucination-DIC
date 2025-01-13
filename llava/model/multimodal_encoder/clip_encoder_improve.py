import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # 添加可选参数，默认查看最后一层
        self.visualize_layer = getattr(args, 'visualize_layer', -1)  

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def visualize_attention_maps(self, attention_maps, save_dir='attention_maps', image=None):
        """
        可视化指定层的平均attention map
        Args:
            attention_maps: 模型的attention权重
            save_dir: 保存可视化结果的目录
            image: 原始输入图像（可选）
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 如果提供了原始图像，保存它
        if image is not None:
            plt.figure(figsize=(5, 5))
            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            plt.imshow(img_np)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, 'original_image.png'))
            plt.close()

        # 获取指定层的attention map
        layer_idx = self.visualize_layer
        if layer_idx < 0:  # 如果是负数，从最后一层往前数
            layer_idx = len(attention_maps) + layer_idx
        
        if 0 <= layer_idx < len(attention_maps):
            attn = attention_maps[layer_idx]
            # attn shape: [batch_size, num_heads, seq_len, seq_len]
            attn_map = attn[0]  # 取第一个样本的所有注意力头
            
            # 计算所有注意力头的平均值
            avg_attn = attn_map.mean(dim=0).cpu().float().numpy()
            
            # 绘制平均注意力图
            plt.figure(figsize=(10, 10))
            sns.heatmap(avg_attn, cmap='viridis')
            plt.title(f'Layer {layer_idx} Average Attention')
            plt.savefig(os.path.join(save_dir, f'layer_{layer_idx}_average_attention.png'))
            plt.close()
        else:
            print(f"Warning: Layer index {layer_idx} is out of range. Total layers: {len(attention_maps)}")

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            attention_maps = []
            for i, image in enumerate(images):
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)

                attention_maps.append(image_forward_out.attentions)
                self.visualize_attention_maps(
                    image_forward_out.attentions,
                    save_dir=f'attention_maps/image_{i}',
                    image=image
                )
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            attention_maps = image_forward_outs.attentions
            
            self.visualize_attention_maps(
                attention_maps,
                save_dir='attention_maps/batch',
                image=images[0] if images.dim() == 4 else None
            )

        return image_features, attention_maps


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2