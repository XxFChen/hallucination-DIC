import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers.modeling_outputs import BaseModelOutput 

class CLIPVisionTransformerWithMoD(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.top_k_ratio = getattr(config, 'mod_top_k_ratio', 0.8)
        self.mod_start_layer = getattr(config, 'mod_start_layer', 3)
        self.mod_end_layer = getattr(config, 'mod_end_layer', -2)
    def forward(self, pixel_values, output_attentions=False, output_hidden_states=True, return_dict=True):
        hidden_states = self.embeddings(pixel_values)
        batch_size, seq_len, _ = hidden_states.size()

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 创建注意力掩码
        attention_mask = None
        causal_attention_mask = None

        for idx, layer_module in enumerate(self.encoder.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=True,  # 我们需要注意力权重
            )
            hidden_states = layer_outputs[0]
            attention_weights = layer_outputs[1]

            # 判断是否应用 MoD
            if (
                idx >= self.mod_start_layer  # 跳过前三层
                and idx <= len(self.encoder.layers) + self.mod_end_layer  # 跳过最后一层
                and (idx - self.mod_start_layer) % 2 == 0  # 每隔一层加 MoD
            ):
                # 计算注意力分数
                attention_weights = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
                attention_scores = attention_weights.sum(dim=2)  # [batch_size, seq_len]
                attention_scores = attention_scores / attention_scores.sum(dim=1, keepdim=True)

                # MoD处理
                top_k = int(seq_len * self.top_k_ratio)
                _, top_k_indices = torch.topk(attention_scores, top_k, dim=1)
                output = hidden_states.clone()

                for b in range(batch_size):
                    curr_indices = top_k_indices[b]
                    mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
                    mask[curr_indices] = True
                    output[b][mask] = hidden_states[b][mask]
                    output[b][~mask] = hidden_states[b][~mask]

                hidden_states = output

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    # def forward(self, pixel_values, output_attentions=False, output_hidden_states=True, return_dict=True):
    #     hidden_states = self.embeddings(pixel_values)
    #     batch_size, seq_len, _ = hidden_states.size()

    #     all_hidden_states = () if output_hidden_states else None
    #     all_attentions = () if output_attentions else None

    #     # 创建注意力掩码
    #     attention_mask = None
    #     causal_attention_mask = None

    #     for idx, layer_module in enumerate(self.encoder.layers):
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (hidden_states,)

    #         layer_outputs = layer_module(
    #             hidden_states,
    #             attention_mask,
    #             causal_attention_mask,
    #             output_attentions=True,  # 我们需要注意力权重
    #         )
    #         hidden_states = layer_outputs[0]
    #         attention_weights = layer_outputs[1]

    #         # 只在指定层范围内应用MoD
    #         if self.mod_start_layer <= idx <= len(self.encoder.layers) + self.mod_end_layer:
    #             # 计算注意力分数
    #             attention_weights = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
    #             attention_scores = attention_weights.sum(dim=2)  # [batch_size, seq_len]
    #             attention_scores = attention_scores / attention_scores.sum(dim=1, keepdim=True)

    #             # MoD处理
    #             top_k = int(seq_len * self.top_k_ratio)
    #             _, top_k_indices = torch.topk(attention_scores, top_k, dim=1)
    #             output = hidden_states.clone()

    #             for b in range(batch_size):
    #                 curr_indices = top_k_indices[b]
    #                 mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden_states.device)
    #                 mask[curr_indices] = True
    #                 output[b][mask] = hidden_states[b][mask]
    #                 output[b][~mask] = hidden_states[b][~mask]

    #             hidden_states = output

    #         if output_attentions:
    #             all_attentions = all_attentions + (layer_outputs[1],)

    #     if output_hidden_states:
    #         all_hidden_states = all_hidden_states + (hidden_states,)

    #     if not return_dict:
    #         return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

    #     return BaseModelOutput(
    #         last_hidden_state=hidden_states,
    #         hidden_states=all_hidden_states,
    #         attentions=all_attentions
    #     )

class ModifiedCLIPVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = CLIPVisionTransformerWithMoD(config)
        self.config = config

    @property
    def device(self):
        return next(self.vision_model.parameters()).device

    @property
    def dtype(self):
        return next(self.vision_model.parameters()).dtype

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=True, return_dict=True):
        return self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        self.vision_tower = ModifiedCLIPVisionModel(config)
        self.vision_tower.requires_grad_(False)

        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

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



# class CLIPVisionTowerS2(CLIPVisionTower):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__(vision_tower, args, delay_load)

#         self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
#         self.s2_scales = list(map(int, self.s2_scales.split(',')))
#         self.s2_scales.sort()
#         self.s2_split_size = self.s2_scales[0]
#         self.s2_image_size = self.s2_scales[-1]

#         try:
#             from s2wrapper import forward as multiscale_forward
#         except ImportError:
#             raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
#         self.multiscale_forward = multiscale_forward

#         if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.image_processor.size['shortest_edge'] = self.s2_image_size
#             self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
#             return

#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
#         self.vision_tower.requires_grad_(False)

#         self.image_processor.size['shortest_edge'] = self.s2_image_size
#         self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

#         self.is_loaded = True

#     @torch.no_grad()
#     def forward_feature(self, images):
#         image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#         image_features = self.feature_select(image_forward_outs).to(images.dtype)
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
#                 image_features.append(image_feature)
#         else:
#             image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

#         return image_features

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size * len(self.s2_scales)
