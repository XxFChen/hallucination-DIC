# #    Copyright 2023 Haotian Liu
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.


# from abc import ABC, abstractmethod

# import torch
# import torch.nn as nn

# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_projector.builder import build_vision_projector

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# from llava.mm_utils import get_anyres_image_grid_shape


# class LlavaMetaModel:

#     def __init__(self, config):
#         super(LlavaMetaModel, self).__init__(config)

#         if hasattr(config, "mm_vision_tower"):
#             self.vision_tower = build_vision_tower(config, delay_load=True)
#             self.mm_projector = build_vision_projector(config)

#             if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
#                 self.image_newline = nn.Parameter(
#                     torch.empty(config.hidden_size, dtype=self.dtype)
#                 )
        

#     def get_vision_tower(self):
#         vision_tower = getattr(self, 'vision_tower', None)
#         if type(vision_tower) is list:
#             vision_tower = vision_tower[0]
#         return vision_tower

#     def initialize_vision_modules(self, model_args, fsdp=None):
#         vision_tower = model_args.vision_tower
#         mm_vision_select_layer = model_args.mm_vision_select_layer
#         mm_vision_select_feature = model_args.mm_vision_select_feature
#         pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
#         mm_patch_merge_type = model_args.mm_patch_merge_type

#         self.config.mm_vision_tower = vision_tower

#         if self.get_vision_tower() is None:
#             vision_tower = build_vision_tower(model_args)

#             if fsdp is not None and len(fsdp) > 0:
#                 self.vision_tower = [vision_tower]
#             else:
#                 self.vision_tower = vision_tower
#         else:
#             if fsdp is not None and len(fsdp) > 0:
#                 vision_tower = self.vision_tower[0]
#             else:
#                 vision_tower = self.vision_tower
#             vision_tower.load_model()

#         self.config.use_mm_proj = True
#         self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
#         self.config.mm_hidden_size = vision_tower.hidden_size
#         self.config.mm_vision_select_layer = mm_vision_select_layer
#         self.config.mm_vision_select_feature = mm_vision_select_feature
#         self.config.mm_patch_merge_type = mm_patch_merge_type

#         if getattr(self, 'mm_projector', None) is None:
#             self.mm_projector = build_vision_projector(self.config)

#             if 'unpad' in mm_patch_merge_type:
#                 embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
#                 self.image_newline = nn.Parameter(
#                     torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
#                 )
#         else:
#             # In case it is frozen by LoRA
#             for p in self.mm_projector.parameters():
#                 p.requires_grad = True

#         if pretrain_mm_mlp_adapter is not None:
#             mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
#             def get_w(weights, keyword):
#                 return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

#             self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


# def unpad_image(tensor, original_size):
#     """
#     Unpads a PyTorch tensor of a padded and resized image.

#     Args:
#     tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
#     original_size (tuple): The original size of PIL image (width, height).

#     Returns:
#     torch.Tensor: The unpadded image tensor.
#     """
#     original_width, original_height = original_size
#     current_height, current_width = tensor.shape[1:]

#     original_aspect_ratio = original_width / original_height
#     current_aspect_ratio = current_width / current_height

#     if original_aspect_ratio > current_aspect_ratio:
#         scale_factor = current_width / original_width
#         new_height = int(original_height * scale_factor)
#         padding = (current_height - new_height) // 2
#         unpadded_tensor = tensor[:, padding:current_height - padding, :]
#     else:
#         scale_factor = current_height / original_height
#         new_width = int(original_width * scale_factor)
#         padding = (current_width - new_width) // 2
#         unpadded_tensor = tensor[:, :, padding:current_width - padding]

#     return unpadded_tensor


# class LlavaMetaForCausalLM(ABC):


#     @abstractmethod
#     def get_model(self):
#         pass

#     def get_vision_tower(self):
#         return self.get_model().get_vision_tower()

#     def encode_images(self, images):
#         image_features = self.get_model().get_vision_tower()(images)
#         image_features = self.get_model().mm_projector(image_features)
#         return image_features

#     def prepare_inputs_labels_for_multimodal(
#         self, input_ids, position_ids, attention_mask, past_key_values, labels,
#         images, image_sizes=None
#     ):
#         vision_tower = self.get_vision_tower()
#         if vision_tower is None or images is None or input_ids.shape[1] == 1:
#             return input_ids, position_ids, attention_mask, past_key_values, None, labels

#         if type(images) is list or images.ndim == 5:
#             if type(images) is list:
#                 images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
#             concat_images = torch.cat([image for image in images], dim=0)
#             image_features = self.encode_images(concat_images)
            
#             token_importance = image_features.mean(dim=-1)  # Average across features
#             top_tokens = torch.topk(token_importance, k=int(image_features.size(1) * 0.3), dim=1).indices  # Top 20% tokens
            
#             # 克隆和增强特征
#             enhanced_features = image_features.clone()
#             enhancement_factor = 1.3  # Adjust this factor based on your needs
#             for i in range(image_features.size(0)):
#                 enhanced_features[i, top_tokens[i]] *= enhancement_factor
                
#             # 使用增强后的特征进行分割
#             split_sizes = [image.shape[0] for image in images]
#             image_features = list(torch.split(enhanced_features, split_sizes, dim=0))
            
#             mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
#             image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
#             if mm_patch_merge_type == 'flat':
#                 image_features = [x.flatten(0, 1) for x in image_features]
#             elif mm_patch_merge_type.startswith('spatial'):
#                 new_image_features = []
#                 for image_idx, image_feature in enumerate(image_features):
#                     if image_feature.shape[0] > 1:
#                         base_image_feature = image_feature[0]
#                         image_feature = image_feature[1:]
#                         height = width = self.get_vision_tower().num_patches_per_side
#                         assert height * width == base_image_feature.shape[0]
#                         if image_aspect_ratio == 'anyres':
#                             num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
#                             image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
#                         else:
#                             raise NotImplementedError
#                         if 'unpad' in mm_patch_merge_type:
#                             image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                             image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                             image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                             image_feature = torch.cat((
#                                 image_feature,
#                                 self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
#                             ), dim=-1)
#                             image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                         else:
#                             image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
#                             image_feature = image_feature.flatten(0, 3)
#                         image_feature = torch.cat((base_image_feature, image_feature), dim=0)
#                     else:
#                         image_feature = image_feature[0]
#                         if 'unpad' in mm_patch_merge_type:
#                             image_feature = torch.cat((
#                                 image_feature,
#                                 self.model.image_newline[None].to(image_feature.device)
#                             ), dim=0)
#                     new_image_features.append(image_feature)
#                 image_features = new_image_features
#             else:
#                 raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
#         else:
#             image_features = self.encode_images(images)

#         # TODO: image start / end is not implemented here to support pretraining.
#         if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
#             raise NotImplementedError

#         # Let's just add dummy tensors if they do not exist,
#         # it is a headache to deal with None all the time.
#         # But it is not ideal, and if you have a better idea,
#         # please open an issue / submit a PR, thanks.
#         _labels = labels
#         _position_ids = position_ids
#         _attention_mask = attention_mask
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
#         else:
#             attention_mask = attention_mask.bool()
#         if position_ids is None:
#             position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
#         if labels is None:
#             labels = torch.full_like(input_ids, IGNORE_INDEX)

#         # remove the padding using attention_mask -- FIXME
#         _input_ids = input_ids
#         input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
#         labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

#         new_input_embeds = []
#         new_labels = []
#         cur_image_idx = 0
#         for batch_idx, cur_input_ids in enumerate(input_ids):
#             num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
#             if num_images == 0:
#                 cur_image_features = image_features[cur_image_idx]
#                 cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
#                 cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
#                 new_input_embeds.append(cur_input_embeds)
#                 new_labels.append(labels[batch_idx])
#                 cur_image_idx += 1
#                 continue

#             image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
#             cur_input_ids_noim = []
#             cur_labels = labels[batch_idx]
#             cur_labels_noim = []
#             for i in range(len(image_token_indices) - 1):
#                 cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
#                 cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
#             split_sizes = [x.shape[0] for x in cur_labels_noim]
#             cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
#             cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
#             cur_new_input_embeds = []
#             cur_new_labels = []

#             for i in range(num_images + 1):
#                 cur_new_input_embeds.append(cur_input_embeds_no_im[i])
#                 cur_new_labels.append(cur_labels_noim[i])
#                 if i < num_images:
#                     cur_image_features = image_features[cur_image_idx]
#                     cur_image_idx += 1
#                     cur_new_input_embeds.append(cur_image_features)
#                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

#             cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

#             cur_new_input_embeds = torch.cat(cur_new_input_embeds)
#             cur_new_labels = torch.cat(cur_new_labels)

#             new_input_embeds.append(cur_new_input_embeds)
#             new_labels.append(cur_new_labels)

#         # Truncate sequences to max length as image embeddings can make the sequence longer
#         tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
#         if tokenizer_model_max_length is not None:
#             new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
#             new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

#         # Combine them
#         max_len = max(x.shape[0] for x in new_input_embeds)
#         batch_size = len(new_input_embeds)

#         new_input_embeds_padded = []
#         new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
#         attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
#         position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

#         for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
#             cur_len = cur_new_embed.shape[0]
#             if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
#                 new_input_embeds_padded.append(torch.cat((
#                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
#                     cur_new_embed
#                 ), dim=0))
#                 if cur_len > 0:
#                     new_labels_padded[i, -cur_len:] = cur_new_labels
#                     attention_mask[i, -cur_len:] = True
#                     position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
#             else:
#                 new_input_embeds_padded.append(torch.cat((
#                     cur_new_embed,
#                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
#                 ), dim=0))
#                 if cur_len > 0:
#                     new_labels_padded[i, :cur_len] = cur_new_labels
#                     attention_mask[i, :cur_len] = True
#                     position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

#         new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

#         if _labels is None:
#             new_labels = None
#         else:
#             new_labels = new_labels_padded

#         if _attention_mask is None:
#             attention_mask = None
#         else:
#             attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

#         if _position_ids is None:
#             position_ids = None

#         return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

#     def initialize_vision_tokenizer(self, model_args, tokenizer):
#         if model_args.mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#             self.resize_token_embeddings(len(tokenizer))

#         if model_args.mm_use_im_start_end:
#             num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#             self.resize_token_embeddings(len(tokenizer))

#             if num_new_tokens > 0:
#                 input_embeddings = self.get_input_embeddings().weight.data
#                 output_embeddings = self.get_output_embeddings().weight.data

#                 input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
#                     dim=0, keepdim=True)
#                 output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
#                     dim=0, keepdim=True)

#                 input_embeddings[-num_new_tokens:] = input_embeddings_avg
#                 output_embeddings[-num_new_tokens:] = output_embeddings_avg

#             if model_args.tune_mm_mlp_adapter:
#                 for p in self.get_input_embeddings().parameters():
#                     p.requires_grad = True
#                 for p in self.get_output_embeddings().parameters():
#                     p.requires_grad = False

#             if model_args.pretrain_mm_mlp_adapter:
#                 mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
#                 embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
#                 assert num_new_tokens == 2
#                 if input_embeddings.shape == embed_tokens_weight.shape:
#                     input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
#                 elif embed_tokens_weight.shape[0] == num_new_tokens:
#                     input_embeddings[-num_new_tokens:] = embed_tokens_weight
#                 else:
#                     raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
#         elif model_args.mm_use_im_patch_token:
#             if model_args.tune_mm_mlp_adapter:
#                 for p in self.get_input_embeddings().parameters():
#                     p.requires_grad = False
#                 for p in self.get_output_embeddings().parameters():
#                     p.requires_grad = False
                    


#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor





class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def compute_token_importance_via_self_attention(self, image_features):
        """
        通过 self-attention 计算 image token 的重要性。
        
        Args:
            image_features (torch.Tensor): Image tokens 的特征表示，形状为 (batch_size, num_tokens, hidden_size)。

        Returns:
            importance_scores (torch.Tensor): 每个 image token 的重要性分数，形状为 (batch_size, num_tokens)。
        """
        # Step 1: 计算 Self-Attention 权重
        # 对 image_features 执行自注意力机制，Q, K, V 都来源于 image_features 本身
        query = image_features  # Q
        key = image_features  # K
        value = image_features  # V

        # 计算 Q 和 K 的相似性 (scaled dot product attention)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_tokens, num_tokens)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(image_features.size(-1), dtype=torch.float32))  # scaling

        # 使用 softmax 获得 attention 权重
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, num_tokens, num_tokens)

        feature_strength = torch.norm(image_features, p=2, dim=-1)
   
        importance_scores = feature_strength * attention_weights.mean(dim=1)
        # Step 2: 计算每个 token 的平均重要性分数
        # 取平均值，衡量每个 token 对所有其他 tokens 的重要性
        # importance_scores = attention_weights.mean(dim=1)  # (batch_size, num_tokens)
        
        return importance_scores
    
    def enhance_image_tokens_in_layers(self,image_features, layer_index, enhance_layers, enhancement_factors):
        """
        在指定的 LLM 层增强 image token。
        
        Args:
            image_features (torch.Tensor): image token 的特征，形状为 (batch_size, num_tokens, hidden_size)。
            layer_index (int): 当前层的索引。
            enhance_layers (list): 需要增强的层索引列表。
            enhancement_factors (dict): 不同类别的 token 的增强因子。

        Returns:
            enhanced_features (torch.Tensor): 增强后的 image token 特征。
        """
        if layer_index in enhance_layers:
            # Step 1: 计算 token 的重要性分数
            token_importance = torch.norm(image_features, p=2, dim=-1)  # (batch_size, num_tokens)

            # Step 2: 分类 token
            num_tokens = token_importance.size(1)
            top_20_percent = int(num_tokens * 0.2)
            top_50_percent = int(num_tokens * 0.5)
            sorted_indices = torch.argsort(token_importance, dim=1, descending=True)

            # Step 3: 应用不同类别的增强因子
            enhanced_features = image_features.clone()
            for i in range(image_features.size(0)):  # 遍历每个 batch
                # Top 20%: 最重要的 tokens
                enhanced_features[i, sorted_indices[i, :top_20_percent]] *= enhancement_factors["important"]

                # 20%-50%: 一般重要的 tokens
                enhanced_features[i, sorted_indices[i, top_20_percent:top_50_percent]] *= enhancement_factors["moderately_important"]

                # 50%-100%: 不重要的 tokens
                enhanced_features[i, sorted_indices[i, top_50_percent:]] *= enhancement_factors["less_important"]

            # Step 4: 对增强后的特征归一化，防止特征值过大
            # enhanced_features = enhanced_features / torch.norm(enhanced_features, p=2, dim=-1, keepdim=True)

            return enhanced_features
        else:
            # 当前层不需要增强，直接返回原始特征
            return image_features


    def visualize_features(self, original_features, enhanced_features, images, save_dir="./feature_visualizations"):
        """
        Visualize the original and enhanced features with the original image below each feature map.
        
        Args:
            original_features (torch.Tensor): Original image features of shape (batch_size, num_tokens, hidden_size).
            enhanced_features (torch.Tensor): Enhanced image features of shape (batch_size, num_tokens, hidden_size).
            images (torch.Tensor): Original input images of shape (batch_size, num_images, 3, height, width).
            save_dir (str): Directory where visualizations will be saved.
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Step 1: Compute the feature strength (e.g., L2 norm)
        original_strength = torch.norm(original_features, p=2, dim=-1).cpu().numpy()  # Shape: (batch_size, num_tokens)
        enhanced_strength = torch.norm(enhanced_features, p=2, dim=-1).cpu().numpy()  # Shape: (batch_size, num_tokens)

        # Step 2: Reshape to square grids if possible
        batch_size, num_tokens = original_strength.shape
        token_per_dim = int(np.sqrt(num_tokens))
        if token_per_dim ** 2 != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens into a square grid. Make sure num_tokens is a perfect square.")

        original_strength_2d = original_strength.reshape(batch_size, token_per_dim, token_per_dim)
        enhanced_strength_2d = enhanced_strength.reshape(batch_size, token_per_dim, token_per_dim)

        # Step 3: Iterate over batches and save visualizations
        for i in range(batch_size):
            fig = plt.figure(figsize=(10, 15))
            gs = GridSpec(3, 1, height_ratios=[1, 1, 1])  # Define grid with 3 rows (image, original features, enhanced features)

            # Original features visualization with values
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(original_strength_2d[i], cmap="jet", vmin=0, vmax=np.percentile(original_strength_2d[i], 99))
            ax1.set_title("Original Features")
            ax1.axis("off")
            for x in range(token_per_dim):
                for y in range(token_per_dim):
                    ax1.text(y, x, f"{original_strength_2d[i][x, y]:.2f}", ha="center", va="center", fontsize=6, color="white")

            # Enhanced features visualization with values
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.imshow(enhanced_strength_2d[i], cmap="jet", vmin=0, vmax=np.percentile(enhanced_strength_2d[i], 99))
            ax2.set_title("Enhanced Features")
            ax2.axis("off")
            for x in range(token_per_dim):
                for y in range(token_per_dim):
                    ax2.text(y, x, f"{enhanced_strength_2d[i][x, y]:.2f}", ha="center", va="center", fontsize=6, color="white")

            # Original image below the features
            ax3 = fig.add_subplot(gs[2, 0])
            image_np = images[i][0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            image_np = image_np.astype(np.float32)
            image_np = np.clip(image_np, 0, 1)  # Normalize to [0, 1]
            ax3.imshow(image_np)
            ax3.set_title("Original Image")
            ax3.axis("off")

            # Save the visualization
            save_path = os.path.join(save_dir, f"feature_visualization_batch_{i}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free memory
            print(f"Saved visualization to {save_path}")




            
        
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
                # Step 1: 定义需要增强的层和增强因子
            enhance_layers = [0]  # 前 6 层进行增强
            enhancement_factors = {
                "important": 1.1,  # Top 20%
                "moderately_important": 1.2,  # 20%-50%
                "less_important": 1.0  # 50%-100%
            }

            # Step 2: 对指定层增强特征
            for layer_index in range(32):  # 假设模型有 6 层
                image_features = self.enhance_image_tokens_in_layers(image_features, layer_index, enhance_layers, enhancement_factors)

            # 使用增强后的特征进行分割
            split_sizes = [image.shape[0] for image in images]
            image_features = list(torch.split(image_features, split_sizes, dim=0))

            
            # # Step 1: 计算 token 的重要性分数
            # token_importance = self.compute_token_importance_via_self_attention(image_features)  # 每个 token 的重要性分数 (batch_size, num_tokens)

            # # Step 2: 排序 tokens 并分层
            # num_tokens = image_features.size(1)
            # top_20_percent = int(num_tokens * 0.15)  # Top 20%
            # top_50_percent = int(num_tokens * 0.4)  # Top 50%
            
            # # 对每个 batch 的 token 排序
            # sorted_indices = torch.argsort(token_importance, dim=1, descending=True)  # 从大到小排序
            
            # # 初始化增强因子矩阵
            # enhancement_factors = torch.ones_like(token_importance)  # 初始化所有 token 的增强因子为 1.0
            
            # for i in range(image_features.size(0)):  # 遍历每个 batch
            #     # Top 20%: 最重要的 tokens，轻微增强
            #     enhancement_factors[i, sorted_indices[i, :top_20_percent]] = 1.1  # 增强因子为 1.2
                
            #     # 20%-50%: 一般重要的 tokens，适度增强
            #     enhancement_factors[i, sorted_indices[i, top_20_percent:top_50_percent]] = 1.2  # 增强因子为 1.5
                
            #     # 50%-100%: 不重要的 tokens，轻微增强但不超过一般重要 tokens
            #     enhancement_factors[i, sorted_indices[i, top_50_percent:]] = 1.0  # 增强因子为 1.1
            
            # # Step 3: 对 token 特征应用增强因子
            # enhanced_features = image_features * enhancement_factors.unsqueeze(-1)  # 按最后一维广播增强因子
            # # 假设 image_features 是增强后的特征 (batch_size, num_tokens, hidden_size)

            # # Step 4: 使用增强后的特征进行分割
            # split_sizes = [image.shape[0] for image in images]
            # image_features = list(torch.split(enhanced_features, split_sizes, dim=0))
            


            
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False