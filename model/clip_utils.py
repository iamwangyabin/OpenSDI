import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Dict, List, Tuple
import warnings
from model.mae import ViT, get_abs_pos


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# https://github.com/SunnyHaze/IML-ViT/blob/8aa67af93e40e4671e4ac5cc69ae5872210d7249/modules/window_attention_ViT.py
class MaskDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 decoder_dim=256,
                 norm="LN",
                 num_heads=8):
        super().__init__()
        backbone_dim = in_channels
        scale_factors = (4.0, 2.0, 1.0, 0.5)

        # Cross-attention components
        self.num_heads = num_heads
        self.scale = (decoder_dim // num_heads) ** -0.5
        self.to_q = nn.Linear(decoder_dim, decoder_dim)
        self.to_kv = nn.Linear(decoder_dim, decoder_dim * 2)
        self.proj = nn.Linear(decoder_dim, decoder_dim)

        # Multi-scale stages
        self.stages = nn.ModuleList()
        for scale in scale_factors:
            layers = []
            if scale == 4.0:
                layers.extend([
                    nn.ConvTranspose2d(backbone_dim, backbone_dim // 2, kernel_size=2, stride=2),
                    LayerNorm(backbone_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(backbone_dim // 2, backbone_dim // 4, kernel_size=2, stride=2),
                ])
                cur_channels = backbone_dim // 4
            elif scale == 2.0:
                layers.append(nn.ConvTranspose2d(backbone_dim, backbone_dim // 2, kernel_size=2, stride=2))
                cur_channels = backbone_dim // 2
            elif scale == 1.0:
                layers = []
                cur_channels = backbone_dim
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                cur_channels = backbone_dim

            # Common layers for each stage
            layers.extend([
                nn.Conv2d(cur_channels, decoder_dim, 1),
                LayerNorm(decoder_dim),
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
                LayerNorm(decoder_dim)
            ])

            self.stages.append(nn.Sequential(*layers))

        # Prediction head
        self.linear_fuse = nn.Conv2d(decoder_dim * len(scale_factors), decoder_dim, 1)

        if norm == "LN":
            self.norm = LayerNorm(decoder_dim)
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(decoder_dim)
        else:
            self.norm = nn.InstanceNorm2d(decoder_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout(0.1)
        self.predict = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def cross_attention(self, x, query):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        query = query.unsqueeze(1)  # B, 1, C

        q = self.to_q(query)
        k, v = self.to_kv(x_flat).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, query.shape[1], self.num_heads, -1).transpose(1, 2)  # B, heads, x, head_dim
        k = k.reshape(B, H * W, self.num_heads, -1).transpose(1, 2)  # B, heads, HW, head_dim
        v = v.reshape(B, H * W, self.num_heads, -1).transpose(1, 2)  # B, heads, HW, head_dim

        # Attention
        # import pdb;pdb.set_trace()

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, x, HW
        attn = attn.softmax(dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2)  # B, 1, heads, head_dim
        out = out.reshape(B, query.shape[1], C)
        out = self.proj(out)

        # Reshape back to spatial dimensions
        return out.reshape(B, C, 1, 1).expand(B, C, H, W)

    def forward(self, query: torch.Tensor, x: torch.Tensor):
        # Process multi-scale features
        features = [stage(x) for stage in self.stages]
        n, _, h, w = features[0].shape
        aligned_features = [
            F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            for feat in features
        ]
        x = self.linear_fuse(torch.cat(aligned_features, dim=1))

        # Apply cross-attention with query
        attn_out = self.cross_attention(x, query)
        x = x + attn_out  # Residual connection

        x = self.norm(x)
        x = self.dropout(x)
        x = self.predict(x)
        return x


class AttentionFusion(nn.Module):
    def __init__(self, clip_dim=1024, x_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        assert self.head_dim * num_heads == x_dim, "x_dim must be divisible by num_heads"

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(clip_dim, x_dim, kernel_size=1),
            LayerNorm(x_dim)
        )

        self.query_proj = nn.Linear(x_dim, x_dim)
        self.key_proj = nn.Linear(x_dim, x_dim)
        self.value_proj = nn.Linear(x_dim, x_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(x_dim, x_dim),
            nn.LayerNorm(x_dim)
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, x, clip_features):
        batch_size = x.shape[0]
        target_h, target_w = x.shape[1:3]

        # Resize and adjust CLIP features
        clip_resized = F.interpolate(
            clip_features,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        clip_adjusted = self.channel_adjust(clip_resized)
        clip_adjusted = clip_adjusted.permute(0, 2, 3, 1)  # [B, H, W, x_dim]

        # Reshape for multi-head attention
        seq_len = target_h * target_w
        query = clip_adjusted.reshape(batch_size, seq_len, -1)
        key = x.reshape(batch_size, seq_len, -1)
        value = x.reshape(batch_size, seq_len, -1)

        # Project and reshape for multi-head attention
        query = self.query_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention calculation
        query = query.permute(0, 2, 1, 3)  # [B, num_heads, seq_len, head_dim]
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention and reshape
        attended_features = torch.matmul(attention_weights, value)  # [B, num_heads, seq_len, head_dim]
        attended_features = attended_features.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Final projection and normalization
        fused = self.output_proj(attended_features)
        fused = fused.reshape(batch_size, target_h, target_w, -1)

        return fused


class SideAdapterNetwork(nn.Module):
    def __init__(
            self,
            img_size,
            align_channels,
            clip_channels,
            fusion_map={},
    ):
        super().__init__()

        self.vit_model = ViT(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            use_rel_pos=True,
        )
        self.vit_model.load_state_dict(
            torch.load('weights/mae_pretrain_vit_base.pth', map_location='cpu')['model'],
            strict=False
        )
        self.text_projection = nn.Linear(align_channels, 256)

        # # Store other parameters
        self.fusion_layers = nn.ModuleDict(
            {
                f"layer_{tgt_idx}": AttentionFusion(clip_channels, 768)
                for tgt_idx, src_idx in fusion_map.items()
            }
        )
        self.fusion_map = fusion_map

        self.mask_decoder = MaskDecoder(
            in_channels=768,
        )

    def forward(self, image: torch.Tensor, text_features: torch.Tensor, clip_features: List[torch.Tensor]):
        image_features = self.forward_features(image, clip_features)
        # only select the fake cls token [:, 1, :]
        text_features = self.text_projection(text_features.expand(image_features.shape[0], -1, -1))[:, 1, :]
        return self.mask_decoder(text_features, image_features)

    def forward_features(self, image: torch.Tensor, clip_features: List[torch.Tensor]):
        x = self.vit_model.patch_embed(image)
        x = x + get_abs_pos(self.vit_model.pos_embed, self.vit_model.pretrain_use_cls_token, (x.shape[1], x.shape[2]))
        for i, blk in enumerate(self.vit_model.blocks):
            x = blk(x)
            x = self.fuse(i, x, clip_features)
        return x.permute(0, 3, 1, 2)

    def fuse(self, block_idx: int, x: torch.Tensor, clip_features: List[torch.Tensor]):
        if block_idx in self.fusion_map.keys():
            x = self.fusion_layers[f"layer_{block_idx}"](x, clip_features[self.fusion_map[block_idx]]) + x
        return x

