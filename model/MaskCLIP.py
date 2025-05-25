import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from IMDLBenCo.registry import MODELS
from model.prompt_learner import TextEncoder, PromptLearner
from model.clip_utils import SideAdapterNetwork


def DICE_loss(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class CLSAttentionAggregator(nn.Module):
    def __init__(self, feature_dim=1024, out_dim=768, num_heads=8, num_layers=4, dropout_rate=0.1):
        super(CLSAttentionAggregator, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'multihead_attn': nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True),
                'layer_norm': nn.LayerNorm(feature_dim),
                'dropout': nn.Dropout(dropout_rate)
            })
            for _ in range(num_layers)
        ])

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, out_dim, bias=False),
        )

    def forward(self, x):
        bs, num_layers_inp, feature_dim = x.size()
        x = x.view(bs, num_layers_inp, feature_dim)

        for layer in self.layers:
            attn_output, _ = layer['multihead_attn'](x, x, x)
            x = layer['layer_norm'](attn_output + x)
            x = layer['dropout'](x)

        x = torch.mean(x, dim=1)

        x = self.projector(x)
        return x


class FeatureHook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


main_keys = {
    'ViTL': {
        'model_name': 'ViT-L/14',
        'resolution': 512,
        "language_ctx": 10,
        "language_depth": 5,
        'adapter_length': 5,
        'selected_layers': [2, 4, 8, 12, 16, 20, 23],
        'fusion_map': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    },

    # add your own setting here
}


@MODELS.register_module()
class MaskCLIP(nn.Module):
    def __init__(self, model_setting_name):
        super().__init__()
        settings = main_keys[model_setting_name]
        self.selected_layers = settings['selected_layers']
        self.resolution = settings['resolution']
        self.clip, _ = clip.load(settings['model_name'])
        self.clip.float()

        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        self.hooks = [
            FeatureHook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        visemb_dim = self.clip.visual.conv1.out_channels
        align_dim = self.clip.ln_final.weight.shape[0]
        self.aggregator = SideAdapterNetwork(
            img_size=self.resolution,
            align_channels=align_dim,
            clip_channels=visemb_dim,
            fusion_map=settings['fusion_map'],
        )
        self.cls_aggregator = CLSAttentionAggregator(feature_dim=visemb_dim, out_dim=align_dim, num_heads=8,
                                                     num_layers=1, dropout_rate=0.1)
        self.prompt_learner = PromptLearner(align_dim, settings['language_ctx'], settings['language_depth'],
                                            self.clip.dtype)



        self.ce_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()

        self.edge_lambda = 20

    def encode_text(self, x, tokenized_prompts):
        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.clip.text_projection
        return x

    def forward(self, image, mask, label, edge_mask, *args, **kwargs):
        clip_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=True)
        with torch.no_grad():
            self.clip.encode_image(clip_image)
            features = torch.stack([h.output for h in self.hooks], dim=2)
            selected_features = [features[:, :, i, :] for i in self.selected_layers]
            selected_features = torch.stack(selected_features, dim=2)
        N, B, L, C = selected_features.shape  # [16:56:43.462182] torch.Size([257, 8, 1, 1024])

        cls_features = selected_features[0, :, :, :]  # [bs, 24, 1024]
        patch_features = selected_features[1:, :, :, :]  # [576, bs, 4, 1024]
        patch_features = [
            patch_features[:, :, i, :].permute(1, 2, 0).reshape(B, C, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)))
            for i in range(patch_features.shape[2])]

        text = ['an image'] * 2
        prompts, tokenized_prompts = self.prompt_learner(self.clip, text, image.device)
        text_features = self.encode_text(prompts, tokenized_prompts)
        text_features = torch.chunk(text_features, dim=0, chunks=2)
        text_features_mean = torch.stack([text_features[0].mean(0), text_features[1].mean(0)], dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)

        mask_pred = self.aggregator(image, text_features_mean, patch_features)
        mask_pred = F.interpolate(mask_pred, size=image.shape[2:], mode='bilinear', align_corners=True)

        edge_loss = F.binary_cross_entropy_with_logits(input=mask_pred, target=mask,
                                                       weight=edge_mask) * self.edge_lambda

        mask_pred = torch.sigmoid(mask_pred)
        # dice_loss = DICE_loss(mask_pred, mask)
        bce_loss = self.bce_criterion(mask_pred.view(-1), mask.view(-1).float())

        cls_features = self.cls_aggregator(cls_features)
        probs = cls_features @ text_features_mean.t()
        ce_loss = self.ce_criterion(probs, label)
        pred_label = torch.argmax(probs, dim=1)

        loss = ce_loss + bce_loss + edge_loss # + dice_loss

        output_dict = {
            "backward_loss": loss,
            "pred_mask": mask_pred,
            "pred_label": pred_label,
            "visual_loss": {
                "loss_ce": ce_loss,
                "loss_bce": bce_loss,
                "loss_edge": edge_loss,
                # "loss_dice": dice_loss,
                "combined_loss": loss
            },
            "visual_image": {
                "pred_mask": mask_pred
            }
        }

        return output_dict






