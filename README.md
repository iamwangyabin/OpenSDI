# OpenSDI: Spotting Diffusion-Generated Images in the Open World
Official repository for our CVPR 2025 paper:
OpenSDI: Spotting Diffusion-Generated Images in the Open World

This paper identifies OpenSDI, a challenge for spotting diffusion-generated images in open-world settings. In response to this challenge, we define a new benchmark, the OpenSDI dataset (OpenSDID), which stands out from existing datasets due to its diverse use of large vision-language models that simulate open-world diffusion-based manipulations. Another outstanding feature of OpenSDID is its inclusion of both detection and localization tasks for images manipulated globally and locally by diffusion models. To address the OpenSDI challenge, we propose a Synergizing Pretrained Models (SPM) scheme to build up a mixture of foundation models. This approach exploits a collaboration mechanism with multiple pretrained foundation models to enhance generalization in the OpenSDI context, moving beyond traditional training by synergizing multiple pretrained models through prompting and attending strategies. Building on this scheme, we introduce MaskCLIP, an SPM-based model that aligns Contrastive Language-Image Pre-Training (CLIP) with Masked Autoencoder (MAE). Extensive evaluations on OpenSDID show that MaskCLIP significantly outperforms current state-of-the-art methods for the OpenSDI challenge, achieving remarkable relative improvements of 14.23% in IoU (14.11% in F1) and 2.05% in accuracy (2.38% in F1) compared to the second-best model in detection and localization tasks, respectively.


# Dataset
We have packaged the dataset into Hugging Face datasets for convenient distribution in the following link:

https://huggingface.co/datasets/nebula/OpenSDI_train
https://huggingface.co/datasets/nebula/OpenSDI_test

The original data (indiviual image files) will be uploaded to cloud storage later.


# Installation
```
conda create -n opensdi python=3.12 -y
conda activate opensdi
pip install -r requirements.txt
wget -P weights https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

# Train
```
train.sh
```

# Test
```
test.sh
```


# Acknowledgement

IML-ViT
