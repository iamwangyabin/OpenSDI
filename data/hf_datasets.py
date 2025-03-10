import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from IMDLBenCo.transforms import get_albu_transforms, EdgeMaskGenerator
from IMDLBenCo.registry import DATASETS


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


@DATASETS.register_module()
class HFCLIPSAMDataset(Dataset):
    def __init__(self, path, split_file, split_name, pixel=False,
                 is_padding=False,
                 is_resizing=False,
                 output_size=(1024, 1024),
                 post_transform=None,
                 post_transform_sam=None,
                 common_transforms=None,
                 edge_width=None,
                 img_loader=pil_loader,
                 ) -> None:
        super().__init__()
        self.dataset = load_dataset(path)[split_name]
        with open(split_file, 'r') as f:
            train_indices = json.load(f)
        self.dataset = self.dataset.select(train_indices)
        if pixel:
            filtered_indices = [
                i for i, key in enumerate(self.dataset['key'])
                if ('partial' in key and 'fake' in key) or (path== "nebula/tmpdata2" and 'fake' in key)
            ]
            self.dataset = self.dataset.select(filtered_indices)


        self.output_size = output_size
        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.post_transform_sam = post_transform_sam
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
        self.img_loader = img_loader 
        self.is_padding = is_padding
        self.is_resizing = is_resizing


    def _prepare_gt_img(self, tp_img, gt_path, label):
        if label == 0 and gt_path is None:
            return np.zeros((*tp_img.shape[:2], 3))
        elif label == 1 and gt_path is None:
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            return np.array(gt_path.convert('RGB'))



    def _process_masks(self, gt_img):
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]
        masks_list = [gt_img]
        if self.edge_mask_generator:
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0]
            masks_list.append(gt_img_edge)
        return masks_list

    def __getitem__(self, index):
        sample = self.dataset[index]
        tp_img = np.array(sample['image'].convert('RGB'))
        gt_img = self._prepare_gt_img(tp_img, sample['mask'], sample['label'])

        if self.common_transforms:
            res_dict = self.common_transforms(image=tp_img, mask=gt_img)
            tp_img, gt_img = res_dict['image'], res_dict['mask']
            label = 0 if np.all(gt_img == 0) else 1
        else:
            label = sample['label']

        masks_list = self._process_masks(gt_img)
        res_dict = self.post_transform(image=tp_img, masks=masks_list)
        res_dict_sam = self.post_transform_sam(image=tp_img, masks=masks_list)
        data_dict = {
            'image': res_dict['image'],
            'mask': res_dict['masks'][0].unsqueeze(0),
            'image_sam': res_dict_sam['image'],
            'mask_sam': res_dict_sam['masks'][0].unsqueeze(0),
            'label': label,
            'shape': torch.tensor(self.output_size if self.is_resizing else tp_img.shape[:2]),
            'name': sample['key']
        }

        if self.edge_mask_generator:
            data_dict['edge_mask'] = res_dict['masks'][1].unsqueeze(0)

        if self.is_padding:
            shape_mask = torch.zeros_like(data_dict['mask'])
            shape_mask[:, :data_dict['shape'][0], :data_dict['shape'][1]] = 1
            data_dict['shape_mask'] = shape_mask
        del sample
        return data_dict

    def __len__(self):
        return len(self.dataset)


from IMDLBenCo.model_zoo.cat_net.cat_net_post_function import cat_net_post_func



@DATASETS.register_module()
class HFDataset(Dataset):
    def __init__(self, path, split_name,  pixel=False,
                 is_padding=False,
                 is_resizing=False,
                 output_size=(1024, 1024),
                 post_transform=None,
                 post_transform_sam=None,
                 common_transforms=None,
                 edge_width=None,
                 img_loader=pil_loader,
                 ) -> None:
        super().__init__()
        self.dataset = load_dataset(path)[split_name]
        if pixel:
            filtered_indices = [
                i for i, key in enumerate(self.dataset['key'])
                if ('partial' in key and 'fake' in key) or (path== "nebula/tmpdata2" and 'fake' in key) or (path== "nebula/tempdata4" and 'fake' in key)
            ]
            self.dataset = self.dataset.select(filtered_indices)

        self.output_size = output_size
        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.post_transform_sam = post_transform_sam
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
        self.img_loader = img_loader
        self.is_padding = is_padding
        self.is_resizing = is_resizing
        if post_transform is None:
            self.post_transform = get_albu_transforms(type_="pad"
            if is_padding else "resize", output_size=output_size)

    def _prepare_gt_img(self, tp_img, gt_path, label):
        if label == 0:
            return np.zeros((*tp_img.shape[:2], 3))
        elif label == 1 and gt_path is None:
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            return np.array(gt_path.convert('RGB'))


    def _process_masks(self, gt_img):
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]
        masks_list = [gt_img]
        if self.edge_mask_generator:
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0]
            masks_list.append(gt_img_edge)
        return masks_list

    def __getitem__(self, index):
        sample = self.dataset[index]
        tp_img = np.array(sample['image'].convert('RGB'))
        gt_img = self._prepare_gt_img(tp_img, sample['mask'], sample['label'])

        if self.common_transforms:
            res_dict = self.common_transforms(image=tp_img, mask=gt_img)
            tp_img, gt_img = res_dict['image'], res_dict['mask']
            # import pdb;pdb.set_trace()
            if np.sum(gt_img>0):
                label=1
            else:
                label=0
            # label = sample['label']
        else:
            label = sample['label']
        # label = sample['label']

        # if label != sample['label']:
        #     print(sample['key'])
        #     print(sample['label'])
        #     print(label)

        masks_list = self._process_masks(gt_img)
        res_dict = self.post_transform(image=tp_img, masks=masks_list)

        data_dict = {
            'image': res_dict['image'],
            'mask': res_dict['masks'][0].unsqueeze(0),
            'label': label,
            'shape': torch.tensor(self.output_size if self.is_resizing else tp_img.shape[:2]),
            'name': sample['key']
        }

        if self.edge_mask_generator:
            data_dict['edge_mask'] = res_dict['masks'][1].unsqueeze(0)

        if self.is_padding:
            shape_mask = torch.zeros_like(data_dict['mask'])
            shape_mask[:, :data_dict['shape'][0], :data_dict['shape'][1]] = 1
            data_dict['shape_mask'] = shape_mask

        cat_net_post_func(data_dict)
        del sample
        del res_dict

        return data_dict

    def __len__(self):
        return len(self.dataset)
