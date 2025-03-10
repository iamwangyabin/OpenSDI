import os
import json
import time
import types
import inspect
import argparse
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.datasets import ManiDataset, JsonDataset
from IMDLBenCo.transforms import get_albu_transforms
from IMDLBenCo.evaluation import PixelF1, ImageF1, PixelIOU, ImageAUC, ImageAccuracy, PixelAccuracy
from IMDLBenCo.training_scripts.tester import test_one_epoch

from data.hf_datasets import HFDataset
from model.MaskCLIP import MaskCLIP


def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBench testing launch!', add_help=True)
    # Model name
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)

    # 可以接受label的模型是否接受label输入，并启用相关的loss。
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')

    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')

    parser.add_argument('--if_resizing', action='store_true',
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--test_data_json', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset json, should be a json file contains many datasets. Details are in readme.md')
    # ------------------------------------
    # Testing 相关的参数
    parser.add_argument('--checkpoint_path', default='/root/workspace/IML-ViT/output_dir', type=str,
                        help='path to the dir where saving checkpoints')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--no_model_eval', action='store_true',
                        help='Do not use model.eval() during testing.')

    # ----输出的日志相关的参数-----------
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    args, remaining_args = parser.parse_known_args()
    model_class = MODELS.get(args.model)
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)
    return args, model_args

def setup_model(args, model_args, device):
    model = MODELS.get(args.model)
    model_init_params = inspect.signature(model.__init__).parameters if not isinstance(model, (types.FunctionType, types.MethodType)) else inspect.signature(model).parameters
    combined_args = {k: v for k, v in {**vars(args), **vars(model_args)}.items() if k in model_init_params}
    model = model(**combined_args)
    model.to(device)
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    
    return model

def post_transforms(type_='train', output_size=(1024, 1024),
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]):
    assert type_ in ['train', 'test', 'pad', 'resize'], "type_ must be 'train' or 'test' of 'pad' "
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2
    trans = None
    if type_ == 'pad':
        trans = albu.Compose([
            albu.PadIfNeeded(min_height=output_size[0], min_width=output_size[1], border_mode=0, value=0, position='top_left', mask_value=0),
            albu.Normalize(mean=mean, std=std),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    if type_ == 'resize':
        trans = albu.Compose([
            albu.Resize(output_size[0], output_size[1]),
            albu.Normalize(mean=mean, std=std),
            albu.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])


    return trans


def setup_dataset(args, name, pixel,split, split_file=None):
    post_function = POSTFUNCS.get(f"{args.model}_post_func".lower()) if POSTFUNCS.has(f"{args.model}_post_func".lower()) else None


    test_transform = get_albu_transforms('test')
    post_transform = post_transforms(type_="pad" if args.if_padding else "resize",
                                             output_size=(args.image_size, args.image_size))
    dataset = HFDataset(
        name,split,pixel,
        is_padding=args.if_padding,
        is_resizing=args.if_resizing,
        output_size=(args.image_size, args.image_size),
        common_transforms=test_transform,
        post_transform=post_transform,
        edge_width=args.edge_mask_width,
        # post_funcs=post_function
    )

    print(f"Dataset: {dataset}")
    print(f"Dataset size: {len(dataset)}")
    
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=False,
            drop_last=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    return data_loader

def test_on_dataset(args, model, data_loader, evaluator_list, device, log_writer):
    test_stats = test_one_epoch(
        model=model,
        data_loader=data_loader,
        evaluator_list=evaluator_list,
        device=device,
        epoch=0,
        log_writer=log_writer,
        args=args
    )

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': 0}
    if args.full_log_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    return log_stats

def main(args, model_args):
    misc.init_distributed_mode(args)
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('Job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====\n{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====\n{}".format(model_args).replace(', ', ',\n'))
    
    device = torch.device(args.device)

    model = setup_model(args, model_args, device)
    ckpt = torch.load(args.checkpoint_path, map_location='cuda')
    model.module.load_state_dict(ckpt['model'])
    model.eval()
    # model.load_state_dict(ckpt['model'])
    allresults = {}
    dataset_name = "nebula/OpenSDI_test"
    for split in ["sd15", "sd3", "sd2", "flux","sdxl"]:
        pixel=True
        evaluator_list = [PixelF1(threshold=0.5, mode="origin"), PixelIOU(), PixelAccuracy()]
        args.full_log_dir = os.path.join(args.log_dir, dataset_name, split, "pixel")
        log_writer = SummaryWriter(log_dir=args.full_log_dir) if misc.is_main_process() and args.full_log_dir else None
        data_loader_test = setup_dataset(args, dataset_name, pixel, split)
        results = test_on_dataset(args, model, data_loader_test, evaluator_list, device, log_writer)
        allresults[os.path.join(split, "pixel")] = results
        # pixel=False
        # evaluator_list = [ImageF1(threshold=0.5), ImageAccuracy()]
        # args.full_log_dir = os.path.join(args.log_dir, dataset_name, split, "image")
        # log_writer = SummaryWriter(log_dir=args.full_log_dir) if misc.is_main_process() and args.full_log_dir else None
        # data_loader_test = setup_dataset(args, dataset_name, pixel, split)
        # results = test_on_dataset(args, model, data_loader_test, evaluator_list, device, log_writer)
        # allresults[os.path.join(split, "image")] = results

    print(allresults)


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)
