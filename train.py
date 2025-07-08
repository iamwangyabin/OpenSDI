import os
import wandb
import json
import time
import types
import inspect
import datetime
import numpy as np
from pathlib import Path
import torch
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import IMDLBenCo.training_scripts.utils.misc as misc
from IMDLBenCo.registry import MODELS, POSTFUNCS
from IMDLBenCo.evaluation import PixelF1, ImageF1, ImageAccuracy, PixelAccuracy
from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.training_scripts.trainer import train_one_epoch

from data.hf_datasets import HFDataset
from model.MaskCLIP import MaskCLIP
from utils.argparser import get_args_parser

class Trainer:
    def __init__(self, args, model_args):
        self.args = args
        self.model_args = model_args
        self.device, self.log_writer = self.prepare_config()
        self.data_loader_train, self.data_loader_test = self.prepare_datasets_and_dataloaders()
        self.model, self.model_without_ddp, self.optimizer, self.loss_scaler, self.evaluator_list, self.eff_batch_size = self.prepare_model()

    def prepare_config(self):
        misc.init_distributed_mode(self.args)
        torch.multiprocessing.set_sharing_strategy('file_system')
        print('Job directory:', os.path.dirname(os.path.realpath(__file__)))

        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("=====args:=====")
        print("{}".format(self.args).replace(', ', ',\n'))
        print("=====Model args:=====")
        print("{}".format(self.model_args).replace(', ', ',\n'))
        device = torch.device(self.args.device)

        seed = self.args.seed + misc.get_rank()
        misc.seed_torch(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        log_writer = None
        if misc.get_rank() == 0 and self.args.log_dir:
            os.makedirs(self.args.log_dir, exist_ok=True)
            wandb.tensorboard.patch(root_logdir=self.args.log_dir, pytorch=False, tensorboard_x=False, save=False)
            wandb.init(project="IMDL", config={"main_cfg": vars(self.args)}, name=self.args.run_name)
            log_writer = SummaryWriter(log_dir=self.args.log_dir)
        
        return device, log_writer
    def get_albu_transforms(self, type_='train', output_size=(1024, 1024),
                            mean=[0.48145466, 0.4578275, 0.40821073], 
                            std=[0.26862954, 0.26130258, 0.27577711]):
        assert type_ in ['train', 'test', 'pad', 'resize'], "type_ must be 'train' or 'test' of 'pad' "
        trans = None
        if type_ == 'train':
            trans = albu.Compose([
                albu.RandomScale(scale_limit=0.2, p=1),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=1),
                albu.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
                albu.RandomRotate90(p=0.5),
                albu.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ])

        if type_ == 'test':
            trans = albu.Compose([
            ])

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
                ToTensorV2(transpose_mask=True)
            ])

        return trans

    def prepare_datasets_and_dataloaders(self):
        train_transform = self.get_albu_transforms('train')
        test_transform = self.get_albu_transforms('test')
        post_function = self.get_albu_transforms(type_="pad" if self.args.if_padding else "resize",
                                                 output_size=(self.args.image_size, self.args.image_size))

        post_function_sam = self.get_albu_transforms(type_="pad" if self.args.if_padding else "resize",
                                                     output_size=(1024, 1024),
                                                     mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225])

        dataset_train = HFDataset(
            self.args.data_path,
            self.args.train_split_name,
            # pixel=True,
            is_padding=self.args.if_padding,
            is_resizing=self.args.if_resizing,
            output_size=(self.args.image_size, self.args.image_size),
            common_transforms=train_transform,
            edge_width=self.args.edge_mask_width,
            post_transform=post_function,
            post_transform_sam=post_function_sam,
        )
        
        dataset_test = HFDataset(
            self.args.test_data_path,
            self.args.test_split_name,
            pixel=True,
            is_padding=self.args.if_padding,
            is_resizing=self.args.if_resizing,
            output_size=(self.args.image_size, self.args.image_size),
            common_transforms=test_transform,
            edge_width=self.args.edge_mask_width,
            post_transform=post_function,
            post_transform_sam=post_function_sam,
        )

        print(dataset_train)
        print(dataset_test)
        
        if self.args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=True
            )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_test = torch.utils.data.RandomSampler(dataset_test)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True
        )
        
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True
        )
        
        return data_loader_train, data_loader_test

    def set_requires_grad(self, model):
        for name, param in model.named_parameters():
            if ("aggregator" in name) or ("prompt" in name) or ('vit_model' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        pass



    def prepare_model(self):
        model = MODELS.get(self.args.model)
        
        if isinstance(model, (types.FunctionType, types.MethodType)):
            model_init_params = inspect.signature(model).parameters
        else:
            model_init_params = inspect.signature(model.__init__).parameters
        
        combined_args = {k: v for k, v in vars(self.args).items() if k in model_init_params}
        combined_args.update({k: v for k, v in vars(self.model_args).items() if k in model_init_params})
        model = model(**combined_args)
        
        evaluator_list = [
            PixelF1(threshold=0.5, mode="origin"),
        ]
        
        if self.args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        model.to(self.device)
        model_without_ddp = model
        
        eff_batch_size = self.args.batch_size * self.args.accum_iter * misc.get_world_size()
        
        if self.args.lr is None:
            self.args.lr = self.args.blr * eff_batch_size / 256
        
        if self.args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.gpu],
                find_unused_parameters=self.args.find_unused_parameters
            )
            model_without_ddp = model.module

        self.args.opt = 'AdamW'
        self.args.betas = (0.9, 0.999)
        self.args.momentum = 0.9
        optimizer = optim_factory.create_optimizer(self.args, model_without_ddp)

        loss_scaler = misc.NativeScalerWithGradNormCount()
        
        misc.load_model(args=self.args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        
        self.set_requires_grad(model_without_ddp)
        
        total_params = sum(p.numel() for p in model_without_ddp.parameters())
        trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        
        print("=" * 50)
        print(f"Model Parameters Summary:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Percentage trainable: {trainable_params/total_params*100:.2f}%")
        print("=" * 50)
        print("Trainable parameters:")
        non_param = []
        for name, param in model_without_ddp.named_parameters():
            if param.requires_grad:
                non_param.append(name)
        print(non_param)
        print("=" * 50)
        return model, model_without_ddp, optimizer, loss_scaler, evaluator_list, eff_batch_size


    def save_checkpoint(self, epoch):
        misc.save_model(
            args=self.args,
            model=self.model,
            model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer,
            loss_scaler=self.loss_scaler,
            epoch=epoch
        )

    def train(self):
        print(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        best_evaluate_metric_value = 0
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.data_loader_train.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.data_loader_train,
                self.optimizer, self.device, epoch, self.loss_scaler,
                log_writer=self.log_writer,
                log_per_epoch_count=self.args.log_per_epoch_count,
                args=self.args
            )
            
            if self.args.output_dir and (epoch % 50 == 0 or epoch + 1 == self.args.epochs):
                self.save_checkpoint(epoch)

            self.optimizer.zero_grad()
            
            if epoch % self.args.test_period == 0 or epoch + 1 == self.args.epochs:
                test_stats = test_one_epoch(
                    self.model,
                    data_loader=self.data_loader_test,
                    evaluator_list=self.evaluator_list,
                    device=self.device,
                    epoch=epoch,
                    log_writer=self.log_writer,
                    args=self.args
                )
                
                evaluate_metric_for_ckpt = self.evaluator_list[0].name
                evaluate_metric_value = test_stats[evaluate_metric_for_ckpt]
                
                if evaluate_metric_value > best_evaluate_metric_value:
                    best_evaluate_metric_value = evaluate_metric_value
                    print(f"Best {evaluate_metric_for_ckpt} = {best_evaluate_metric_value}")
                    self.save_checkpoint(epoch)
                
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                }
            else:
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch,
                }
            
            if self.args.output_dir and misc.is_main_process():
                if self.log_writer:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time', total_time_str)


if __name__ == '__main__':
    args, model_args = get_args_parser()
    job_str = args.exp_name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    args.output_dir = os.path.join(args.output_dir, job_str)
    args.log_dir = os.path.join(args.log_dir, job_str)
    args.run_name = job_str
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(args, model_args)
    trainer.train()


