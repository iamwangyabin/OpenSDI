
import argparse

from IMDLBenCo.registry import MODELS, POSTFUNCS
import IMDLBenCo.training_scripts.utils.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo training launch!', add_help=True)

    """
    personal added 
    """
    parser.add_argument('--exp_name', default=None, type=str,
                        help='experiment name', required=True)

    # ----输出的日志相关的参数 Dont use this-----------
    parser.add_argument('--output_dir', default='./output_dir', # Dont use this 
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',  # Dont use this 
                        help='path where to tensorboard log')
    # -----------------------

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
    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str,
                        help='dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--train_split_name', default='sd15', type=str, help='dataset split name')
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--test_split_name', default='sd15', type=str, help='dataset split name')

    # ------------------------------------
    # training related
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=16, type=int,
                        help="batch size for testing")
    parser.add_argument('--epochs', default=200, type=int)
    # Test related
    parser.add_argument('--no_model_eval', action='store_true',
                        help='Do not use model.eval() during testing.')
    parser.add_argument('--test_period', default=4, type=int,
                        help="how many epoch per testing one time")

    # 一个epoch在tensorboard中打几个loss的data point
    parser.add_argument('--log_per_epoch_count', default=20, type=int,
                        help="how many loggings (data points for loss) per testing epoch in Tensorboard")

    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='find_unused_parameters for DDP. Mainly solve issue for model with image-level prediction but not activate during training.')

    # 不启用AMP（自动精度）进行训练
    parser.add_argument('--if_not_amp', action='store_false',
                        help='Do not use automatic precision.')
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint, input the path of a ckpt.')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
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
    # 获取对应的模型类
    model_class = MODELS.get(args.model)
    # 根据模型类动态创建参数解析器
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

