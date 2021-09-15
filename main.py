# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from utils.utils import setup_logger
import time
import torch
# from src.models.myNet import MainNet, SetCriterion
from src.models.myNet import MainNet, SetCriterion
from src.models.transformer import Transformer
# from src.models.transformerV2 import Transformer
from src.models.backbone import Backbone
from src.data_structure.data_structure import Dataset
from src.trainer.trainer import BaseTrainer
from src.models.matcher import HungarianMatcher
from data.const import task_rel_labels, task_ner_labels
import os
import numpy as np
import random
import torch.distributed as dist
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,4'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument('--train_from_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_name', type=str, default='output_checkpoint.pth')
    parser.add_argument('--PRETRAINED_MODEL_NAME', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--final_output_dir', type=str, default='./output')
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--train_data_path', type=str, default='./data/data_mini.json')
    parser.add_argument('--test_data_path', type=str, default='./data/data_mini.json')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--WORKERS', type=int, default=4)
    parser.add_argument('--MAX_EPOCH', type=int, default=1000)
    parser.add_argument('--ENTITY_NUM', type=int, default=25)
    parser.add_argument('--RELATION_NUM', type=int, default=30)
    parser.add_argument('--MAX_LEN', type=int, default=129)
    parser.add_argument('--COST_CLASS', type=int, default=1)
    parser.add_argument('--COST_POS', type=int, default=9)
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument('--task', type=str, default='drug', choices=['drug'])
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--task_learning_rate', type=float, default=1e-4,
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--bert_learning_rate', type=float, default=1e-5,
                        help="learning rate for the BERT encoder")
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-8)
    parser.add_argument('--LR_DROP', type=int, default=200)
    parser.add_argument('--PRINT_FREQ', type=int, default=5)
    parser.add_argument('--SAVE_INTERVAL', type=int, default=10)
    parser.add_argument('--VAL_WHEN_TRAIN', type=bool, default=True)
    parser.add_argument('--SAVE_EVERY_CHECKPOINT', type=bool, default=False)
    parser.add_argument('--MODEL_NAME', type=str, default='andiNet_1')
    parser.add_argument('--OUTPUT_ROOT', type=str, default='output')
    parser.add_argument('--DIST_BACKEND', type=str, default='nccl')
    parser.add_argument('--LOOSE', type=list, default=['labels', 'pos'])
    parser.add_argument('--LOOSE_WEIGHT', type=dict, default={'loss_ce': 1, 'loss_pos': 9, 'loss_push': 1})
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')
    parser.add_argument(
        '--output_attn_figure',
        default=False,
        type=bool)

    args = parser.parse_args()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    logger = setup_logger(args.final_output_dir, time_str)
    # # distribution
    # if args.distributed:
    #     # 1) 初始化
    #     torch.distributed.init_process_group(backend=args.DIST_BACKEND)
    #     # 2） 配置每个进程的gpu
    #     local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #
    #     #
    #     # 设置随机参数
    #     seed = args.SEED
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     # 3）使用DistributedSampler
    #     train_dataset = Dataset(args, args.train_data_path)
    #     test_dataset = Dataset(args, args.test_data_path)
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         # shuffle=False,
    #         drop_last=True,
    #         # num_workers=args.WORKERS,
    #         # collate_fn=collate_fn,
    #         # pin_memory=True,
    #         # sampler=train_sampler
    #     )
    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=1,
    #         # shuffle=False,
    #         drop_last=True,
    #         # num_workers=args.WORKERS,
    #         # collate_fn=collate_fn,
    #         # pin_memory=True,
    #         sampler=test_sampler
    #     )
    #
    #     backbone = Backbone(args)
    #     transformer = Transformer()
    #     num_classes = dict(
    #         entity_labels=len(task_ner_labels[args.task]),
    #         rel_labels=len(task_rel_labels[args.task])
    #     )
    #     matcher = HungarianMatcher(cost_class=args.COST_CLASS,
    #                                cost_pos=args.COST_POS)
    #     losses = ['labels', 'pos']
    #     weight_dict = {'loss_ce': 1, 'loss_pos': 9, 'loss_push': 1}
    #     criterion = SetCriterion(matcher=matcher, losses=losses, weight_dict=weight_dict, eos_coef=0.1,
    #                              num_classes=num_classes)
    #     model = MainNet(backbone, transformer, num_classes=num_classes)
    #     # 4) 封装之前要把模型移到对应的gpu
    #     # 将模型移到cuda
    #     if not torch.cuda.is_available():
    #         logger.error('No CUDA found!')
    #         exit(-1)
    #     model.to(device)
    #     criterion.to(device)
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         # 5) 封装
    #         model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                           device_ids=[local_rank],
    #                                                           output_device=local_rank,
    #                                                           find_unused_parameters=True)
    # else:
    #     # 处理设备的问题的参数问题
    #     ngpus_per_node = torch.cuda.device_count()
    #     if 'SLURM_PROCID' in os.environ.keys():
    #         proc_rank = int(os.environ['SLURM_PROCID'])
    #         local_rank = proc_rank % ngpus_per_node
    #         args.world_size = int(os.environ['SLURM_NTASKS'])
    #     else:
    #         proc_rank = 0
    #         local_rank = 0
    #         args.world_size = 1
    #     if args.DEVICE == 'cuda':
    #         torch.cuda.set_device(local_rank)
    #     device = torch.device(args.DEVICE)
    #
    #     # 设置随机参数
    #     seed = args.SEED
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #
    #     train_dataset = Dataset(args, args.train_data_path)
    #     test_dataset = Dataset(args, args.test_data_path)
    #     # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         # shuffle=False,
    #         drop_last=True,
    #         # num_workers=args.WORKERS,
    #         # collate_fn=collate_fn,
    #         # pin_memory=True,
    #         # sampler=train_sampler
    #     )
    #     # 注意在eval的过程中是在一个gpu上运行的所以需要减小batch_size
    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=args.eval_batch_size,
    #         # shuffle=False,
    #         drop_last=True,
    #         # num_workers=args.WORKERS,
    #         # collate_fn=collate_fn,
    #         # pin_memory=True,
    #         # sampler=train_sampler
    #     )
    #     backbone = Backbone(args)
    #     transformer = Transformer()
    #     num_classes = dict(
    #         entity_labels=len(task_ner_labels[args.task]),
    #         rel_labels=len(task_rel_labels[args.task])
    #     )
    #
    #     matcher = HungarianMatcher(cost_class=args.COST_CLASS,
    #                                cost_pos=args.COST_POS)
    #     # get_loss 会遍历losses
    #     # losses = ['labels', 'pos', 'push']
    #     # weight_dict = {'loss_ce': 1, 'loss_pos': 9, 'loss_push': 1}
    #     criterion = SetCriterion(matcher=matcher, losses=args.LOOSE, weight_dict=args.LOOSE_WEIGHT, eos_coef=0.1,
    #                              num_classes=num_classes)
    #     model = MainNet(backbone, transformer, criterion, num_classes=num_classes)
    #     model = torch.nn.DataParallel(model)
    #     # 将模型移到cuda
    #     if not torch.cuda.is_available():
    #         logger.error('No CUDA found!')
    #         exit(-1)
    #     model.to(device)
    # 处理设备的问题的参数问题
    ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_PROCID' in os.environ.keys():
        proc_rank = int(os.environ['SLURM_PROCID'])
        local_rank = proc_rank % ngpus_per_node
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        proc_rank = 0
        local_rank = 0
        args.world_size = 1
    if args.DEVICE == 'cuda':
        torch.cuda.set_device(local_rank)
    device = torch.device(args.DEVICE)

    # 设置随机参数
    seed = args.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = Dataset(args, args.train_data_path)
    test_dataset = Dataset(args, args.test_data_path)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=False,
        drop_last=True,
        # num_workers=args.WORKERS,
        # collate_fn=collate_fn,
        # pin_memory=True,
        # sampler=train_sampler
    )
    # 注意在eval的过程中是在一个gpu上运行的所以需要减小batch_size
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        # shuffle=False,
        drop_last=True,
        # num_workers=args.WORKERS,
        # collate_fn=collate_fn,
        # pin_memory=True,
        # sampler=train_sampler
    )
    backbone = Backbone(args)
    transformer = Transformer()
    num_classes = dict(
        entity_labels=len(task_ner_labels[args.task]),
        rel_labels=len(task_rel_labels[args.task])
    )

    matcher = HungarianMatcher(cost_class=args.COST_CLASS,
                               cost_pos=args.COST_POS)
    # get_loss 会遍历losses
    # losses = ['labels', 'pos', 'push']
    # weight_dict = {'loss_ce': 1, 'loss_pos': 9, 'loss_push': 1}
    criterion = SetCriterion(matcher=matcher, losses=args.LOOSE, weight_dict=args.LOOSE_WEIGHT, eos_coef=0.1,
                             num_classes=num_classes)
    model = MainNet(backbone, transformer, criterion, num_classes=num_classes,
                    output_attn_figure=args.output_attn_figure)

    # 将模型移到cuda
    if not torch.cuda.is_available():
        logger.error('No CUDA found!')
        exit(-1)
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # 优化器 这里主要是为了区别backbone和上层的lr
    param_dicts = [
        {'params': [p for n, p in model_without_ddp.named_parameters()
                    if 'bert' in n]},
        {'params': [p for n, p in model_without_ddp.named_parameters()
                    if 'bert' not in n], 'lr': args.task_learning_rate},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.bert_learning_rate,
                                  weight_decay=args.WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.LR_DROP)
    if args.train_from_checkpoint:
        # 如果要从checkpoint继续训练的话需要先载入相应的参数
        checkpoint = torch.load(args.final_output_dir + '/' + args.checkpoint_name)
        if args.train_from_checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
    else:
        epoch_start = -1
    model = torch.nn.DataParallel(model)
    trainer = BaseTrainer(args, model, criterion, optimizer, lr_scheduler, epoch_start)

    while True:
        trainer.train(train_loader, test_loader)
