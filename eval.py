# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from utils.utils import setup_logger
import time
import torch
from src.models.myNet_entity_only import MainNet, SetCriterion
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
from sklearn.metrics import classification_report
from data.const import task_ner_labels
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default distributed training
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='if use distribute train')
    parser.add_argument('--PRETRAINED_MODEL_NAME', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--final_output_dir', type=str, default='./output')
    parser.add_argument('--CHECK_POINT_NAME', type=str, default='model_best.pth')
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--eval_data_path', type=str, default='./data/data_mini.json')
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--WORKERS', type=int, default=4)
    parser.add_argument('--ENTITY_NUM', type=int, default=25)
    parser.add_argument('--RELATION_NUM', type=int, default=30)
    parser.add_argument('--MAX_LEN', type=int, default=512)
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
    parser.add_argument('--PRINT_FREQ', type=int, default=5)
    parser.add_argument('--VAL_WHEN_TRAIN', type=bool, default=True)
    parser.add_argument('--MODEL_NAME', type=str, default='andiNet')
    parser.add_argument('--OUTPUT_ROOT', type=str, default='output')
    parser.add_argument('--DIST_BACKEND', type=str, default='nccl')
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training, machine level')

    args = parser.parse_args()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    logger = setup_logger(args.final_output_dir, time_str)
    # distribution

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

    eval_dataset = Dataset(args, args.eval_data_path)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        # shuffle=False,
        drop_last=True,
        # num_workers=args.WORKERS,
        # collate_fn=collate_fn,
        # pin_memory=True,
        # sampler=train_sampler
    )
    # 注意在eval的过程中是在一个gpu上运行的所以需要减小batch_size
    backbone = Backbone(args)
    transformer = Transformer()
    num_classes = dict(
        entity_labels=len(task_ner_labels[args.task]),
        rel_labels=len(task_rel_labels[args.task])
    )

    matcher = HungarianMatcher(cost_class=args.COST_CLASS,
                               cost_pos=args.COST_POS)
    losses = ['labels', 'pos']
    weight_dict = {'loss_ce': 1, 'loss_pos': 9}
    criterion = SetCriterion(matcher=matcher, losses=losses, weight_dict=weight_dict, eos_coef=0.1,
                             num_classes=num_classes)
    model = MainNet(backbone, transformer, criterion, num_classes=num_classes)
    # 加载模型的参数
    model.load_state_dict(torch.load(args.final_output_dir + '/' + args.CHECK_POINT_NAME))
    model = torch.nn.DataParallel(model)
    # 将模型移到cuda
    if not torch.cuda.is_available():
        logger.error('No CUDA found!')
        exit(-1)
    model.to(device)

    # 开始进行evaluation
    model.eval()
    results = []
    targets = []
    count = 0
    for data in tqdm(eval_loader):
        t_label, o_label = model.module.evaluate(data)
        targets += t_label
        results += o_label
        count += 1
    # save the result

    f1 = classification_report(targets, results, labels=list(range(len(task_ner_labels['drug']) + 1)), target_names=task_ner_labels['drug'],
                               output_dict=True,
                               zero_division=0)
    print(f1)
