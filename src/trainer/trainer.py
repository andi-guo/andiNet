from utils.utils import AverageMeter
import logging
import sys
import time
from torch import autograd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.utils import save_checkpoint, write_dict_to_json
import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report
from data.const import ENTITY_PADDING, RELATION_PADDING, task_ner_labels


class BaseTrainer(object):
    def __init__(self, args, model, criterion, optimizer, lr_scheduler, log_dir='output', rank=0, max_norm=0,
                 performance_indicator='f1'):
        self.model = model
        self.args = args
        self.epoch = 0
        self.max_epoch = args.MAX_EPOCH
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if self.optimizer is not None and rank == 0:
            self.writer = SummaryWriter(log_dir, comment=f'_rank{rank}')
            logging.info(f"max epochs = {self.max_epoch} ")
        self.max_norm = max_norm
        self.best_performance = 0.0
        self.is_best = False
        self.PI = performance_indicator
        self.model_name = self.args.MODEL_NAME
        self.log_dir = log_dir
        self.rank = rank
        self.label_index = list(range(len(task_ner_labels['drug']) + 1))
        task_ner_labels['drug'].append('pos_error')
        self.label_name = task_ner_labels['drug']
        # self.criterion = criterion

    def _forward(self, data):
        loss_dict = self.model(data)
        return loss_dict

    def train(self, train_loader, eval_loader):
        losses_batch = AverageMeter()
        start_time = time.time()
        self.model.train()
        self.criterion.train()
        if self.epoch > self.max_epoch:
            logging.info("Optimization is done !")
            sys.exit(0)
        for data in train_loader:
            # get loss
            loss_dict = self._forward(data)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            self.optimizer.zero_grad()
            # 将各个卡的结果取平均反向传递
            losses.mean().backward()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            losses_batch.update(losses)
        logging.info(f'Now: loss is {losses_batch.avg}')
        # save checkpoint
        if self.rank == 0 and self.epoch >= 0 and self.epoch % self.args.SAVE_INTERVAL == 0:
            # evaluation
            if self.args.VAL_WHEN_TRAIN:
                self.model.eval()
                performance = self.evaluate(eval_loader)
                if performance > self.best_performance:
                    self.is_best = True
                    self.best_performance = performance
                else:
                    self.is_best = False
                logging.info(f'Now: best {self.PI} is {self.best_performance}')
            else:
                performance = -1

            # save checkpoint
            try:
                state_dict = self.model.module.state_dict()  # remove prefix of multi GPUs
            except AttributeError:
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if self.args.SAVE_EVERY_CHECKPOINT:
                    filename = f"{self.model_name}_epoch{self.epoch:03d}_checkpoint.pth"
                else:
                    filename = "checkpoint.pth"
                save_checkpoint(
                    {
                        'epoch': self.epoch,
                        'model': self.model_name,
                        f'performance/{self.PI}': performance,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                    },
                    self.is_best,
                    self.log_dir,
                    filename=f'{self.args.OUTPUT_ROOT}_{filename}'
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        self.epoch += 1

    def evaluate(self, eval_loader):
        self.model.eval()
        results = []
        targets = []
        count = 0
        for data in eval_loader:
            t_label, o_label = self.model.module.evaluate(data)
            targets += t_label
            results += o_label
            count += 1
        # save the result

        f1 = classification_report(targets, results, self.label_index, self.label_name, output_dict=True,
                                   zero_division=0)

        result_path = f'{self.args.OUTPUT_ROOT}/pred.json'
        write_dict_to_json(f1, result_path)
        # eval

        return f1['weighted avg']['f1-score']
