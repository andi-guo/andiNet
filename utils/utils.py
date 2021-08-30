import logging
import os
import torch
import logging
from collections import defaultdict, deque


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logger(final_output_dir, time_str, phase='train'):
    log_file = f'{phase}_{time_str}.log'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def get_dataset():
    pass


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    logging.info(f'save model to {output_dir}')
    if is_best:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))


def write_dict_to_json(mydict, f_path):
    import json
    import numpy
    class DateEnconding(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                                numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                                numpy.uint16, numpy.uint32, numpy.uint64)):
                return int(obj)
            elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                                  numpy.float64)):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)):  # add this line
                return obj.tolist()  # add this line
            return json.JSONEncoder.default(self, obj)

    with open(f_path, 'w') as f:
        json.dump(mydict, f, cls=DateEnconding)
        print("write down det dict to %s!" % (f_path))
# class SmoothedValue(object):
#     """Track a series of values and provide access to smoothed values over a
#     window or the global series average.
#     """
#
#     def __init__(self, window_size=20, fmt=None):
#         if fmt is None:
#             fmt = "{median:.4f} ({global_avg:.4f})"
#         self.deque = deque(maxlen=window_size)
#         self.total = 0.0
#         self.count = 0
#         self.fmt = fmt
#
#     def update(self, value, n=1):
#         self.deque.append(value)
#         self.count += n
#         self.total += value * n
#
#     def synchronize_between_processes(self):
#         """
#         Warning: does not synchronize the deque!
#         """
#         if not is_dist_avail_and_initialized():
#             return
#         t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
#         dist.barrier()
#         dist.all_reduce(t)
#         t = t.tolist()
#         self.count = int(t[0])
#         self.total = t[1]
#
#     @property
#     def median(self):
#         d = torch.tensor(list(self.deque))
#         return d.median().item()
#
#     @property
#     def avg(self):
#         d = torch.tensor(list(self.deque), dtype=torch.float32)
#         return d.mean().item()
#
#     @property
#     def global_avg(self):
#         return self.total / self.count
#
#     @property
#     def max(self):
#         return max(self.deque)
#
#     @property
#     def value(self):
#         return self.deque[-1]
#
#     def __str__(self):
#         return self.fmt.format(
#             median=self.median,
#             avg=self.avg,
#             global_avg=self.global_avg,
#             max=self.max,
#             value=self.value)
#
#
# class MetricLogger(object):
#     def __init__(self, delimiter="\t"):
#         self.meters = defaultdict(SmoothedValue)
#         self.delimiter = delimiter
#
#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.meters[k].update(v)
#
#     def __getattr__(self, attr):
#         if attr in self.meters:
#             return self.meters[attr]
#         if attr in self.__dict__:
#             return self.__dict__[attr]
#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, attr))
#
#     def __str__(self):
#         loss_str = []
#         for name, meter in self.meters.items():
#             loss_str.append(
#                 "{}: {}".format(name, str(meter))
#             )
#         return self.delimiter.join(loss_str)
#
#     def synchronize_between_processes(self):
#         for meter in self.meters.values():
#             meter.synchronize_between_processes()
#
#     def add_meter(self, name, meter):
#         self.meters[name] = meter
#
#     def log_every(self, iterable, print_freq, header=None):
#         i = 0
#         if not header:
#             header = ''
#         start_time = time.time()
#         end = time.time()
#         iter_time = SmoothedValue(fmt='{avg:.4f}')
#         data_time = SmoothedValue(fmt='{avg:.4f}')
#         space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
#         if torch.cuda.is_available():
#             log_msg = self.delimiter.join([
#                 header,
#                 '[{0' + space_fmt + '}/{1}]',
#                 'eta: {eta}',
#                 '{meters}',
#                 'time: {time}',
#                 'data: {data}',
#                 'max mem: {memory:.0f}'
#             ])
#         else:
#             log_msg = self.delimiter.join([
#                 header,
#                 '[{0' + space_fmt + '}/{1}]',
#                 'eta: {eta}',
#                 '{meters}',
#                 'time: {time}',
#                 'data: {data}'
#             ])
#         MB = 1024.0 * 1024.0
#         for obj in iterable:
#             data_time.update(time.time() - end)
#             yield obj
#             iter_time.update(time.time() - end)
#             if i % print_freq == 0 or i == len(iterable) - 1:
#                 eta_seconds = iter_time.global_avg * (len(iterable) - i)
#                 eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#                 if torch.cuda.is_available():
#                     if is_main_process():
#                         print(log_msg.format(
#                             i, len(iterable), eta=eta_string,
#                             meters=str(self),
#                             time=str(iter_time), data=str(data_time),
#                             memory=torch.cuda.max_memory_allocated() / MB))
#                 else:
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time)))
#             i += 1
#             end = time.time()
#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('{} Total time: {} ({:.4f} s / it)'.format(
#             header, total_time_str, total_time / len(iterable)))
