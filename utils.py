from collections import OrderedDict
import logging
import logzero
from pathlib import Path
from tensorboardX import SummaryWriter
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target):
    """Computes the accuracy of minibatch"""
    _, predicted = torch.max(output.data, 1)
    print predicted, target
    total = target.size(0)
    correct = (predicted == target).sum().item()
    acc = correct * 1.0 / (total * 1.0)

    return acc


def mae(output, target):
    """Computes the average mae of minibatch"""

    target = target.data.cpu().numpy()
    target = target.reshape((target.shape[0], 1))
    output = output.data.cpu().numpy()
    err = np.abs(output - target)
    total = len(target)
    mae = (np.sum(err) * 1.0) / (total * 1.0)
    return mae


def save_checkpoint(model, epoch, filename, optimizer=None):
    if optimizer is None:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, filename)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)


def load_checkpoint(model, path, optimizer=None):
    resume = torch.load(path)

    if ('module' in list(resume['state_dict'].keys())[0]) \
            and not (isinstance(model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in resume['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(resume['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        return model, optimizer
    else:
        return model


def set_logger(path, loglevel=logging.INFO, tf_board_path=None):
    path_dir = '/'.join(path.split('/')[:-1])
    if not Path(path_dir).exists():
        Path(path_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.formatter(logging.Formatter(
        '[%(asctime)s %(levelname)s] %(message)s'))
    logzero.logfile(path)

    if tf_board_path is not None:
        tb_path_dir = '/'.join(tf_board_path.split('/')[:-1])
        if not Path(tb_path_dir).exists():
            Path(tb_path_dir).mkdir(parents=True)
        writer = SummaryWriter(tf_board_path)

        return writer
