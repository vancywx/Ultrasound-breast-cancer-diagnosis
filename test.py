import os
import argparse
import logging
import time
import random
import numpy as np
from core.evaluate import test
from misc import log
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from nets import network_att
from dataloaders.medicalData import USBreast
from misc import readfilelist
import utils.util as util

# '/home/xiwang/Mount/Data3/'
parser = argparse.ArgumentParser()
parser.add_argument('--root_dataPath', type=str,
                    default='/research/dept7/xiwang/', help='root path of dataset')
parser.add_argument('--root_modelPath', type=str,
                    default='model', help='root path of saved model')
parser.add_argument('--net', type=str,
                    default='densenet_da2', help='network_name')
parser.add_argument('--checkpoint', type=int,  default=20000,
                    help='which model to test')
parser.add_argument('--image_height', type=int, default=300,
                    help='image height')
parser.add_argument('--image_width', type=int, default=500,
                    help='image width')
parser.add_argument('--image_mean', type=int, default=73.,
                    help='image mean')
parser.add_argument('--image_std', type=int, default=21.,
                    help='image std')

parser.add_argument('--batch_size', type=int, default=40,
                    help='batch_size per gpu')

parser.add_argument('--result-root', type=str, default='Deploys',
                    help='where to save results')

parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

model_name = 'iter_%d.pth' % (args.checkpoint)
snapshot_path = '%s/%s/' % (args.root_modelPath, args.net)
model_path = '%s/%s' % (snapshot_path, model_name)
if not os.path.exists(model_path):
    print('%s dose not exist.. Please check it again..Exit!' % (model_path))
    exit(0)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
print('Batch size: %d' % (batch_size))

patch_size = (args.image_height, args.image_width, 3)
num_classes = 2

if __name__ == "__main__":
    # make logger file
    net = network_att.DenseNet_DA()
    net = net.cuda()
    print('Load weight from %s...' % (model_path))
    net.load_state_dict(torch.load(model_path))
    print('Finishing loading weight from %s...' % (model_path))

    Dir = args.root_dataPath
    vallist = []
    vallist_B = readfilelist.getFileList_all('Txt_mix/test_B.txt', Dir)
    vallist_M = readfilelist.getFileList_all('Txt_mix/test_M.txt', Dir)
    vallist = vallist_B + vallist_M
    print ('Number of test set is:', (len(vallist)))

    dataset_mean_value = args.image_mean
    dataset_std_value = args.image_std

    dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
    dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

    db_test = USBreast(vallist, patch_size, False,
                       transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(
                                               mean=dataset_mean,
                                          std=dataset_std)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    steps = int(np.ceil(len(vallist) / (1.0 * args.batch_size)))
    bar = log.ProgressBar(total=steps, width=40)
    print('---------------------------------------------------------------------')
    print('Testing begins...')
    test(args, net, testloader, bar, vallist)
    print('Tesing ends...')
