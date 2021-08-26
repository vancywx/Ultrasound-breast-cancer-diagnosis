import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from core.evaluate import validate
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from misc import log
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
                    default='densenet_da', help='network_name')

parser.add_argument('--image_height', type=int, default=300,
                    help='image height')
parser.add_argument('--image_width', type=int, default=500,
                    help='image width')
parser.add_argument('--image_mean', type=int, default=73.,
                    help='image mean')
parser.add_argument('--image_std', type=int, default=21.,
                    help='image std')

parser.add_argument('--max_iterations', type=int,
                    default=500000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

snapshot_path = "%s/%s/" %(args.root_modelPath, args.net)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
print('Batch size: %d' % (batch_size))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (args.image_height, args.image_width, 3)
num_classes = 2

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = network_att.DenseNet_DA()
    net = net.cuda()

    Dir = args.root_dataPath
    trainlist, vallist = [], []
    trainlist_B = readfilelist.getFileList_all('Txt_mix/train_B.txt', Dir)
    trainlist_M = readfilelist.getFileList_all('Txt_mix/train_M.txt', Dir)
    vallist_B = readfilelist.getFileList_all('Txt_mix/val_B.txt', Dir)
    vallist_M = readfilelist.getFileList_all('Txt_mix/val_M.txt', Dir)
    trainlist = trainlist_B + trainlist_B + trainlist_M
    vallist = vallist_B + vallist_B + vallist_M
    print ('Train list:', (len(trainlist)))
    print ('Test list:', (len(vallist)))

    dataset_mean_value = args.image_mean
    dataset_std_value = args.image_std

    dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
    dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

    db_train = USBreast(trainlist, patch_size, True,
                        transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(),
                                            transforms.Normalize(
                            mean=dataset_mean,
                            std=dataset_std)]))
    db_test = USBreast(vallist, patch_size, False,
                       transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(
                                               mean=dataset_mean,
                                          std=dataset_std)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
#     optimizer = optim.SGD(net.parameters(), lr=base_lr,
#                           momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    net.train()

    w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'w')

    w_id.write('Initial lr=%f\n' % (base_lr))
    w_id.close()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        net.train()
        time1 = time.time()
        ce_losses = util.AverageMeter()
        ce_accs = util.AverageMeter()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = net(image_batch)

            cross_entropy = torch.nn.CrossEntropyLoss()
            loss_clf = cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss = loss_clf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = util.accuracy(outputs_soft, label_batch)
            ce_losses.update(loss, image_batch.shape[0])
            ce_accs.update(acc, image_batch.shape[0])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 500 == 0:
                logging.info('iteration %d : accumulated mean loss : %f; accumulated mean acc : %f' % (
                    iter_num, ce_losses.avg, ce_accs.avg))
                w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'a')
                w_id.write('Iteration: %d\t train: CE_Loss: %.4f \t  Accuracy: %.4f\n' % (
                    iter_num, ce_losses.avg, ce_accs.avg))
                w_id.close()

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
        steps = int(np.ceil(len(vallist) / (1.0 * args.batch_size)))
        bar = log.ProgressBar(total=steps, width=40)
        print('---------------------------------------------------------------------')
        print('Validation begins...')
        acc, sensitivity, specificity, precision,  F1, auc, kappa = validate(
            args, net, testloader, bar)
        w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'a')
        w_id.write('Epoch: %d\tAccuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f\n' % (
            epoch_num, acc, sensitivity, specificity, precision,  2 * sensitivity * precision / (sensitivity + precision), auc, kappa))
        w_id.close()
        save_mode_path = os.path.join(
            snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(net.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        print('Validation ends...')
        print('---------------------------------------------------------------------')

    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
