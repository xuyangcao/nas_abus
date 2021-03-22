import os
import cv2
import sys
import tqdm
import shutil
import random
import logging
import argparse
import numpy as np
import setproctitle
import matplotlib.pyplot as plt
from skimage.color import label2rgb
plt.switch_backend('agg')
from ast import literal_eval

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from dataloaders import make_data_loader
from utils.utils import save_checkpoint, confusion, get_dice
from utils.loss import DiceLoss, dice_loss
from utils.lr_scheduler import LR_Scheduler
from models.denseuxnet import UXNet

def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()

    # general config
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--ngpu', default=1, type=str)
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--start_epoch', default=1, type=int)

    # dataset config
    parser.add_argument('--dataset', type=str, default='abus')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)

    # optimizer config
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'])
    parser.add_argument('--lr', default=1e-3, type=float) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # network config
    parser.add_argument('--arch', default='uxnet', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet', 'sdnet', 'uxnet', 'd2unet'))
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--growth_rate', default=32, type=int)
    parser.add_argument('--num_init_features', default=64, type=int)
    parser.add_argument('--bn_size', default=4, type=int)
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--block_config', default='[6, 6, 6, 6]', type=str)
    parser.add_argument('--threshold', default=None, type=float)

    # save config
    parser.add_argument('--log_dir', default='./log/train')
    parser.add_argument('--save', default='./work/train/uxnet')

    args = parser.parse_args()
    return args


def main():
    #############
    # init args #
    #############
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    # creat save path
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # logger
    logging.basicConfig(filename=args.save+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('--- init parameters ---')

    # writer
    idx = args.save.rfind('/')
    log_dir = args.log_dir + args.save[idx:]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # set title of the current process
    setproctitle.setproctitle('...')

    # random
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')

    if args.arch == 'dense121': 
        model = DenseUnet(arch='121', pretrained=True, num_classes=2)
    elif args.arch == 'dense161': 
        model = DenseUnet(arch='161', pretrained=True, num_classes=2)
    elif args.arch == 'dense201': 
        model = DenseUnet(arch='201', pretrained=True, num_classes=2)
    elif args.arch == 'resunet': 
        model = ResUNet(in_ch=3, num_classes=2, relu=False)
    elif args.arch == 'unet': 
        model = UNet(n_channels=3, n_classes=2)
    elif args.arch == 'd2unet': 
        model = D2UNet(in_channels=3, num_classes=2)
    elif args.arch == 'uxnet':
        model_params = {
                'in_channels': args.in_channels,
                'num_classes': args.num_classes,
                'growth_rate': args.growth_rate,
                'num_init_features': args.num_init_features,
                'bn_size': args.bn_size,
                'drop_rate': args.drop_rate,
                'block_config': literal_eval(args.block_config),
                'threshold': args.threshold,
                }
        model = UXNet(**model_params)
    else:
        raise(NotImplementedError('model {} not implement'.format(args.arch))) 
    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))


    model = model.cuda()

    ################
    # prepare data #
    ################
    logging.info('--- loading dataset ---')
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_loader, _, val_loader = make_data_loader(args, **kwargs)
    
    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    lr_scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epoch, len(train_loader))

    loss_fn = {}
    loss_fn['dice_loss'] = dice_loss
    loss_fn['l1_loss'] = nn.L1Loss()


    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    nTrain = len(train_loader.dataset)
    for epoch in range(args.start_epoch, args.epoch + 1):
        train(args, epoch, model, train_loader, optimizer, loss_fn, writer, lr_scheduler)
        dice = val(args, epoch, model, val_loader, optimizer, loss_fn, writer)
        is_best = False
        if dice > best_pre:
            is_best = True
            best_pre = dice
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_pre': best_pre},
                          is_best, 
                          args.save, 
                          args.arch)
    writer.close()


def train(args, epoch, model, train_loader, optimizer, loss_fn, writer, lr_scheduler):
    model.train()
    nProcessed = 0
    batch_size = args.ngpu * args.batch_size
    nTrain = len(train_loader.dataset)
    loss_list = []

    for batch_idx, sample in enumerate(train_loader):
        # read data
        image, target = sample['image'], sample['target']
        image, target = Variable(image.cuda()), Variable(target.cuda(), requires_grad=False)

        # forward
        seg_pred = model(image)
        seg_pred = F.softmax(seg_pred, dim=1)
        #print(seg_pred.shape)
        #print(target.shape)
        loss = loss_fn['dice_loss'](seg_pred[:, 1, ...], target==1)

        # backward
        lr = lr_scheduler(optimizer, batch_idx, epoch, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        writer.add_scalar('lr', lr, epoch)

        # visualization
        nProcessed += len(image)
        partialEpoch = epoch + batch_idx / len(train_loader)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

    writer.add_scalar('train_loss',float(np.mean(loss_list)), epoch)


def val(args, epoch, model, val_loader, optimizer, loss_fn, writer):
    model.eval()
    dice_list = []

    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(val_loader)):
            image, target = sample['image'], sample['target']
            image, target = image.cuda(), target.cuda()

            # forward
            seg_pred = model(image)
            seg_pred = F.softmax(seg_pred, dim=1)
            seg_pred = seg_pred.max(1)[1]
            dice = get_dice(seg_pred.cpu().numpy(), target.cpu().numpy())
            dice_list.append(dice)

        writer.add_scalar('val_dice', float(np.mean(dice_list)), epoch)
        return np.mean(dice_list)


if __name__ == '__main__':
    main()
