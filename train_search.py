import os 
import sys
import shutil
import tqdm
import numpy as np
import logging
import argparse
import setproctitle
from skimage import measure
from skimage.color import label2rgb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import json
from ast import literal_eval

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from dataloaders import make_data_loader
from architect import Architect
#from models.uxnet import UXNet
from models.model_search import UXNet 
#from models.denseuxnet import UXNet

from utils.lr_scheduler import LR_Scheduler
from utils.loss import DiceLoss, dice_loss


parser = argparse.ArgumentParser("auto unet")
# general config
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--val_frequency', type=int, default=1)

# dataset
parser.add_argument('--dataset', type=str, default='abus3d')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--num_workers', type=int, default=0)

# save
parser.add_argument('--save', type=str, default='./work/search/uxnet3d', help='experiment name')
parser.add_argument('--log_dir', type=str, default='./log/search')

# network
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--growth_rate', default=16, type=int)
parser.add_argument('--num_init_features', default=32, type=int)
parser.add_argument('--bn_size', default=4, type=int)
parser.add_argument('--drop_rate', default=0.3, type=float)
parser.add_argument('--block_config', default='[6, 6, 6, 6]', type=str)
#parser.add_argument('--block_config', default='[6, 12, 12, 6]', type=str)
#parser.add_argument('--block_config', default='[6, 12, 24, 36]', type=str)

# optimizer
parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'])
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# architect search config 
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--start_nas', default=20, type=int)

args = parser.parse_args()


class Trainer(object):
    def __init__(self, ):
        self.args = args

        # logger
        logging.basicConfig(filename=args.save+"/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        # writer
        idx = args.save.rfind('/')
        log_dir = args.log_dir + args.save[idx:]
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir)

        # criterion
        #self.criterion = DiceLoss() 
        self.criterion = dice_loss 

        # dataloader
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        # network & architect
        in_channels = 3 if args.dataset=='isic' else 1
        model_params = {
                'in_channels': in_channels,
                'num_classes': args.num_classes,
                'criterion': self.criterion,
                'growth_rate': args.growth_rate,
                'num_init_features': args.num_init_features,
                'bn_size': args.bn_size,
                'drop_rate': args.drop_rate,
                'block_config': literal_eval(args.block_config) 
                }
        self.model = UXNet(**model_params)
        n_params = sum([p.data.nelement() for p in self.model.parameters()])
        self.logger.info('--- total parameters = {} ---'.format(n_params))
        self.logger.info(self.model.config)
        self.model = self.model.cuda()
        self.architect = Architect(self.model, args)

        # optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.learning_rate, args.epochs, len(self.train_loader))

        self.best_pred = 0.0

    def training(self, epoch):
        train_loss = []
        self.model.train()

        self.logger.info('epoch: {} genotype: {}'.format(epoch, self.model.genotype()))
        self.logger.info('epoch: {} alphas_network: {}'.format(epoch, F.softmax(self.model.alphas_network, dim=-1).cpu().detach().numpy()))

        for i, sample in enumerate(tqdm.tqdm(self.train_loader)):
            if i % 100 == 0:
                print('alphas_network: {}'.format(F.softmax(self.model.alphas_network, dim=-1).cpu().detach().numpy()))
            image, target = sample['image'], sample['target']
            image, target = image.cuda(), target.cuda()
            search = next(iter(self.val_loader))
            image_search, target_search = search['image'], search['target'] 
            image_search, target_search = image_search.cuda(), target_search.cuda()

            lr = self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            output = F.softmax(output, dim=1)
            torch.cuda.empty_cache()
            loss = self.criterion(output[:, 1, ...], target==1)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())

            if epoch > args.start_epoch:
                self.architect.step(image, target, image_search, target_search, lr, self.optimizer, unrolled=args.unrolled)

        self.writer.add_scalar('train_loss_search', np.mean(train_loss), epoch)
        self.writer.add_scalar('lr_network', lr, epoch)

    def validation(self, epoch):
        self.model.eval()
        val_dice = []
        #for i, sample in enumerate(tqdm.tqdm(self.val_loader)):
        for i, sample in enumerate(tqdm.tqdm(self.test_loader)):
            image, target = sample['image'], sample['target']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                output = F.softmax(output, dim=1)
                dice = DiceLoss.dice_coeficient(output.max(1)[1], target)
                val_dice.append(dice.item())

        mean_dice = np.mean(val_dice) 
        self.writer.add_scalar('val_dice_search', mean_dice, epoch)


def main():
    # creat save path
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # set title of the current process
    setproctitle.setproctitle('...')

    # random
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # train and val
    trainer = Trainer()
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if epoch % args.val_frequency == 0:
            trainer.validation(epoch)
    trainer.writer.close()

    trainer.logger.info('epoch: {} genotype: {}'.format(epoch, trainer.model.genotype()))
    trainer.logger.info('epoch: {} alphas_network: {}'.format(epoch, F.softmax(trainer.model.alphas_network, dim=-1).cpu().detach().numpy()))

if __name__ == "__main__":
    main()
