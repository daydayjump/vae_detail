#! /usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# File Name: /home/daydayjump/test/from_zero_to_success/vae_detail/pre_vgg.py
# Author: daydayjump
# mail: newlifestyle2014@126.com
# Created Time: 2018年06月22日 星期五 12时20分23秒
###################################################################

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import imageloader
from imageloader import MyDataset
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE using VGG-16_bn')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for traning(default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train(default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--hidden-size', type=int, default=200, help='size of z')

args = parser.parse_args()

print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

###################  Dataset ################################
data_path = '/home/daydayjump/test/from_zero_to_success/dataset'
data_transforms = transforms.Compose([
     #transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = MyDataset(root=data_path, extensions='.png',
          loader=imageloader.default_loader, transform=data_transforms)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                           shuffle=True, **kwargs)

#####################  model #################################
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        n = 16
        self.vgg_16_bn = models.vgg16_bn(pretrained=True)
        self.vgg_16_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )


        self.fc11 = nn.Linear(4096, args.hidden_size)
        self.fc12 = nn.Linear(4096, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, n * 10 * 10)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(n, n*3, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*3),
                      nn.ReLU())
        # 28 x 28
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(n*3, n*2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*2),
                      nn.ReLU())
        # 56 x 56
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(n*2, n, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n),
                      nn.ReLU())
        # 112 x 112
        self.deconv4= nn.Sequential(nn.ConvTranspose2d(n, 3, kernel_size=4, stride=2, padding=1),
                      nn.Sigmoid())
        # 224 x 224


    def encoder(self, x):
        # input: noise output: mu and sigma
        # input: args.batch_size x 3 x 224 x 224
        out = self.vgg_16_bn(x)
        return self.fc11(out.view(out.size(0),-1)),self.fc12(out.view(out.size(0),-1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self,x):
        n = 16
        # input: args.batch_size x args.hidden_size
        out = self.fc2(x)
        # args.batch_size x (n x 14 x 14)
        out = self.deconv1(out.view(x.size(0), n, 14, 14))
        # args.batch_size x n*4 x 28 x 28
        out = self.deconv2(out)
        # args.batch_size x n*3 x 56 x 56
        out = self.deconv3(out)
        # args.batch_size x n*2 x 112 x 112
        out = self.deconv4(out)
        # args.batch_size x n x 224 x 224
        return out

    def forward(self, x):
        mu, logvar = self.encoder(x)
        out = self.reparameterize(mu, logvar)
        return self.decoder(out), mu, logvar


model = VAE().to(device)

##############  optimizer & loss function ################

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x,
                                 x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


##############  train & test ###############

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100 * batch_idx / len(loader),
                       loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader.dataset)))


def sample(epoch):
    model.eval()
    with torch.no_grad():
        eps = torch.randn(args.batch_size, args.hidden_size).to(device)
        if (args.cuda):
            fake = model.decoder(eps).cuda()
            save_image(fake.view(args.batch_size, 3, 160, 160),
                       filename='/home/daydayjump/test/from_zero_to_success/vae_detail/results/fake_' +
                                str(epoch) + '.png')


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        sample(epoch)
