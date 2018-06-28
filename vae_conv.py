#! /usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# File Name: /home/daydayjump/test/from_zero_to_success/vae_detail//vae_conv.py
# Author: daydayjump
# mail: newlifestyle2014@126.com
# Created Time: 2018年06月08日 星期五 14时48分50秒
###################################################################

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import imageloader
from imageloader import MyDataset
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE in conv')
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
parser.add_argument('--hidden-size', type=int, default=20, help='size of z')

args = parser.parse_args()

print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

###############   DATASET   #################

data_path = '/home/daydayjump/test/from_zero_to_success/dataset'
data_transforms = transforms.Compose([
     transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = MyDataset(root=data_path, extensions='.png',
          loader=imageloader.default_loader, transform=data_transforms)

loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                           shuffle=True, **kwargs)

###############  MODEL   ##################


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        n = 16
        # 160 x 160
        self.conv1 = nn.Sequential(nn.Conv2d(3, n, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n),
                      nn.LeakyReLU(0.2, inplace=False))
        # 80 x 80
        self.conv2 = nn.Sequential(nn.Conv2d(n, n*2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*2),
                      nn.LeakyReLU(0.2, inplace=False))
        # 40 x 40
        self.conv3 = nn.Sequential(nn.Conv2d(n*2, n*3, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*3),
                      nn.LeakyReLU(0.2))
        # 20 x 20
        self.conv4 = nn.Sequential(nn.Conv2d(n*3, n*4, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*4),
                      nn.LeakyReLU(0.2))
        # 10 x 10
        self.conv5 = nn.Sequential(nn.Conv2d(n*4, n, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(n),
                       nn.LeakyReLU(0.2))
        # 5 x 5

        self.fc11 = nn.Linear(n * 5 * 5, args.hidden_size)
        self.fc12 = nn.Linear(n * 5 * 5, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, n * 5 * 5)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(n, n*4, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*4),
                      nn.ReLU())
        # 10 x 10
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(n*4, n*3, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*3),
                      nn.ReLU())
        # 20 x 20
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(n*3, n*2, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n*2),
                      nn.ReLU())
        # 40 x 40
        self.deconv4= nn.Sequential(nn.ConvTranspose2d(n*2, n, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(n),
                      nn.ReLU())
        # 80 x 80
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(n,3,kernel_size=4,stride=2,padding=1),
                      nn.Sigmoid())
        # 160 x 160

    def encoder(self, x):
        # input: noise output: mu and sigma
        # input: args.batch_size x 3 x 160 x160
        out = self.conv1(x)
        # args.batch_size x n x 80 x 80
        out = self.conv2(out)
        # args.batch_size x n*2 x 40 x 40
        out = self.conv3(out)
        # args.batch_size x n*3 x 20 x 20
        out = self.conv4(out)
        # args.batch_size x n*4 x 10 x 10
        out = self.conv5(out)
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
        # args.batch_size x (n x 5 x 5)
        out = self.deconv1(out.view(x.size(0), n, 5, 5))
        # args.batch_size x n*4 x 10 x 10
        out = self.deconv2(out)
        # args.batch_size x n*3 x 20 x 20
        out = self.deconv3(out)
        # args.batch_size x n*2 x 40 x 40
        out = self.deconv4(out)
        # args.batch_size x n x 80 x 80
        out = self.deconv5(out)
        # args.batch_size x 3 x 160 x160
        return out

    def forward(self, x):
        mu, logvar = self.encoder(x)
        out = self.reparameterize(mu, logvar)
        return self.decoder(out), mu, logvar
    
model = VAE().to(device)

##############  optimizer & loss function ################

optimizer = optim.Adam(model.parameters(),lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x,
                                 x,size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

##############  train & test ###############

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx , (data,_) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len (data), len(loader.dataset),
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
                       filename='/home/daydayjump/test/from_zero_to_success/vae_detail/results/fake_'+
                       str(epoch)+'.png')

if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        sample(epoch)

    
