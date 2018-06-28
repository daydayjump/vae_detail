#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from os import path

base_dir = path.dirname(path.relpath(__file__))

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
# 先执行not，再执行and
# 当两个都是true时，代表可用cuda,等号右侧的返回值就是true或者fale
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 为cpu设置种子用于生成随机数，并且结果是确定的。
torch.manual_seed(args.seed)

# 如果args.cuda为true，则使用cuda，否则使用cpu
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            # 乘0.5，是因为logvar= ln(var*var)
            std = torch.exp(0.5*logvar)
            # define epslion
            eps = torch.randn_like(std)
            # mul()don't change the value of origin tensor，returns a new
            # resulting tensor.
            # add_() change the value of tensor.
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        # view()中-1代表由x本身的形状，以及view（）的另一个参数推算出
        # 即x本身是28*28，则-1为28*28/784=1
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# kl divergence + reconstruction(mean,variance)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    # 设置为训练模式
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # 清除所有参数的梯度，归零操作。
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        # 反向传播求导，类似于执行
        loss.backward()
        # 记录损失值
        train_loss += loss.item()
        # 更新参数
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            '''
            len(train_loader.dataset)是数据集的样本个数
            len(train_loader)是数据集按batch_size大小分组之后的组数
            打印的信息是处理的数据个数占总数据的百分比
            '''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data))
            )

    print ('===> Epoch: {} Average loss : {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    '''
    torch.no_grad() 属于上下文管理器(context-manager),需要使用with来调用
    作用是禁止梯度计算，即使requires_grad=True,也不会计算梯度
    可以在test使用，本身不需要更新梯度，省去不必要的计算。
    '''
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # 这三个变量与model的forward的返回值对应
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
		# n = 8, data.size(0)=128
                n = min(data.size(0), 8)
		# [:n]代表选取了前8个样本
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1,
                                                         28, 28)[:n]])
                save_image(comparison.cpu(),
                           '/home/daydayjump/test/from_zero_to_success/vae_detail/results/reconstruction_'+str(epoch) +
                           '.png',nrow=n)

    test_loss /= len(test_loader.dataset)
    print('===> Test set loss : {:.4f}' .format(test_loss))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            filename = path.join(base_dir,'results/sample_'+str(epoch)+'.png')
            print (filename)
	    #因为一共是64个样本，默认nrow=8，为8列，所以图片是8*8的形状
            save_image(sample.view(64, 1, 28, 28),
                       filename=filename)
