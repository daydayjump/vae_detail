#! /usr/bin/env python
# -*- coding: utf-8 -*-
###################################################################
# File Name: /home/daydayjump/test/from_zero_to_success/vae_detail/onehot.py
# Author: daydayjump
# mail: newlifestyle2014@126.com
# Created Time: 2018年06月15日 星期五 11时01分51秒
###################################################################

import torch

def labelonehot(label, n):
    assert label.size(1) == 1
    onehot = torch.zeros(label.size(0),3,n)
    onehot.scatter_(1, label, 1)
    return onehot



