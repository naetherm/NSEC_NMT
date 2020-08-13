# -*- coding: utf-8 -*-
'''
Copyright 2019, University of Freiburg.
Chair of Algorithms and Data Structures.
Markus Näther <naetherm@informatik.uni-freiburg.de>
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SqueezeExcitationLayer(nn.Module):

  def __init__(self, in_features, out_features, reduction=16, bias=True):
    super(SqueezeExcitationLayer, self).__init__()

    #if in_features % reduction != 0:
    #    raise ValueError('n_features must be divisible by reduction (default = 16)')

    self.linear1 = nn.Linear(in_features, in_features // reduction, bias=bias)
    self.nonlin1 = nn.ReLU(inplace=True)
    self.linear2 = nn.Linear(in_features // reduction, out_features, bias=bias)
    self.nonlin2 = nn.Sigmoid()

  def forward(self, x):

    print("x.size(): {}".format(x.size()))
    y = F.avg_pool1d(x, kernel_size=x.size()[2:3])
    #y = y.permute(0, 2, 1)
    y = self.nonlin1(self.linear1(y))
    y = self.linear2(y)
    y = nn.weight_norm(y)
    y = self.nonlin2(y)
    #y = self.linear2(y)
    #y = y.permute(0, 1, 2)
    y = x * y
    return y
