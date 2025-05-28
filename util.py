#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by
@Author: Chang Zhao
@Contact: 2023201666@aust.edu.cn
@Time: 2025/5/28 4:33 PM
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, channels_list):
        super(MultiScaleFeatureFusion, self).__init__()
        self.channels_list = channels_list
        self.num_scales = len(channels_list)
        self.fuse_conv = nn.Conv1d(sum(channels_list), max(channels_list), kernel_size=1)
        self.bn = nn.BatchNorm1d(max(channels_list))

    def forward(self, features):
        batch_size, num_points = features[0].size(0), features[0].size(2)
        fused_features = []
        for i in range(self.num_scales):
            if features[i].size(1) != max(self.channels_list):
                fused_features.append(features[i])
            else:
                fused_features.append(features[i])
        fused_features = torch.cat(fused_features, dim=1)
        fused_features = F.relu(self.bn(self.fuse_conv(fused_features)))
        return fused_features
