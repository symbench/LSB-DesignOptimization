# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:11:45 2021

@author: HPP
"""
import glob
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

device = torch.device("cpu")
import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import OneHotEncoder


class utilities:
    def __init__(self):
        self.lr = 0

    def set_lr(self, lr):
        self.lr = lr

    def set_lr_auto(self):
        self.lr = np.random.choice(np.logspace(-3, 0, base=10))

    def get_optimizer(self, model):
        optimizer_class = optim.Adam
        # print('***learning rate is:',self.lr)
        return optimizer_class(model.parameters(), lr=0.001)

    def get_lossfunc(self, net_type):
        if net_type == "S":
            return nn.L1Loss()
        elif net_type == "T":
            return nn.BCELoss()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
        torch.nn.init.constant_(m.bias.data, 0)






def scale_data(data, ranges):
    minimum = []
    total_range = []
    for i in range(data.shape[1]):
        minimum.append(ranges[2 * i])
        total_range.append((ranges[2 * i + 1] - ranges[2 * i]))
    # print('min is:',minimum,'Range is:',total_range,'data b/f scaling is:',data)
    minimum = np.array(minimum).reshape(1, -1)
    total_range = np.array(total_range).reshape(1, -1)
    data = np.divide((data - minimum), total_range)
    # print('min is:',minimum,'Range is:',total_range,'data a/f scaling is:',data)
    return data


def rescale_data(data, ranges, mask):
    # print('data b/f rescaling:',data)
    for i in range(len(mask)):
        # print('i is:',i,'mask is:',mask[i])

        if mask[i] == "real":
            data[:, i] = (data[:, i] * (ranges[2 * i + 1] - ranges[2 * i])) + ranges[
                2 * i
            ]
        elif mask[i] == "int":
            data[:, i] = (data[:, i] * (ranges[2 * i + 1] - ranges[2 * i])) + ranges[
                2 * i
            ]
            data[:, i] = np.array(data[:, i], dtype=np.int16)
    # print('data after rescaling:',data)
    return data


# on a given input prepare data for training
def data_preperation(data, ranges):
    for i in range(int(len(ranges)/2)):
       data[:, i] = (data[:, i] - ranges[2 * i]) / (ranges[2 * i + 1] - ranges[2 * i])
    return data



def label_data(data, stest_pred):
    # create label for data(if predicted vale is >/< 10% of error then it labels it '1' or else it is '0')
    ones = np.ones(stest_pred.shape[0])
    zeros = np.zeros(stest_pred.shape[0])
    # print('test shape:',test_data[:,-1].shape,'zeros shape:',zeros.shape,'ones shape:',ones.shape,'stest shape',stest_pred.flatten().shape)
    result = np.where(
        np.absolute((data[:, -1] - stest_pred.flatten()))
        > (0.05 * np.absolute(data[:, -1])),
        ones,
        zeros,
    )
    data[:, -1] = result
    return data




def data_split_size(data, size):
    # on a given dataset return the splitted data=> train_data(based on size),validate_data(leftover)
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    alist = a_list[0:size]
    train_data = data[alist]
    d = np.arange(data.shape[0])
    leftover = np.delete(d, alist)
    validate_data = data[leftover]
    return train_data, validate_data


def data_split(data, proportion=0.2):
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    alist = a_list[0 : int(data.shape[0] * (1 - proportion))]
    train_data = data[alist]
    d = np.arange(data.shape[0])
    leftover = np.delete(d, alist)
    validate_data = data[leftover]
    return train_data, validate_data

 

def create_datafiles(data, test_fraction=0.1):
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    alist = a_list[0 : int(data.shape[0] * (1 - test_fraction))]
    train_data = data[alist]
    d = np.arange(data.shape[0])
    leftover = np.delete(d, alist)
    test_data = data[leftover]
    # print('train_data to create file:',train_data,'test data in create file:',test_data)
    np.savetxt("./data/train_data.txt", train_data, delimiter=" ")
    np.savetxt("./data/test_data.txt", test_data, delimiter=" ")
    return 0


def create_files(data, file_name):
    name_file = "./data/" + file_name + ".txt"
    np.savetxt(name_file, data, delimiter=" ")
    return 0


class SimDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        x_tmp = dataset
        y_tmp = dataset
        # print('X_tmp is:',x_tmp,'Y_tmp is:',y_tmp)

        self.x = torch.tensor(x_tmp, dtype=torch.float32).to(device)
        self.y = torch.tensor(y_tmp, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)  # required

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x[idx, :]
        pol = self.y[idx]
        sample = {preds, pol}
        return preds, pol



def gen_test_data(n,dim,ranges):
    data= np.random.random((n,dim))
    for i in range(int(len(ranges)/2)):
       data[:, i] = (data[:, i] * (ranges[2 * i + 1] - ranges[2 * i]))+ranges[2 * i]
    data = data.astype(np.float32)
    return data

if __name__ == "__main__":
    data=gen_test_data(10,2,[-5,0,5,10])
    print(data) 
