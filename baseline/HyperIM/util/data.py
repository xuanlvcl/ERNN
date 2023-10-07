import os
import sys
sys.path.append('..') 

import torch as th
import torch.utils.data

import numpy as np
import scipy.sparse

import json
import numpy.matlib

def load_train_data(data_path='./data/sample', train_batch_size=50, word_num=500):
    json_val = None
    with open(os.path.join(data_path, 'train.json')) as fp:
        json_val = json.load(fp)
    X_train = np.zeros((len(json_val), word_num), dtype = np.int64) 
    y_train = np.matlib.zeros((len(json_val),5), dtype=np.int64) #类别为5
    for i in range(len(json_val)):
        label = json_val[i]['label']
        flow = json_val[i]['flow']
        if len(flow) < word_num:
            X_train[i][:len(flow)] = flow
        else:
            X_train[i] = flow[:word_num]
        y_train[i, label] = 1

    # X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    # y_train = scipy.sparse.load_npz(os.path.join(data_path, 'y_train.npz'))

    X_train = th.LongTensor(X_train)
    y_train = th.Tensor(y_train)

    train_dataset = th.utils.data.TensorDataset(X_train, y_train)
    train_data_loader = th.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    return train_data_loader


def load_test_data(data_path='./data/sample', test_batch_size=50, word_num=500):
    json_val = None
    with open(os.path.join(data_path, 'test.json')) as fp:
        json_val = json.load(fp)
    X_test = np.zeros((len(json_val), word_num), dtype = np.int64) 
    y_test = np.matlib.zeros((len(json_val),5), dtype=np.int64) #类别为5
    for i in range(len(json_val)):
        label = json_val[i]['label']
        flow = json_val[i]['flow']
        if len(flow) < word_num:
            X_test[i][:len(flow)] = flow
        else:
            X_test[i] = flow[:word_num]
        y_test[i, label] = 1
        
    # X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    # y_test = scipy.sparse.load_npz(os.path.join(data_path, 'y_test.npz'))

    X_test = th.LongTensor(X_test)
    y_test = th.Tensor(y_test)

    test_dataset = th.utils.data.TensorDataset(X_test, y_test)
    test_data_loader = th.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)
    
    return test_data_loader


def load_data(data_path='./data/sample', train_batch_size=50, test_batch_size=50, word_num=500):
    train_data_loader = load_train_data(data_path, train_batch_size, word_num)
    test_data_loader = load_test_data(data_path, test_batch_size, word_num)

    for X_train_batch, y_train_batch in train_data_loader:
        print('X_train shape', X_train_batch.shape, 'y_train shape', y_train_batch.shape)
        break
    print('train_batch_num', len(train_data_loader))
    for X_test_batch, y_test_batch in test_data_loader:
        print('X_test shape', X_test_batch.shape, 'y_test shape', y_test_batch.shape)
        break
    print('test_batch_num', len(test_data_loader))
    
    return train_data_loader, test_data_loader
