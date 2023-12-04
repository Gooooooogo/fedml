# -*- coding: utf-8 -*-
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
import copy
import argparse
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, random_split ,TensorDataset
import resample_data
import choose_models 
import tools
import fedserver
import fedclient
def main_fedfastslowmon(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_0_1(train_dataset,len(train_dataset.classes), num_clients)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.global_optimizer=optim.SGD(server.global_model.parameters(), lr=learning_rate, momentum=0.5, nesterov=True)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    client1.local_model=copy.deepcopy(model)
    client2.local_model=copy.deepcopy(model)
    client3.local_model=copy.deepcopy(model)
    client4.local_model=copy.deepcopy(model)
    client1.local_optimizer= optim.SGD(client1.local_model.parameters(), lr=learning_rate, momentum=0.5, nesterov=True)
    client2.local_optimizer= optim.SGD(client2.local_model.parameters(), lr=learning_rate, momentum=0.5, nesterov=True)
    client3.local_optimizer= optim.SGD(client3.local_model.parameters(), lr=learning_rate, momentum=0.5, nesterov=True)
    client4.local_optimizer= optim.SGD(client4.local_model.parameters(), lr=learning_rate, momentum=0.5, nesterov=True)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          print(i)
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fastslowmon()
          server.download_model_faseslowmon('client1')   
          server.download_model_faseslowmon('client2')
          server.download_model_faseslowmon('client3')
          server.download_model_faseslowmon('client4')
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          print(server.acc)
          result=server.result()
          if model_type == 'linear':
               tools.save2excel('result_all.xlsx','mnist_linear_fastslowmon',result)
          if model_type == 'log':
               tools.save2excel('result_all.xlsx','mnist_log_fastslowmon',result)
          if model_type == 'cnn' and dataset== 'mnist':
               tools.save2excel('result_all.xlsx','mnist_cnn_fastslowmon',result)
          if model_type == 'cnn' and dataset== 'cifar10':
               tools.save2excel('result_all.xlsx','cifar10_cnn_fastslowmon',result)
          if model_type == 'vgg16' and dataset== 'cifar10':
               tools.save2excel('result_all.xlsx','cifar10_vgg_fastslowmon',result)

# tools.cleanexcel('result_all.xlsx','mnist_linear_fastslowmon')
# tools.cleanexcel('result_all.xlsx','mnist_log_fastslowmon')
# tools.cleanexcel('result_all.xlsx','mnist_cnn_fastslowmon')
tools.cleanexcel('result_all.xlsx','cifar10_vgg_fastslowmon')
# main_fedfastslowmon('linear',0.01,0.5,True,50,20,4,64,'MSE','mnist')
# main_fedfastslowmon('log',0.01,0.5,True,50,20,4,64,'CrossEntropy','mnist')
# main_fedfastslowmon('cnn',0.01,0.5,True,25,40,4,64,'nll_loss','mnist')
#main_fedfastslowmon('cnn',0.01,0.5,True,25,40,4,64,'nll_loss','cifar10')
main_fedfastslowmon('vgg16',0.01,0.5,True,25,40,4,64,'CrossEntropy','cifar10')
