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
import Resample
import choose_models 
import tools
class Server():
    def __init__(self, object, learning_rate, momentum, nesterov, device):
        self.clients={}
        self.global_model= object
        self.global_model.to(device)
        self.global_optimizer= None
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.nesterov=nesterov
        self.device=device
        self.current_global_round=0
        self.acc=0
        self.loss_func= None
        self.loss=0
        self.v= self.global_model.state_dict()
    def register(self,client):
          self.clients[client.id]= client
          client.registered= True
    def unregister(self,client):
          if client.id in self.clients:
              del self.clients[client.id]
              client.registered= False
    def fednoavg(self):
        idx=0
        #shuffled_list = random.sample(list(self.clients.values()), len(list(self.clients.values())))
        shuffled_list=list(self.clients.values())
        shuffled_list.insert(0, shuffled_list.pop())
        for key,value in self.clients.items():
            self.clients[key].local_model=copy.deepcopy(shuffled_list[idx].local_model)
            self.clients[key].local_optimizer=copy.deepcopy(shuffled_list[idx].local_optimizer)
            idx+=1
    def aggregate_fedmon(self):
        new_w_local = [[] for _ in range(len(self.clients))]
        # get all local_model's params
        for clt,idx in zip(self.clients.values(),range(len(self.clients))):
               for param in clt.local_model.state_dict().values():
                    new_w_local[idx].append(param)
        average_list = tools.average(new_w_local)
        new_w_average={}
        
        idx=0
        for key in self.global_model.state_dict().keys():
            new_w_average[key]= average_list[idx].to(self.device)
            idx+=1
        w=self.global_model.state_dict()
        v=self.v
        #get v
        v_new={}
        w_new={}
        for key,value in v.items():
            v_new[key]=w[key]-1* (w[key]-new_w_average[key])            
            w_new[key]=v_new[key] + self.momentum * (v_new[key]-v[key] )
        self.v.update(v_new)
        self.global_model.load_state_dict(w_new)
        self.global_optimizer=optim.SGD(self.global_model.parameters(), lr=self.learning_rate)
    def aggregate_fednag(self):
        learning_rate=self.learning_rate
        momentum=self.momentum
        nesterov=self.nesterov
        averaged_weights = {}
        # for key in self.global_model.state_dict().keys():
        #     averaged_weights[key] = sum([i.local_model.state_dict()[key] for i in self.clients.values()]) / len(self.clients)
        model_param_list = [[] for _ in range(len(self.clients))]
        # get all locl_model's params
        for clt,idx in zip(self.clients.values(),range(len(self.clients))):
               for param in clt.local_model.state_dict().values():
                    model_param_list[idx].append(param)
        average_list = tools.average(model_param_list)
        #set global_model's params
        idx=0
        for key in self.global_model.state_dict().keys():
            averaged_weights[key]= average_list[idx].to(self.device)
            idx+=1
        self.global_model.load_state_dict(averaged_weights)



        self.global_optimizer=optim.SGD(self.global_model.parameters(), lr=learning_rate, momentum=momentum, nesterov= nesterov)
        # momentum_buffer_list : [[]*len(self.clients)]
        if self.momentum != 0:
            momentum_buffer_list = [[] for _ in range(len(self.clients))]
            #get local_momentum_buffer
            for clt , idx in zip(self.clients.values(), range(len(self.clients))):
                for group in clt.local_optimizer.param_groups:
                    for p in group['params']:
                        param_state = clt.local_optimizer.state[p]
                        momentum_buffer_list[idx].append(torch.clone(param_state['momentum_buffer']).detach())
            #average momentum_buffer
            column_means = tools.average(momentum_buffer_list)
            # set global_momentum_buffer
            idx=0
            for group in self.global_optimizer.param_groups:
                    for p in group['params']:
                        if momentum != 0:
                            self.global_optimizer.state[p]['momentum_buffer']= torch.clone(column_means[idx]).detach()
                            idx+=1
    def download_model(self, client_id):
        client=self.clients[client_id]
        client.local_model = copy.deepcopy(self.global_model)
        client.local_optimizer = optim.SGD(client.local_model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov= self.nesterov)
        client.local_optimizer.load_state_dict(self.global_optimizer.state_dict())
        client.local_model.to(self.device)
    def download_model_mon(self, client_id):
        client=self.clients[client_id]
        client.local_model = copy.deepcopy(self.global_model)
        client.local_optimizer = optim.SGD(client.local_model.parameters(), lr=self.learning_rate)
        client.local_optimizer.load_state_dict(self.global_optimizer.state_dict())
        client.local_model.to(self.device)

    def local_train(self, client_id):
            client=self.clients[client_id]
            client.local_model.to(self.device)
            criterion = self.loss_func
            ## note
            ## for epoch in local_round:
            ##      train 1 time then iterate
            ##      then break
            ##      if iterate all then initialize data_iter and trained_idx
            loss=0
            loss_sum=0
            for epoch in range(client.local_round):

                for (data, labels) in client.data_iter:
                    client.trained_idx+=1
                    client.trained_nums+=1
                    client.local_optimizer.zero_grad()
                    data, labels = data.to(self.device), labels.to(self.device)
                    output = client.local_model(data)
                    if self.loss_func.__class__.__name__== 'MSELoss':
                        labels_one_hot = torch.zeros_like(output)
                        labels_one_hot.scatter_(1, labels.view(-1, 1), 1.0)
                        labels=labels_one_hot.float()
                    loss = criterion(output, labels)
                    loss.to(self.device)
                    loss.backward()
                    client.local_optimizer.step()
                    #loss_sum+= loss.item()
                    if client.trained_idx == len(client.data):
                        client.data_iter=iter(client.data)
                        client.trained_idx=0
                    break
            client.loss =  loss.item()
            #client.loss=loss_sum/client.local_round



    def get_loss(self,model,test_loader):
          loss_sum=0
          model.eval()
          criterion = self.loss_func
          with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)  # 将数据移动到 GPU 上
                #labels= labels.float()
                outputs = model(images)
                if self.loss_func.__class__.__name__== 'MSELoss':
                    labels_one_hot = torch.zeros_like(outputs)
                    labels_one_hot.scatter_(1, labels.view(-1, 1), 1.0)
                    labels= labels_one_hot.float()
                batch_loss = criterion(outputs, labels)
                loss_sum += batch_loss.item()
          return loss_sum/len(test_loader)

    def get_accuracy(self,model,test_loader):
      model.eval()
      criterion = self.loss_func
      accuracy=0
      total=0
      correct=0
      with torch.no_grad():
          for data in test_loader:
              images, labels = data
              images, labels = images.to(self.device), labels.to(self.device)  # 将数据移动到 GPU 上
              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
          accuracy = 100 * correct / total
      return accuracy


    def result(self):
        current_global_round=self.current_global_round
        client_list=list(self.clients.values())
        current_iteration= sum([i.trained_nums for i in client_list])/len(client_list)
        average_training_loss = sum([i.loss/len(client_list) for i in client_list])
        test_loss= self.loss
        accuracy= self.acc

        data = {
        "current_global_round": current_global_round,
        "current_iteration": current_iteration,
        "average_training_loss": average_training_loss,
        "test_loss": test_loss,
        "accuracy": accuracy
        }

        csv_filename = "result.csv"
        # with open(csv_filename, "a", newline="") as csvfile:
        #     fieldnames = data.keys()
        #     csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        #     csv_writer.writerow(data)

        return data

    def loss_function(self,loss_func):
        if loss_func=='MSE':
            self.loss_func = torch.nn.MSELoss()

        elif loss_func=='CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif loss_func=='nll_loss':
            self.loss_func = torch.nn.NLLLoss()
