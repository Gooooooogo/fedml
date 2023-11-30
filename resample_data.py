import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
import copy
import argparse
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, random_split ,TensorDataset,Subset
from PIL import Image
class CustomDataset():
    def __init__(self, data, labels, transform=None):      
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(self.data[index].numpy(), mode="L")
        if self.transform:
            img = self.transform(img)
        return img, target
#1.展示最后的分布
#2. 不重复
def data_distribution_0(dataset,num_classes,num_clients):
    client_datasets=[]
    choose_num=len(dataset)//num_clients
    data = copy.deepcopy(dataset.data)
    labels = copy.deepcopy(dataset.targets)
    client_data = []
    client_labels = []
    for i in range(num_clients):
        client_data.append(data[:choose_num])
        client_labels.append(labels[:choose_num])
        data=np.delete(data,slice(0,choose_num),axis=0)
        labels=np.delete(labels,slice(0,choose_num),axis=0)
        print(len(data))
    for i in range(num_clients):
          client_datasets.append(CustomDataset(client_data[i], client_labels[i],dataset.transform))

   
    return   client_datasets    
def data_distribution_0_1(dataset,num_classes,num_clients):
    client_datasets = []
    choose_num=len(dataset)//num_clients
    for i in range(num_clients):
        start_idx = i * choose_num
        end_idx = (i + 1) * choose_num
        client_dataset = Subset(dataset, range(start_idx, end_idx))
        client_datasets.append(client_dataset)
    return   client_datasets  

def data_distribution_1(dataset,num_classes,num_clients,a):
    print('start')
    data = copy.deepcopy(dataset.data)
    labels = copy.deepcopy(dataset.targets)
    client_datasets=[]
     # 定义每个客户端的数据比例
    client_ratios=[]
    for i in range(num_clients):
            client_ratios.append([(1 - a) / 9] * i +[a]+ [(1 - a) / 9] * (num_classes -i))
    # 将数据分配给客户端
    client_data = [torch.empty(0, dtype=torch.uint8) for _ in range(num_clients)]
    client_labels = [torch.empty(0, dtype=torch.uint8) for _ in range(num_clients)]

    for i in range(num_clients):
        # 计算每个类别应该分配给客户端的数量
        class_counts = [int(ratio * len(dataset.data) / num_clients) for ratio in client_ratios[i]]

        # 分配数据和标签给客户端
        for class_idx in range(num_classes):
            class_data = data[labels == class_idx]
            class_labels = labels[labels == class_idx]

            # 选择要分配给客户端的数据
            client_data[i]=torch.cat((client_data[i],class_data[:class_counts[class_idx]]),dim=0)
            client_labels[i]=torch.cat((client_labels[i],(class_labels[:class_counts[class_idx]])),dim=0)
            delete_list=np.where(labels == class_idx)[0][:class_counts[class_idx]]
            data = np.delete(data,delete_list,axis=0)
            labels = np.delete(labels, delete_list,axis=0)
            print(len(data))
    for i in range(num_clients):
        for class_idx in range(num_classes):
                count_ones = sum(1 for item in client_labels[i] if item == class_idx)
                print(count_ones)

    for i in range(num_clients):
          client_datasets.append(CustomDataset(client_data[i], client_labels[i],dataset.transform))
    #         
    #         # 选择要分配给客户端的数据
    #         client_data[i]=torch.cat((client_data[i],class_data[:class_counts[class_idx]]),dim=0)
    #         client_labels[i]=torch.cat((client_labels[i],(class_labels[:class_counts[class_idx]])),dim=0)
    #         delete_list=np.where(labels == class_idx)[0][:class_counts[class_idx]]
    #         data = np.delete(data,delete_list,axis=0)
    #         labels = np.delete(labels, delete_list,axis=0)
    #         print(len(data))
    # for i in range(num_clients):
    #     for class_idx in range(num_classes):
    #             count_ones = sum(1 for item in client_labels[i] if item == class_idx)
    #             print(count_ones)

    # for i in range(num_clients):
    #       client_datasets.append(TensorDataset(client_data[i], client_labels[i]))
    return client_datasets
def data_distribution_2(dataset,num_classes,num_clients,theta):
    data = dataset.data
    labels = dataset.targets
    client_data = [torch.empty(0, dtype=torch.uint8) for _ in range(num_clients)]
    client_labels = [torch.empty(0, dtype=torch.uint8) for _ in range(num_clients)]
    client_datasets=[]
    D = len(data)
    start = (D//num_clients-3*theta)//num_classes
    end = (D//num_clients+3*theta)//num_classes
    step = math.ceil((end - start) / num_clients)
    choice_list = [start + i * step for i in range(4)]
    for i in range(num_clients):
        for j in range(num_classes):
            class_data = data[labels == j]
            class_labels = labels[labels  == j]

            client_data[i]=torch.cat((client_data[i],class_data[:choice_list[i]]),dim=0)
            client_labels[i]=torch.cat((client_labels[i],class_labels[:choice_list[i]]),dim=0)
            delete_list=np.where(labels==j)[0][:choice_list[i]]
            data=np.delete(data,delete_list,axis=0)
            labels=np.delete(labels,delete_list,axis=0)
            print(len(data))
    for i in range(num_clients):
        for class_idx in range(num_classes):
                count_ones = sum(1 for item in client_labels[i] if item == class_idx)
                print(count_ones)
    for i in range(num_clients):
          client_datasets.append(CustomDataset(client_data[i], client_labels[i],dataset.transform))
    return client_datasets

def data_distribution_3(dataset,num_classes,num_clients,ch):
    data = copy.deepcopy(dataset.data)
    labels = copy.deepcopy(dataset.targets)
    client_data = [torch.empty(0, dtype=torch.uint8) for _ in range(num_clients)]
    client_labels = [torch.empty(0, dtype=torch.uint8) for _ in range(num_clients)]
    client_datasets=[]
    choice_list = [i for i in range(num_clients+1)]
    client_class_ch= []
    assign_num= len(data)//(ch*num_clients)

    # for i in range(num_clients):
    #     client_class_ch.append([(j+i*ch)%num_classes for j in choice_list])
    if ch==5:
        client_class_ch=[[0,1,2,3,4],
                        [3,4,5,6,7],
                        [5,6,7,8,9],
                        [0,1,2,8,9]]
    if ch==3:
         client_class_ch=[[0,1,2],
                        [3,4,5],
                        [4,9,6],
                        [6,7,8]]
    if ch==6:
         client_class_ch=[[0,1,2,3,4,5],
                        [3,4,5,6,7,8],
                        [5,6,7,8,9,1],
                        [1,3,5,7,9,2]]     
    if ch==9:
         client_class_ch=[[0,1,2,3,4,5,6,7,8],
                        [3,4,5,6,7,8,9,1,2],
                        [4,5,6,7,8,9,1,2,0],
                        [0,1,3,4,5,6,2,8,9]]  
    for i in range(num_clients):
        for j in range(ch):
            class_data = data[labels == client_class_ch[i][j]]
            class_labels = labels[labels == client_class_ch[i][j]]
            client_data[i]=torch.cat((client_data[i],class_data[:assign_num]),dim=0)
            client_labels[i]=torch.cat((client_labels[i],class_labels[:assign_num]),dim=0)
            delete_list=np.where(labels==client_class_ch[i][j])[0][:assign_num]
            data=np.delete(data,delete_list,axis=0)
            labels=np.delete(labels,delete_list,axis=0)
            print(len(data))
        client_datasets.append(CustomDataset(client_data[i], client_labels[i],dataset.transform))
    for i in range(num_clients):
        for class_idx in range(num_classes):
                count_ones = sum(1 for item in client_labels[i] if item == class_idx)
                print(count_ones)        
    return client_datasets



if __name__ == "__main__":
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    d0=data_distribution_3(train_dataset,10,4,10)
    d0_loader=torch.utils.data.DataLoader(dataset=d0[0], batch_size=64, shuffle=False)
    process=d0_loader.dataset[0][0]
    original=train_dataset[1][0]
    print(process)
    print(torch.equal(original,process))
