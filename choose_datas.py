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


def data_distribution_0(dataset,num_classes,num_clients):
    client_datasets=[]
    data_splits = random_split(dataset, [len(dataset)//num_clients]*num_clients)
    for data_split in data_splits:
        client_datasets.append(data_split)
    return client_datasets
def data_distribution_1(dataset,num_classes,num_clients,a):
    data = dataset.data.numpy()
    labels = dataset.targets.numpy()
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    client_datasets=[]
     # 定义每个客户端的数据比例
    client_ratios=[]
    for i in range(num_clients):
            client_ratios.append([(1 - a) / 9] * i +[a]+ [(1 - a) / 9] * (num_classes -i))
    # 将数据分配给客户端
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]

    for i in range(num_clients):
        # 计算每个类别应该分配给客户端的数量
        class_counts = [int(ratio * len(data) / num_clients) for ratio in client_ratios[i]]

        # 分配数据和标签给客户端
        for class_idx in range(num_classes):
            class_data = data[labels == class_idx]
            class_labels = labels[labels == class_idx]

            # 随机打乱数据
            np.random.shuffle(class_data)
            np.random.shuffle(class_labels)

            # 选择要分配给客户端的数据
            client_data[i].extend(class_data[:class_counts[class_idx]])
            client_labels[i].extend(class_labels[:class_counts[class_idx]])
    for i in range(num_clients):
          # Convert client data and labels to PyTorch tensors
          client_data_tensor = torch.Tensor(client_data[i])
          client_labels_tensor = torch.LongTensor(client_labels[i])  # Assuming labels are integers

          # Create a TensorDataset
          client_datasets.append(TensorDataset(client_data_tensor, client_labels_tensor))
    return client_datasets
def data_distribution_2(dataset,num_classes,num_clients,theta):
    data = dataset.data.numpy()
    labels = dataset.targets.numpy()
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
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
            np.random.shuffle(class_data)
            np.random.shuffle(class_labels)
            client_data[i].extend(class_data[:choice_list[i]])
            client_labels[i].extend(class_data[:choice_list[i]])
    for i in range(num_clients):
          # Convert client data and labels to PyTorch tensors
          client_data_tensor = torch.Tensor(client_data[i])
          client_labels_tensor = torch.LongTensor(client_labels[i])  # Assuming labels are integers
          # Create a TensorDataset
          client_datasets.append(TensorDataset(client_data_tensor, client_labels_tensor))
    return client_datasets
def data_distribution_3(dataset,num_classes,num_clients,ch):
    data = dataset.data.numpy()
    labels = dataset.targets.numpy()
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    client_datasets=[]
    choice_list = [i for i in range(num_classes)]
    for i in range(num_clients):
        for j in range(ch):
            random.shuffle(choice_list)
            class_data = data[labels == choice_list[j]]
            class_labels = labels[labels  == choice_list[j]]
            client_data[i].extend(class_data)
            client_labels[i].extend(class_labels)
    for i in range(num_clients):
          # Convert client data and labels to PyTorch tensors
          client_data_tensor = torch.Tensor(client_data[i])
          client_labels_tensor = torch.LongTensor(client_labels[i])  # Assuming labels are integers

          # Create a TensorDataset
          client_datasets.append(TensorDataset(client_data_tensor, client_labels_tensor))
    return client_datasets
