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


class Net1(nn.Module):
      def __init__(self):
          super(Net1, self).__init__()
          self.fc= nn.Linear(784, 1)
      def forward(self, x):
          x = x.view(-1, 28*28)
          x = self.fc(x)
          return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(784, 1)  # input and output is 1 dimension

    def forward(self, x):
        x = x.view(-1, 28*28)
        output = self.linear(x)
        output=output.squeeze()
        return output

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class Server():
    def __init__(self, object, learning_rate, momentum, nesterov, device):
        self.clients={}
        self.global_model= object
        self.global_model.to(device)
        self.global_optimizer=optim.SGD(self.global_model.parameters(), lr=learning_rate, momentum=momentum, nesterov= nesterov)
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.nesterov=nesterov
        self.device=device
        self.current_global_round=0
        self.acc=0
        self.loss_func= None
        self.loss=0
    def register(self,client):
          self.clients[client.id]= client
          client.registered= True
    def unregister(self,client):
          if client.id in self.clients:
              del self.clients[client.id]
              client.registered= False


    def fednag(self):
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
        average_list = average(model_param_list)
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
            column_means = average(momentum_buffer_list)
            # set global_momentum_buffer
            idx=0
            for group in self.global_optimizer.param_groups:
                    for p in group['params']:
                        if momentum != 0:
                            self.global_optimizer.state[p]['momentum_buffer']= torch.clone(column_means[idx]).detach()
                            idx+=1
# train
    # def local_train_old(self, client_id):
    #     client=self.clients[client_id]
    #     client.local_model = copy.deepcopy(self.global_model)
    #     client.local_optimizer = optim.SGD(client.local_model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov= 'True')
    #     client.local_optimizer.load_state_dict(self.global_optimizer.state_dict())
    #     client.local_model.to(self.device)
    #     for batch_idx, (data, target) in enumerate(client.data): #note: batch_idx start from 0
    #         data.to(self.device)
    #         target.to(self.device)
    #         if batch_idx >= client.trained_idx and  batch_idx< client.trained_idx+client.local_round:
    #             client.local_optimizer.zero_grad()
    #             output = client.local_model(data.to(self.device))
    #             loss = nn.CrossEntropyLoss()(output.to(self.device), target.to(self.device))
    #             loss.to(self.device)
    #             loss.backward()
    #             client.local_optimizer.step()
    #         if batch_idx == client.trained_idx+client.local_round:
    #             client.trained_idx =batch_idx
    #             break

    def local_train(self, client_id):

            client=self.clients[client_id]
            client.local_model = copy.deepcopy(self.global_model)
            client.local_optimizer = optim.SGD(client.local_model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov= self.nesterov)
            client.local_optimizer.load_state_dict(self.global_optimizer.state_dict())
            client.local_model.to(self.device)
            criterion = self.loss_func
            ## note
            ## for epoch in local_round:
            ##      train 1 time then iterate
            ##      then break
            ##      if iterate all then initialize data_iter and trained_idx
            loss=0
            for epoch in range(client.local_round):

                for (data, target) in client.data_iter:
                    client.trained_idx+=1
                    client.trained_nums+=1
                    client.local_optimizer.zero_grad()
                    data, target = data.to(self.device), target.to(self.device)
                    #target=target.float()
                    output = client.local_model(data)
                    loss = criterion(output, target)
                    loss.to(self.device)
                    loss.backward()
                    client.local_optimizer.step()
                    if client.trained_idx == len(client.data):
                        client.data_iter=iter(client.data)
                        client.trained_idx=0
                    break
            client.loss=loss.item()



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
                batch_loss = criterion(outputs, labels)
                loss_sum += batch_loss.item()
          return loss_sum

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
        current_iteration= sum([i.trained_nums for i in client_list])
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
        with open(csv_filename, "a", newline="") as csvfile:
            fieldnames = data.keys()
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            #csv_writer.writeheader()
            csv_writer.writerow(data)

        #return current_global_round, current_iteration, average_training_loss, test_loss, accuracy

    def loss_function(self,loss_func):
        if loss_func=='MSE':
            self.loss_func = torch.nn.MSELoss()
        elif loss_func=='CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif loss_func=='nll_loss':
            self.loss_func = torch.nn.NLLLoss()






class Client:
    def __init__(self, id ,data, local_round, device):
        self.id=id
        self.data=data
        self.data_iter=iter(data)
        self.local_model = torch.tensor(0)
        self.local_model.to(device)
        self.local_optimizer =  0
        self.registered = False
        self.local_round=local_round
        self.device=device
        self.trained_idx=0
        self.trained_nums=0
        self.loss=0



def average(list):
    transposed = zip(* list)
    averages = [sum(column) / len(column) for column in transposed]
    return averages




def select_model(model_type):
    model = None
    train_dataset = None
    test_dataset = None
    if model_type =='linear':
        model=LinearRegression()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        return model, train_dataset, test_dataset


    elif model_type == 'VGG16':
        model=models.vgg16()
        model.classifier[6] = nn.Linear(4096, 10)
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        return model, train_dataset, test_dataset

    elif model_type == 'log':
        model=LogisticRegression()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        return model, train_dataset, test_dataset

    elif model_type == 'cnn':
          model=cnn()
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])
          train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
          test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
          return model, train_dataset, test_dataset

    return model, train_dataset, test_dataset


def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def main(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function):
    # Load MNIST dataset
    device=choose_device()
    model, train_dataset, test_dataset= select_model(model_type)
    model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    data_per_client = len(train_dataset) // num_clients

    # # Create data loaders for each client
    train_loaders = []

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        subset_dataset = torch.utils.data.Subset(train_dataset, list(range(start_idx,end_idx)))
        train_loader = torch.utils.data.DataLoader(dataset=subset_dataset, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=Server(model, learning_rate, momentum, nesterov, device )
    server.loss_function(loss_function)
    client1= Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    for i in range(num_rounds):
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.fednag()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()


if __name__ == "__main__":
    #main('VGG16',0.01,0.5,True,25,40,4,64)
    #main('linear',0.01,0,False,25,40,4,64,'MSE')
    #main('linear',0.01,0.05,True,25,40,4,64,'MSE')
    main('log',0.01,0,False,25,40,4,64,'CrossEntropy')
    main('log',0.01,0.5,True,25,40,4,64,'CrossEntropy')
    main('cnn',0.01,0,False,25,40,4,64,'nll_loss')
    main('cnn',0.01,0.5,True,25,40,4,64,'nll_loss')