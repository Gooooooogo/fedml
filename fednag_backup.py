# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
import copy
import argparse



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

    def local_train_fednag(self, client_id):
        client=self.clients[client_id]
        client.local_model = copy.deepcopy(self.global_model)
        client.local_optimizer = optim.SGD(client.local_model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov= self.nesterov)
        client.local_optimizer.load_state_dict(self.global_optimizer.state_dict())
        client.local_model.to(self.device)
        for batch_idx, (data, target) in enumerate(client.data): #note: batch_idx start from 0
            data.to(self.device)
            target.to(self.device)
            if batch_idx >= client.trained_idx and  batch_idx< client.trained_idx+client.local_round:
                client.local_optimizer.zero_grad()
                output = client.local_model(data.to(self.device))
                loss = nn.CrossEntropyLoss()(output.to(self.device), target.to(self.device))
                loss.to(self.device)
                loss.backward()
                client.local_optimizer.step()
            if batch_idx == client.trained_idx+client.local_round: 
                client.trained_idx = batch_idx
                break

    def local_train_fedmon(self, client_id):
        client=self.clients[client_id]
        client.local_model = copy.deepcopy(self.global_model)
        client.local_optimizer = optim.SGD(client.local_model.parameters(), lr=self.learning_rate)
        client.local_optimizer.load_state_dict(self.global_optimizer.state_dict())
        client.local_model.to(self.device)
        for batch_idx, (data, target) in enumerate(client.data): #note: batch_idx start from 0
            data.to(self.device)
            target.to(self.device)
            if batch_idx >= client.trained_idx and  batch_idx< client.trained_idx+client.local_round:
                client.local_optimizer.zero_grad()
                output = client.local_model(data.to(self.device))
                loss = nn.CrossEntropyLoss()(output.to(self.device), target.to(self.device))
                loss.to(self.device)
                loss.backward()
                client.local_optimizer.step()
            if batch_idx == client.trained_idx+client.local_round: 
                client.trained_idx = batch_idx
                break

    def accuracy(self, model,test_loader):
        model.eval()
        correct = 0 
        total = 0
        # with torch.no_grad():
        #     for data, target in test_loader:
        #         output = model(data)
        #         _, predicted = torch.max(output.data, 1)
        #         total += target.size(0)
        #         correct += (predicted == target).sum().item()
        # print(f" Accuracy: {100 * correct / total:.2f}%")
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)  # 将数据移动到 GPU 上
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(accuracy)

class Client:
    def __init__(self, id ,data, local_round, device):
        self.id=id
        self.data=data
        self.local_model = torch.tensor(0)
        self.local_model.to(device)
        self.local_optimizer =  0
        self.registered = False
        self.local_round=local_round
        self.device=device
        self.trained_idx=0




def average(list):
    transposed = zip(* list)
    averages = [sum(column) / len(column) for column in transposed]
    return averages




def select_model(model_type):
    model = None
    train_dataset = None
    test_dataset = None
    if model_type =='linear':
        model=Net()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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
    return model, train_dataset, test_dataset

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device







def main(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size):
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
        train_loader = torch.utils.data.DataLoader(dataset=subset_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=Server(model, learning_rate, momentum, nesterov, device)
    client1= Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    for i in range(num_rounds):
          server.local_train_fednag('client1')
          server.local_train_fednag('client2')
          server.local_train_fednag('client3')
          server.local_train_fednag('client4')
          server.fednag()
          server.accuracy(server.global_model,test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_type", choices=['linear', 'VGG16'], required=True, help="Specify the model type")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--momentum", type=float, help="Momentum")
    parser.add_argument("--nesterov", action="store_true", help="Enable Nesterov acceleration")
    parser.add_argument("--num_rounds", type=int,  help="Number of training rounds")
    parser.add_argument("--local_round", type=int, help="Number of local training rounds")
    parser.add_argument("--num_clients", type=int, help="Number of clients")
    parser.add_argument("--batch_size", type=int,  help="Batch size")

    args = parser.parse_args()
    main(args.model_type,args.learning_rate, args.momentum, args.nesterov ,args.num_rounds, args.local_round, args.num_clients,args.batch_size)



