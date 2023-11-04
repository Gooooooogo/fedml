

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
def main1(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function):
    # Load MNIST dataset
    device=tools.choose_device()
    model, train_dataset, test_dataset= choose_models.select_model(model_type)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_0(train_dataset,len(train_dataset.classes), num_clients )
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device )
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    print(server.loss_func.__class__.__name__=='MSELoss')
    
    for i in range(num_rounds):
          server.download_model('client1')
          server.download_model('client2')
          server.download_model('client3')
          server.download_model('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.fednoavg()
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.fednoavg()
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.fednoavg()
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fednag()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()
def main(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_2(train_dataset,len(train_dataset.classes), num_clients,1500)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          server.download_model('client1')
          server.download_model('client2')
          server.download_model('client3')
          server.download_model('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fednag()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()
def main3(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_3(train_dataset,len(train_dataset.classes), num_clients,5)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          server.download_model('client1')
          server.download_model('client2')
          server.download_model('client3')
          server.download_model('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fednag()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()

def main_fedmon(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_0(train_dataset,len(train_dataset.classes), num_clients)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.global_optimizer=optim.SGD(server.global_model.parameters(), lr=learning_rate)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          server.download_model_mon('client1')   
          server.download_model_mon('client2')
          server.download_model_mon('client3')
          server.download_model_mon('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fedmon()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          result=server.result()
          tools.save2excel('result_all.xlsx','test',result)
def main_fedmon_1(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_1(train_dataset,len(train_dataset.classes), num_clients,0.2)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.global_optimizer=optim.SGD(server.global_model.parameters(), lr=learning_rate)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          server.download_model_mon('client1')   
          server.download_model_mon('client2')
          server.download_model_mon('client3')
          server.download_model_mon('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fedmon()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()

def main_fedmon_2(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_2(train_dataset,len(train_dataset.classes), num_clients,1500)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.global_optimizer=optim.SGD(server.global_model.parameters(), lr=learning_rate)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          server.download_model_mon('client1')   
          server.download_model_mon('client2')
          server.download_model_mon('client3')
          server.download_model_mon('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fedmon()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()


def main_fedmon_3(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset):
    device=tools.choose_device()
    model=choose_models.select_model(model_type)
    train_dataset, test_dataset= choose_models.select_dataset(dataset)
    model.to(device)
    # # Create data loaders for each client
    client_datasets= resample_data.data_distribution_3(train_dataset,len(train_dataset.classes), num_clients, 5)
    train_loaders = []
    for i in range(num_clients):
        train_loader = torch.utils.data.DataLoader(dataset=client_datasets[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #server and client
    model=copy.deepcopy(model)
    model.to(device)
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device)
    server.global_optimizer=optim.SGD(server.global_model.parameters(), lr=learning_rate)
    server.loss_function(loss_function)
    client1= fedclient.Client(id= 'client1',data=train_loaders[0],local_round=local_round, device=device)
    client2= fedclient.Client(id= 'client2',data=train_loaders[1],local_round=local_round, device=device)
    client3= fedclient.Client(id= 'client3',data=train_loaders[2],local_round=local_round, device=device)
    client4= fedclient.Client(id= 'client4',data=train_loaders[3],local_round=local_round, device=device)
    server.register(client1)
    server.register(client2)
    server.register(client3)
    server.register(client4)
    
    for i in range(num_rounds):
          server.download_model_mon('client1')   
          server.download_model_mon('client2')
          server.download_model_mon('client3')
          server.download_model_mon('client4')
          server.current_global_round=i
          server.local_train('client1')
          server.local_train('client2')
          server.local_train('client3')
          server.local_train('client4')
          server.aggregate_fedmon()
          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          server.result()


if __name__ == "__main__":



    #main('linear',0.01,0,False,20,20,4,64,'MSE','mnist')
    # main('log',0.01,0,False,50,20,4,64,'CrossEntropy','mnist')
    #main('cnn',0.01,0,False,25,40,4,64,'nll_loss','mnist')
    # #main('linear',0.01,0.9,True,50,20,4,64,'MSE','mnist')
    # main('log',0.01,0.9,True,50,20,4,64,'CrossEntropy','mnist')
    #main('cnn',0.01,0.9,True,25,40,4,64,'nll_loss','mnist')
    #main3('cnn',0.01,0,False,25,40,4,64,'nll_loss','mnist')
    #main3('cnn',0.01,0.9,True,25,40,4,64,'nll_loss','mnist')

    main_fedmon('linear',0.01,0.9,False,50,20,4,64,'MSE','mnist')
    #main_fedmon('log',0.01,0.9,False,50,20,4,64,'CrossEntropy','mnist')
    #main_fedmon('cnn',0.01,0.9,False,25,40,4,64,'nll_loss','mnist')
    # main_fedmon_1('cnn',0.01,0.9,False,25,40,4,64,'nll_loss','mnist')
    # main_fedmon_2('cnn',0.01,0.9,False,25,40,4,64,'nll_loss','mnist')
    # main_fedmon_3('cnn',0.01,0.9,False,25,40,4,64,'nll_loss','mnist')