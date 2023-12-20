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


def main_fedmon(model_type,learning_rate, momentum, nesterov ,num_rounds, local_round, num_clients ,batch_size, loss_function,dataset, global_momentum):
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
    server=fedserver.Server(model, learning_rate, momentum, nesterov, device,global_momentum)
    server.global_optimizer=optim.SGD(server.global_model.parameters(), lr=learning_rate)
    server.loss_function(loss_function)
    client_name_list=tools.set_client(num_clients)
    clients=[]
    for i in range(len(client_name_list)):
         clients.append(fedclient.Client(id= client_name_list[i],data=train_loaders[i],local_round=local_round, device=device))
         clients[i].local_model=copy.deepcopy(model)
         server.register(clients[i])

    results=[]
    for i in range(num_rounds):
          for i in range(len(client_name_list)): 
            server.download_model_mon(client_name_list[i])   
            server.current_global_round=i
            server.local_train(client_name_list[i])
          server.aggregate_fedmon()

          server.loss=server.get_loss(server.global_model,test_loader)
          server.acc=server.get_accuracy(server.global_model,test_loader)
          result=server.result()
          results.append(result)
    return results

# T=[10,20,40,80,160]
# N=[2,4,8,16]
# for i in range(5):
#     for j in range(4):
#         result=main_fedmon('cnn',0.01,0,False,math.ceil(1000/T[i]),T[i],N[j],64,'nll_loss','mnist')
#         sheetname='fedmo_'+str(num_clients)+'_'+str(local_round)
#         tools.save2excel_batch('result_N_T.xlsx',sheetname,results)
# global_momentum=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# for i in range(10):
#         results=main_fedmon('cnn',0.01,0,False,25,40,4,64,'nll_loss','mnist',global_momentum[i])
#         sheetname='fedmo_'+str(global_momentum[i])
#         tools.save2excel_batch('result_fedmon_momentum.xlsx',sheetname,results)

