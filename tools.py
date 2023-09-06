import torch
def average(list):
    transposed = zip(* list)
    averages = [sum(column) / len(column) for column in transposed]
    return averages

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device