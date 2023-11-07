import torch

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
        # fastslowmon
        self.x=torch.tensor(0)
        self.y=torch.tensor(0)