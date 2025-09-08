import torch 
import torch.nn as nn
class FFNetwork(nn.Module):
    def __init__(self, w, d_model):
        super(FFNetwork, self).__init__()
        self.w = w
        self.linear1 = nn.Linear(d_model, w, bias=True)
        self.linear2 = nn.Linear(w, d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x