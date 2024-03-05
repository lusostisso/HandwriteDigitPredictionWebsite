import torch
import torch.nn as nn
from torchvision import transforms

def carregar_modelo():
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(torch.load('pesos_do_modelo.pth'))
    
    model.eval()
    
    return model