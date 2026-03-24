import torch
import torch.nn as nn

class BPNetwork(nn.Module):
    def __init__(self, layers=[10, 64, 32, 4]):
        super(BPNetwork, self).__init__()
        
        self.layers = layers
        
        # 构建网络层
        network_layers = []
        current_dim = layers[0]
        
        # 添加隐藏层和输出层
        for dim in layers[1:]:
            network_layers.append(nn.Linear(current_dim, dim))
            if dim != layers[-1]:
                network_layers.append(nn.ReLU())
            current_dim = dim
        
        self.network = nn.Sequential(*network_layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_config(self):
        return {
            'layers': self.layers,
        }