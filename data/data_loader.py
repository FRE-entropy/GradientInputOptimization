import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_data_loaders(X, Y, batch_size=32, test_split=0.2):
    """
    创建训练和测试数据加载器
    params:
        X: 输入特征数据，numpy数组或类似数组结构
        Y: 目标标签数据，numpy数组或类似数组结构
        batch_size: 批次大小，默认为32
        test_split: 测试集比例，默认为0.2（即20%数据用于测试）
    return:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 分割数据
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # 创建数据集
    train_dataset = CustomDataset(X_train, Y_train)
    test_dataset = CustomDataset(X_test, Y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader