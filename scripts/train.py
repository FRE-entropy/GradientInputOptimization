import torch
import torch.nn as nn
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bp_network import BPNetwork
from data.data_loader import get_data_loaders

def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    """
    训练模型
    """
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam 优化器
    
    # 训练循环
    for epoch in range(epochs):
        model.train()  # 切换到训练模式
        train_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()   # 清空梯度
            loss.backward()         # 反向传播更新梯度
            optimizer.step()        # 更新参数
            
            train_loss += loss.item() * X_batch.size(0)
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        
        # 评估模型
        model.eval()  # 切换到评估模式
        test_loss = 0.0
        
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                test_loss += loss.item() * X_batch.size(0)
        
        # 计算平均测试损失
        test_loss = test_loss / len(test_loader.dataset)
        
        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return model

def save_model(model, path):
    """
    保存模型
    """
    # 使用模型的 save 方法保存
    model.save(path)
    print(f"模型保存到 {path}")

def load_model(model, path):
    """
    加载模型
    """
    # 使用模型的 load 方法加载
    return BPNetwork.load(path)

if __name__ == "__main__":
    # 加载数据
    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')
    
    # 为每个输入特征指定预处理方法
    # 例如：对前5个特征使用minmax归一化，后5个特征使用standard标准化
    preprocessing_methods = ['minmax'] * 5 + ['standard'] * 5
    
    # 创建模型
    model = BPNetwork(layers=[10, 64, 32, 4], preprocessing_methods=preprocessing_methods)
    print(f"模型配置: {model.get_config()}")
    
    # 拟合预处理器参数
    model.fit_preprocessor(X)
    
    # 创建数据加载器
    train_loader, test_loader = get_data_loaders(X, Y)
    
    # 训练模型
    model = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)
    
    # 保存模型
    save_model(model, 'models/bp_network.pth')