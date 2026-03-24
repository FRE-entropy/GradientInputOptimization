import torch
import numpy as np
from models.bp_network import BPNetwork
from scripts.train import load_model

def predict(model, X):
    """
    使用模型进行预测
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = model(X_tensor)
    return outputs.numpy()

def load_trained_model(layers=[10, 64, 32, 4]):
    """
    加载训练好的模型
    """
    model = BPNetwork(layers=layers)
    model = load_model(model, 'models/bp_network.pth')
    return model

if __name__ == "__main__":
    # 加载训练好的模型
    model = load_trained_model()
    
    # 生成测试样本
    test_sample = np.random.randn(1, 10)
    print(f"测试输入: {test_sample}")
    
    # 进行预测
    prediction = predict(model, test_sample)
    print(f"预测输出: {prediction}")