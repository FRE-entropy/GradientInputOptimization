import torch
import torch.nn as nn
from data.preprocessing import Preprocessor

class BPNetwork(nn.Module):
    def __init__(self, layers=[10, 64, 32, 4], preprocessing_methods=None):
        super(BPNetwork, self).__init__()
        
        self.layers = layers
        self.preprocessing_methods = preprocessing_methods
        
        # 初始化预处理器
        if preprocessing_methods:
            self.preprocessor = Preprocessor(preprocessing_methods)
        else:
            self.preprocessor = None
        
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
        # 应用预处理
        if self.preprocessor:
            # 确保输入是numpy数组
            if isinstance(x, torch.Tensor):
                # 使用torch操作实现预处理，保持梯度信息
                x_transformed = x.clone()
                for i in range(self.preprocessor.input_dim):
                    method = self.preprocessor.methods[i]
                    if method == 'minmax':
                        min_val = self.preprocessor.params[i]['min']
                        max_val = self.preprocessor.params[i]['max']
                        if max_val > min_val:
                            x_transformed[:, i] = (x_transformed[:, i] - min_val) / (max_val - min_val)
                    elif method == 'standard':
                        mean_val = self.preprocessor.params[i]['mean']
                        std_val = self.preprocessor.params[i]['std']
                        if std_val > 0:
                            x_transformed[:, i] = (x_transformed[:, i] - mean_val) / std_val
                x = x_transformed
            else:
                x_transformed = self.preprocessor.transform(x)
                x = torch.tensor(x_transformed, dtype=torch.float32)
        return self.network(x)
    
    def fit_preprocessor(self, X):
        """
        拟合预处理器参数
        params:
            X: 输入数据，numpy数组
        """
        if self.preprocessor:
            self.preprocessor.fit(X)
    
    def get_config(self):
        return {
            'layers': self.layers,
            'preprocessing_methods': self.preprocessing_methods
        }
    
    def save(self, path):
        """
        保存模型和预处理参数
        params:
            path: 保存路径
        """
        import os
        import torch
        
        # 保存模型参数
        torch.save({
            'model_state_dict': self.state_dict(),
            'layers': self.layers,
            'preprocessing_methods': self.preprocessing_methods
        }, path)
        
        # 保存预处理参数
        if self.preprocessor:
            preprocessor_path = path.replace('.pth', '_preprocessor.npz')
            self.preprocessor.save(preprocessor_path)
    
    @classmethod
    def load(cls, path):
        """
        加载模型和预处理参数
        params:
            path: 加载路径
        return:
            模型实例
        """
        import torch
        
        # 加载模型参数
        checkpoint = torch.load(path)
        model = cls(
            layers=checkpoint['layers'],
            preprocessing_methods=checkpoint['preprocessing_methods']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载预处理参数
        if model.preprocessing_methods:
            preprocessor_path = path.replace('.pth', '_preprocessor.npz')
            try:
                model.preprocessor = Preprocessor.load(preprocessor_path)
            except:
                pass
        
        return model