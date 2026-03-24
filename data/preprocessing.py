import numpy as np
import os

class Preprocessor:
    """
    数据预处理类，支持每个输入特征使用不同的预处理方法
    """
    def __init__(self, methods=None):
        """
        初始化预处理器
        params:
            methods: 预处理方法列表，每个元素对应一个输入特征的预处理方法
                    支持的方法：'none', 'minmax', 'standard'
        """
        self.methods = methods
        self.params = {}
        self.input_dim = len(methods) if methods else 0
    
    def fit(self, X):
        """
        拟合预处理参数
        params:
            X: 输入数据，shape (n_samples, input_dim)
        """
        if self.methods is None:
            return
        
        for i in range(self.input_dim):
            method = self.methods[i]
            if method == 'minmax':
                min_val = np.min(X[:, i])
                max_val = np.max(X[:, i])
                self.params[i] = {'min': min_val, 'max': max_val}
            elif method == 'standard':
                mean_val = np.mean(X[:, i])
                std_val = np.std(X[:, i])
                self.params[i] = {'mean': mean_val, 'std': std_val}
            elif method == 'none':
                self.params[i] = {}
    
    def transform(self, X):
        """
        应用预处理
        params:
            X: 输入数据，shape (n_samples, input_dim)
        return:
            预处理后的数据
        """
        if self.methods is None:
            return X
        
        X_transformed = np.copy(X)
        for i in range(self.input_dim):
            method = self.methods[i]
            if method == 'minmax':
                min_val = self.params[i]['min']
                max_val = self.params[i]['max']
                if max_val > min_val:
                    X_transformed[:, i] = (X[:, i] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = self.params[i]['mean']
                std_val = self.params[i]['std']
                if std_val > 0:
                    X_transformed[:, i] = (X[:, i] - mean_val) / std_val
        return X_transformed
    
    def inverse_transform(self, X):
        """
        反向预处理，将预处理后的数据转换回原始尺度
        params:
            X: 预处理后的数据，shape (n_samples, input_dim)
        return:
            原始尺度的数据
        """
        if self.methods is None:
            return X
        
        X_inverse = np.copy(X)
        for i in range(self.input_dim):
            method = self.methods[i]
            if method == 'minmax':
                min_val = self.params[i]['min']
                max_val = self.params[i]['max']
                X_inverse[:, i] = X[:, i] * (max_val - min_val) + min_val
            elif method == 'standard':
                mean_val = self.params[i]['mean']
                std_val = self.params[i]['std']
                X_inverse[:, i] = X[:, i] * std_val + mean_val
        return X_inverse
    
    def save(self, path):
        """
        保存预处理参数
        params:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, methods=self.methods, params=self.params)
    
    @classmethod
    def load(cls, path):
        """
        加载预处理参数
        params:
            path: 加载路径
        return:
            预处理器实例
        """
        data = np.load(path, allow_pickle=True)
        
        # 正确加载方法列表
        if 'methods' in data:
            methods = data['methods']
            if isinstance(methods, np.ndarray):
                # 如果是数组，尝试转换为列表
                if methods.ndim == 0:
                    methods = methods.item()
                else:
                    methods = methods.tolist()
        else:
            methods = None
        
        # 正确加载参数字典
        if 'params' in data:
            params = data['params']
            if isinstance(params, np.ndarray):
                params = params.item()
        else:
            params = {}
        
        preprocessor = cls(methods)
        preprocessor.params = params
        preprocessor.input_dim = len(methods) if methods else 0
        return preprocessor

if __name__ == "__main__":
    # 测试预处理类
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    
    # 为每个特征指定不同的预处理方法
    methods = ['minmax', 'standard', 'none']
    preprocessor = Preprocessor(methods)
    
    # 拟合预处理参数
    preprocessor.fit(X)
    print("预处理参数:", preprocessor.params)
    
    # 应用预处理
    X_transformed = preprocessor.transform(X)
    print("预处理后的数据:", X_transformed)
    
    # 反向预处理
    X_inverse = preprocessor.inverse_transform(X_transformed)
    print("反向预处理后的数据:", X_inverse)
    
    # 测试保存和加载
    preprocessor.save('data/preprocessor.npz')
    loaded_preprocessor = Preprocessor.load('data/preprocessor.npz')
    X_transformed_loaded = loaded_preprocessor.transform(X)
    print("加载后预处理的数据:", X_transformed_loaded)