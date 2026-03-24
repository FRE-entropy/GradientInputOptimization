import numpy as np
import torch
import os

def generate_synthetic_data(input_dim=10, output_dim=4, num_samples=1000):
    """
    生成合成数据
    params:
        input_dim: 输入特征维度，默认为10
        output_dim: 输出标签维度，默认为4
        num_samples: 样本数，默认为1000
    return:
        X: 输入特征数据，numpy数组
        Y: 目标标签数据，numpy数组
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 生成输入数据
    X = np.random.randn(num_samples, input_dim)
    
    # 生成权重矩阵
    weights = np.random.randn(input_dim, output_dim)
    bias = np.random.randn(output_dim)
    
    # 生成输出数据（添加一些噪声）
    Y = X.dot(weights) + bias + np.random.randn(num_samples, output_dim) * 0.1
    
    # 保存数据
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    np.save(os.path.join(data_dir, 'X.npy'), X)
    np.save(os.path.join(data_dir, 'Y.npy'), Y)
    
    print(f"生成数据完成，保存到 {data_dir} 目录")
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}, 样本数: {num_samples}")
    
    return X, Y

if __name__ == "__main__":
    generate_synthetic_data()