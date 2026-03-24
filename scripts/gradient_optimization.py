import torch
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bp_network import BPNetwork
from scripts.train import load_model

def optimize_input(model, target_output, initial_input=None, fixed_inputs=None, 
                  lr=0.1, epochs=1000, tolerance=1e-6):
    """
    通过目标输出值优化输入
    params:
    - model: 训练好的模型
    - target_output: 目标输出值 (shape: (1, output_dim))
    - initial_input: 初始输入值 (shape: (1, input_dim))，如果为None则随机初始化
    - fixed_inputs: 固定的输入索引列表，这些输入值不会被优化
    - lr: 学习率
    - epochs: 优化轮数
    - tolerance: 误差容忍度
    
    return:
    - optimized_input: 优化后的输入值
    - loss_history: 损失历史
    """
    # 确保目标输出是张量
    target_output = torch.tensor(target_output, dtype=torch.float32)
    
    # 初始化输入
    if initial_input is None:
        # 如果没有提供初始输入，完全随机初始化
        initial_input = np.random.randn(1, model.layers[0])
    else:
        # 如果提供了初始输入，只随机化其中为None的元素
        # 先创建一个浮点数数组
        input_array = np.zeros((1, model.layers[0]), dtype=np.float32)
        for i in range(len(initial_input[0])):
            if initial_input[0, i] is None:
                input_array[0, i] = np.random.randn()
            else:
                input_array[0, i] = float(initial_input[0, i])
        initial_input = input_array
    
    # 保存固定的输入值
    fixed_values = None
    if fixed_inputs is not None:
        # 转换为Python浮点数列表
        fixed_values = [float(val) for val in initial_input[0, fixed_inputs]]
    
    # 转换为张量
    initial_input = torch.tensor(initial_input, dtype=torch.float32, requires_grad=True)
    
    # 定义优化器
    optimizer = torch.optim.Adam([initial_input], lr=lr)
    
    # 定义损失函数
    criterion = torch.nn.MSELoss()
    
    loss_history = []
    
    for epoch in range(epochs):
        # 前向传播
        output = model(initial_input)
        
        # 计算损失
        loss = criterion(output, target_output)
        loss_history.append(loss.item())
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新输入
        optimizer.step()
        
        # 重置固定的输入值
        if fixed_inputs is not None and fixed_values is not None:
            with torch.no_grad():
                for i, idx in enumerate(fixed_inputs):
                    initial_input[0, idx] = fixed_values[i]
        
        # 检查收敛
        if loss.item() < tolerance:
            print(f"优化在第 {epoch+1} 轮收敛")
            break
        
        # 打印优化进度
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # 获取优化后的输入
    optimized_input = initial_input.detach().numpy()
    
    return optimized_input, loss_history

def load_trained_model(input_dim=10, output_dim=4):
    """
    加载训练好的模型
    """
    model = BPNetwork(layers=[input_dim, 64, 32, output_dim])
    model = load_model(model, 'models/bp_network.pth')
    return model

if __name__ == "__main__":
    # 加载训练好的模型
    model = load_trained_model()
    
    # 定义目标输出
    target_output = np.array([[1.0, 2.0, 3.0, 4.0]])
    print(f"目标输出: {target_output}")
    
    # 固定部分输入（例如固定第0和第1个输入）
    fixed_inputs = [0, 1]
    print(f"固定输入索引: {fixed_inputs}")
    
    # 初始输入（固定的输入值设为特定值）
    initial_input = np.random.randn(1, 10)
    initial_input[0, fixed_inputs] = [0.5, -0.5]  # 设置固定输入的值
    print(f"初始输入: {initial_input}")
    
    # 优化输入
    optimized_input, loss_history = optimize_input(
        model, target_output, initial_input, fixed_inputs
    )
    
    # 验证优化结果
    model.eval()
    with torch.no_grad():
        final_output = model(torch.tensor(optimized_input, dtype=torch.float32))
    
    print(f"优化后的输入: {optimized_input}")
    print(f"优化后的输出: {final_output.numpy()}")
    print(f"目标输出: {target_output}")
    print(f"最终损失: {loss_history[-1]:.6f}")