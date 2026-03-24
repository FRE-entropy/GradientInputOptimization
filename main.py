import numpy as np
import torch
from data.generate_data import generate_synthetic_data
from data.data_loader import get_data_loaders
from models.bp_network import BPNetwork
from scripts.train import train_model, save_model, load_model
from scripts.predict import predict
from scripts.gradient_optimization import optimize_input

def main():
    print("=== 梯度输入优化项目 ===")
    
    # 1. 生成数据
    print("\n1. 生成训练数据...")
    X, Y = generate_synthetic_data(input_dim=10, output_dim=4, num_samples=1000)
    
    # 2. 创建数据加载器
    print("\n2. 创建数据加载器...")
    train_loader, test_loader = get_data_loaders(X, Y, batch_size=32, test_split=0.2)
    
    # 3. 创建和训练模型
    print("\n3. 创建和训练模型...")
    model = BPNetwork(layers=[10, 64, 32, 4])
    print(f"模型配置: {model.get_config()}")
    
    model = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)
    save_model(model, 'models/bp_network.pth')
    
    # 4. 预测示例
    print("\n4. 预测示例...")
    test_sample = np.random.randn(1, 10)
    print(f"测试输入: {test_sample}")
    prediction = predict(model, test_sample)
    print(f"预测输出: {prediction}")
    
    # 5. 梯度输入优化示例
    print("\n5. 梯度输入优化示例...")
    target_output = np.array([[1.0, 2.0, 3.0, 4.0]])
    print(f"目标输出: {target_output}")
    
    # 固定部分输入
    fixed_inputs = [0, 1, 2]  # 固定前3个输入
    
    # 初始输入
    initial_input = np.random.randn(1, 10)
    initial_input[0, fixed_inputs] = [0.1, -0.2, 0.3]  # 设置固定输入的值
    print(f"初始输入: {initial_input}")
    
    # 优化输入
    optimized_input, loss_history = optimize_input(
        model, target_output, initial_input, fixed_inputs, lr=0.1
    )
    
    # 验证结果
    model.eval()
    with torch.no_grad():
        final_output = model(torch.tensor(optimized_input, dtype=torch.float32))
    
    print(f"优化后的输入: {optimized_input}")
    print(f"优化后的输出: {final_output.numpy()}")
    print(f"目标输出: {target_output}")
    print(f"最终损失: {loss_history[-1]:.6f}")
    
    print("\n=== 项目完成 ===")

if __name__ == "__main__":
    main()