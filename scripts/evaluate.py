import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bp_network import BPNetwork
from scripts.train import load_model
from scripts.gradient_optimization import optimize_input
from data.data_loader import get_data_loaders

def evaluate_model_performance(model, X_test, Y_test):
    """
    评估模型性能
    params:
        model: 训练好的模型
        X_test: 测试输入数据
        Y_test: 测试目标数据
    return:
        评估指标字典
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_tensor)
        Y_tensor = torch.tensor(Y_test, dtype=torch.float32)
        
        # 计算均方误差
        mse_tensor = torch.mean((outputs - Y_tensor) ** 2)
        mse = mse_tensor.item()
        # 计算均方根误差
        rmse = torch.sqrt(mse_tensor).item()
        # 计算平均绝对误差
        mae = torch.mean(torch.abs(outputs - Y_tensor)).item()
        # 计算R²
        y_mean = torch.mean(Y_tensor)
        ss_total = torch.sum((Y_tensor - y_mean) ** 2)
        ss_residual = torch.sum((Y_tensor - outputs) ** 2)
        r2 = 1 - (ss_residual / ss_total).item()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def evaluate_gradient_optimization(model, target_outputs, initial_inputs=None, fixed_inputs=None):
    """
    评估梯度优化效果
    params:
        model: 训练好的模型
        target_outputs: 目标输出值列表
        initial_inputs: 初始输入值列表
        fixed_inputs: 固定的输入索引列表
    return:
        优化结果列表
    """
    results = []
    
    for i, target_output in enumerate(target_outputs):
        print(f"\n优化测试 {i+1}/{len(target_outputs)}")
        print(f"目标输出: {target_output}")
        
        # 使用随机初始输入或提供的初始输入
        if initial_inputs is not None and i < len(initial_inputs):
            initial_input = initial_inputs[i]
        else:
            initial_input = np.random.randn(1, model.layers[0])
        
        print(f"初始输入: {initial_input}")
        
        # 优化输入
        optimized_input, loss_history = optimize_input(
            model, target_output, initial_input, fixed_inputs
        )
        
        # 验证优化结果
        model.eval()
        with torch.no_grad():
            final_output = model(torch.tensor(optimized_input, dtype=torch.float32))
        
        # 计算优化误差
        optimization_error = np.mean((final_output.numpy() - target_output) ** 2)
        
        results.append({
            'target_output': target_output,
            'initial_input': initial_input,
            'optimized_input': optimized_input,
            'final_output': final_output.numpy(),
            'optimization_error': optimization_error,
            'loss_history': loss_history,
            'convergence_epochs': len(loss_history)
        })
        
        print(f"优化后的输入: {optimized_input}")
        print(f"优化后的输出: {final_output.numpy()}")
        print(f"目标输出: {target_output}")
        print(f"优化误差: {optimization_error:.6f}")
        print(f"收敛轮数: {len(loss_history)}")
    
    return results

def plot_loss_curve(loss_history, title="Loss Curve"):
    """
    绘制损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('output/evaluation/evaluation_loss_curve.png')
    plt.close()

def plot_prediction_comparison(Y_true, Y_pred, title="Prediction vs True"):
    """
    绘制预测值与真实值的对比
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_true, Y_pred, alpha=0.5)
    plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig('output/evaluation/evaluation_prediction_comparison.png')
    plt.close()

def generate_evaluation_report(model, X_test, Y_test, optimization_results):
    """
    生成评估报告
    """
    # 评估模型性能
    model_metrics = evaluate_model_performance(model, X_test, Y_test)
    
    # 计算优化结果统计
    optimization_errors = [result['optimization_error'] for result in optimization_results]
    convergence_epochs = [result['convergence_epochs'] for result in optimization_results]
    
    # 生成报告
    report = "# 模型评估报告\n\n"
    
    report += "## 1. 模型性能评估\n\n"
    report += f"- 均方误差 (MSE): {model_metrics['mse']:.6f}\n"
    report += f"- 均方根误差 (RMSE): {model_metrics['rmse']:.6f}\n"
    report += f"- 平均绝对误差 (MAE): {model_metrics['mae']:.6f}\n"
    report += f"- R² 评分: {model_metrics['r2']:.6f}\n\n"
    
    report += "## 2. 梯度优化评估\n\n"
    report += f"- 平均优化误差: {np.mean(optimization_errors):.6f}\n"
    report += f"- 优化误差标准差: {np.std(optimization_errors):.6f}\n"
    report += f"- 平均收敛轮数: {np.mean(convergence_epochs):.2f}\n"
    report += f"- 收敛轮数标准差: {np.std(convergence_epochs):.2f}\n\n"
    
    report += "## 3. 优化示例\n\n"
    for i, result in enumerate(optimization_results[:3]):  # 只展示前3个示例
        report += f"### 示例 {i+1}\n"
        report += f"- 目标输出: {result['target_output']}\n"
        report += f"- 初始输入: {result['initial_input']}\n"
        report += f"- 优化后的输入: {result['optimized_input']}\n"
        report += f"- 优化后的输出: {result['final_output']}\n"
        report += f"- 优化误差: {result['optimization_error']:.6f}\n"
        report += f"- 收敛轮数: {result['convergence_epochs']}\n\n"
    
    # 保存报告
    with open('output/evaluation/evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("评估报告已生成: output/evaluation/evaluation_report.md")

def main():
    """
    主函数
    """
    # 加载训练好的模型
    print("加载模型...")
    model = BPNetwork.load('models/bp_network.pth')
    print(f"模型配置: {model.get_config()}")
    
    # 加载数据
    print("加载数据...")
    X = np.load('data/X.npy')
    Y = np.load('data/Y.npy')
    
    # 创建数据加载器
    _, test_loader = get_data_loaders(X, Y, test_split=0.2)
    
    # 提取测试数据
    X_test = []
    Y_test = []
    for X_batch, Y_batch in test_loader:
        X_test.extend(X_batch.numpy())
        Y_test.extend(Y_batch.numpy())
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    # 评估模型性能
    print("\n评估模型性能...")
    model_metrics = evaluate_model_performance(model, X_test, Y_test)
    print(f"模型性能指标:")
    print(f"- MSE: {model_metrics['mse']:.6f}")
    print(f"- RMSE: {model_metrics['rmse']:.6f}")
    print(f"- MAE: {model_metrics['mae']:.6f}")
    print(f"- R²: {model_metrics['r2']:.6f}")
    
    # 生成预测值用于可视化
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_pred = model(X_tensor).numpy()
    
    # 可视化预测结果
    print("\n生成可视化结果...")
    # 展平数据用于散点图
    Y_true_flat = Y_test.flatten()
    Y_pred_flat = Y_pred.flatten()
    plot_prediction_comparison(Y_true_flat, Y_pred_flat)
    
    # 评估梯度优化效果
    print("\n评估梯度优化效果...")
    # 生成5个随机目标输出
    target_outputs = []
    for _ in range(5):
        # 生成与输出维度匹配的随机目标值
        target = np.random.randn(1, model.layers[-1])
        target_outputs.append(target)
    
    # 评估梯度优化
    optimization_results = evaluate_gradient_optimization(model, target_outputs)
    
    # 绘制优化损失曲线（使用第一个优化结果）
    if optimization_results:
        plot_loss_curve(optimization_results[0]['loss_history'], "Gradient Optimization Loss Curve")
    
    # 生成评估报告
    print("\n生成评估报告...")
    generate_evaluation_report(model, X_test, Y_test, optimization_results)
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()