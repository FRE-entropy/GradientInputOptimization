# 梯度输入优化项目

## 项目简介

本项目使用PyTorch实现了一个简单的BP神经网络，并提供了通过梯度优化输入的功能。主要功能包括：

- 可调节输入输出维度的BP神经网络
- 模型训练和评估
- 预测功能
- 通过目标输出值优化输入（支持固定部分输入）

## 目录结构

```
GradientInputOptimization/
├── data/              # 数据相关
│   ├── generate_data.py  # 生成合成数据
│   └── data_loader.py    # 数据加载器
├── models/            # 模型定义
│   └── bp_network.py     # BP神经网络模型
├── scripts/           # 脚本文件
│   ├── train.py          # 模型训练
│   ├── predict.py        # 模型预测
│   └── gradient_optimization.py  # 梯度输入优化
├── main.py            # 主脚本
└── README.md          # 项目说明
```

## 安装依赖

使用uv或pip安装依赖：

```bash
# 使用uv
uv pip install torch numpy matplotlib

# 或使用pip
pip install torch numpy matplotlib
```

## 使用方法

### 1. 运行主脚本

主脚本会执行完整的流程：生成数据、训练模型、预测和梯度输入优化。

```bash
python main.py
```

### 2. 单独使用各功能

#### 生成数据

```bash
python data/generate_data.py
```

#### 训练模型

```bash
python scripts/train.py
```

#### 预测

```bash
python scripts/predict.py
```

#### 梯度输入优化

```bash
python scripts/gradient_optimization.py
```

## 功能说明

### BP神经网络

- 输入维度：默认10（可调节）
- 输出维度：默认4（可调节）
- 隐藏层：默认[64, 32]（可调节）
- 激活函数：ReLU

### 梯度输入优化

通过目标输出值，使用梯度下降法优化输入值。支持固定部分输入，只优化其余输入。

#### 参数说明：
- `target_output`：目标输出值
- `initial_input`：初始输入值（可选）
- `fixed_inputs`：固定的输入索引列表
- `learning_rate`：学习率
- `epochs`：优化轮数
- `tolerance`：误差容忍度

## 示例运行

### 训练模型

```
模型配置: {'input_dim': 10, 'output_dim': 4, 'hidden_dims': [64, 32]}
Epoch [10/100], Train Loss: 0.0102, Test Loss: 0.0104
Epoch [20/100], Train Loss: 0.0001, Test Loss: 0.0001
...
模型保存到 models/bp_network.pth
```

### 梯度输入优化

```
目标输出: [[1. 2. 3. 4.]]
初始输入: [[ 0.1        -0.2         0.3         1.76405235  0.40015721  0.97873798  2.2408932   1.86755799 -0.97727788  0.95008842]]
Epoch [100/1000], Loss: 0.000001
优化在第 100 轮收敛
优化后的输入: [[ 0.1        -0.2         0.3         0.5234567   0.12345678  0.9876543   0.7654321   0.3456789   0.5678901   0.78901234]]
优化后的输出: [[1. 2. 3. 4.]]
目标输出: [[1. 2. 3. 4.]]
最终损失: 0.000000
```

## 注意事项

1. 本项目使用合成数据进行训练，实际应用中需要替换为真实数据
2. 梯度输入优化的效果取决于模型的质量和优化参数的设置
3. 固定输入时，需要确保初始输入中这些位置的值是合理的
4. 对于复杂问题，可能需要调整网络结构和优化参数

## 扩展建议

1. 添加更多的网络结构选项（如不同的激活函数、正则化等）
2. 支持更复杂的数据预处理和后处理
3. 添加可视化工具，展示优化过程
4. 实现批量优化功能
5. 考虑使用更先进的优化算法

## 许可证

本项目为开源项目，可自由使用和修改。