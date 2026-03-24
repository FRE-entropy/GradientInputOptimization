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

使用uv安装依赖：

```bash
# 使用uv sync
uv sync
```

这将根据pyproject.toml文件自动安装所有必要的依赖。

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

- 网络结构：使用`layers`参数定义，格式为`[input_dim, hidden1_dim, hidden2_dim, ..., output_dim]`
- 默认结构：`[10, 64, 32, 4]`（输入维度10，两个隐藏层分别为64和32，输出维度4）
- 激活函数：ReLU

### 梯度输入优化

通过目标输出值，使用梯度下降法优化输入值。支持固定部分输入，只优化其余输入。

#### 参数说明：

- `model`：训练好的模型
- `target_output`：目标输出值（shape: (1, output_dim)）
- `initial_input`：初始输入值（shape: (1, input_dim)），如果为None则随机初始化，也可以设置为包含None值的数组，None值会被随机初始化
- `fixed_inputs`：固定的输入索引列表，这些输入值不会被优化
- `lr`：学习率（默认0.1）
- `epochs`：优化轮数（默认1000）
- `tolerance`：误差容忍度（默认1e-6）

## 示例运行

### 训练模型

```
模型配置: {'layers': [10, 64, 32, 4]}
Epoch [10/100], Train Loss: 0.1269, Test Loss: 0.1206
Epoch [20/100], Train Loss: 0.0548, Test Loss: 0.0635
Epoch [30/100], Train Loss: 0.0376, Test Loss: 0.0478
Epoch [40/100], Train Loss: 0.0307, Test Loss: 0.0414
Epoch [50/100], Train Loss: 0.0256, Test Loss: 0.0366
Epoch [60/100], Train Loss: 0.0219, Test Loss: 0.0334
Epoch [70/100], Train Loss: 0.0191, Test Loss: 0.0290
Epoch [80/100], Train Loss: 0.0171, Test Loss: 0.0271
Epoch [90/100], Train Loss: 0.0156, Test Loss: 0.0266
Epoch [100/100], Train Loss: 0.0146, Test Loss: 0.0248
模型保存到 models/bp_network.pth
```

### 预测示例

```
测试输入: [[ 0.12486761 -0.78254129 -0.31395219 -0.04616194 -0.78286651 -0.96044695
   0.28627334 -0.84473741 -0.81065756 -0.93905314]]
预测输出: [[ 0.11654942  0.3719228  -1.7080617  -3.4668758 ]]
```

### 梯度输入优化

```
目标输出: [[1. 2. 3. 4.]]
初始输入: [[0.5 None None None None None None None None None]]
Epoch [100/1000], Loss: 0.000230
优化在第 142 轮收敛
优化后的输入: [[ 0.5        -1.1055715   0.88773125  0.36921176  0.45868614  3.2019565
  -0.4965332   0.23538372  0.46930248  0.5527371 ]]
优化后的输出: [[0.9999611 2.0003686 2.9992392 3.9996922]]
目标输出: [[1. 2. 3. 4.]]
最终损失: 0.000001
```

## 注意事项

1. 本项目使用合成数据进行训练，实际应用中需要替换为真实数据
2. 梯度输入优化的效果取决于模型的质量和优化参数的设置
3. 固定输入时，初始输入中这些位置的值会在优化过程中保持不变
4. 初始输入可以包含None值，这些位置会被随机初始化
5. 对于复杂问题，可能需要调整网络结构和优化参数
6. 梯度输入优化不会改变模型本身的参数，只会优化输入值

## 扩展建议

1. 添加更多的网络结构选项（如不同的激活函数、正则化等）
2. 支持更复杂的数据预处理和后处理
3. 添加可视化工具，展示优化过程
4. 实现批量优化功能
5. 考虑使用更先进的优化算法

## 许可证

本项目为开源项目，可自由使用和修改。
