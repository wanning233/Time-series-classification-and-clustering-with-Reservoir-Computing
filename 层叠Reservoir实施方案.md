# 层叠Reservoir（StackedReservoir）完整实施方案

## 1. 方案概述

### 1.1 目标
在现有Reservoir模块基础上实现层叠Reservoir架构（StackedReservoir），通过多个Reservoir串联，逐层提取抽象特征，提升模型表达能力。

### 1.2 核心思想
- **层叠结构**：多个Reservoir串联，每层处理前层的输出
- **逐层抽象**：底层捕获局部模式，高层捕获全局模式
- **状态序列传递**：每层的输出状态序列（形状[N,T,H]）直接作为下一层的输入
- **参数配置**：每层可独立配置超参数，也支持统一配置或默认渐进式配置

### 1.3 优势
- 增强特征表达能力
- 支持多时间尺度特征提取
- 保持Reservoir计算效率优势
- 向后兼容现有代码（保留原始RC_model类）

## 2. 实现细节

### 2.1 类设计
实现了 `StackedRC_model` 类，位于 `reservoir_computing/modules.py` 中。

### 2.2 层间传递机制
- **第一层**：处理原始输入 `X` [N,T,V]
- **中间层**：处理前一层状态序列 `states` [N,T,H]
- **最后一层**：输出状态序列用于生成表示（与RC_model相同）

### 2.3 配置方式
支持三种配置方式：

1. **统一配置**：传入单个字典，所有层使用相同超参数
```python
config = {
    'n_internal_units': 200,
    'spectral_radius': 0.99,
    'connectivity': 0.3
}
model = StackedRC_model(n_layers=2, reservoir_configs=config)
```

2. **独立配置**：传入字典列表，每层独立配置
```python
configs = [
    {'n_internal_units': 300, 'spectral_radius': 0.95},
    {'n_internal_units': 200, 'spectral_radius': 0.99}
]
model = StackedRC_model(n_layers=2, reservoir_configs=configs)
```

3. **默认配置**：传入None，使用默认渐进式配置
```python
model = StackedRC_model(n_layers=2, reservoir_configs=None)
# 默认：第一层200单元，第二层150单元
```

### 2.4 使用示例

#### 基本使用（聚类）
```python
from reservoir_computing.modules import StackedRC_model
import numpy as np

# 初始化层叠模型
model = StackedRC_model(
    n_layers=2,
    reservoir_configs=None,  # 使用默认配置
    readout_type=None  # 聚类时设置为None
)

# 训练（生成表示）
X = ...  # 形状 [N, T, V]
model.fit(X)

# 获取表示用于聚类
representations = model.input_repr  # 形状 [N, D]
```

#### 分类任务
```python
model = StackedRC_model(
    n_layers=2,
    readout_type='lin',  # 岭回归
    w_ridge=1.0
)

# 训练
Xtr, Ytr = ...  # 训练数据和标签
model.fit(Xtr, Ytr)

# 预测
Xte = ...  # 测试数据
predictions = model.predict(Xte)
```

## 3. 对比实验

### 3.1 实验脚本
提供了对比实验脚本 `examples/stacked_clustering_comparison.py`，用于对比原始RC_model和层叠StackedRC_model的聚类效果。

### 3.2 评估指标
- **NMI (Normalized Mutual Information)**：标准化互信息
- **ARI (Adjusted Rand Index)**：调整兰德指数
- **聚类数量**：发现的聚类数量

### 3.3 运行对比实验
```bash
cd examples
python stacked_clustering_comparison.py
```

## 4. 技术细节

### 4.1 实现位置
- **类定义**：`reservoir_computing/modules.py` 中的 `StackedRC_model` 类
- **对比脚本**：`examples/stacked_clustering_comparison.py`

### 4.2 关键方法
- `__init__()`: 初始化多个Reservoir对象
- `fit()`: 逐层计算状态，生成表示
- `predict()`: 逐层处理测试数据，进行预测

### 4.3 与原始模型的区别
- **原始RC_model**：单层Reservoir，直接处理输入
- **层叠StackedRC_model**：多层Reservoir串联，逐层提取特征

### 4.4 向后兼容性
- 原始 `RC_model` 类完全保持不变
- 现有代码无需修改即可继续使用
- 新功能通过新类实现

## 5. 性能考虑

### 5.1 计算复杂度
- 层叠模型的计算时间约为单层模型的 `n_layers` 倍
- 内存占用增加，但仍在可接受范围内

### 5.2 参数选择建议
- **层数**：建议2-3层，过多层数可能带来过拟合风险
- **单元数**：建议使用渐进式配置，高层单元数略少于底层
- **谱半径**：保持0.95-0.99范围，确保回声状态属性

---
