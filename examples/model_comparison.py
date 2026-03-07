"""
四种Reservoir计算模型对比实验

本脚本对比了以下四种模型在时间序列分类任务上的表现：
1. RC_model: 原始单层Reservoir模型
2. StackedRC_model: 层叠Reservoir模型
3. MultiExpertStackedRC_model: 多专家+残差式层叠Reservoir模型
4. MoEStackedRC_model: 混合专家（MoE）层叠Reservoir模型

评估指标：
- 训练时间
- 测试准确率
- F1分数
- 模型参数量（近似）
"""

import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

from reservoir_computing.modules import (
    RC_model,
    StackedRC_model,
    MultiExpertStackedRC_model,
    MoEStackedRC_model
)
from reservoir_computing.datasets import ClfLoader

# 设置随机种子以确保可重复性
np.random.seed(42)

print("=" * 80)
print("四种Reservoir计算模型对比实验")
print("=" * 80)

# ============ 加载数据 ============
print("\n1. 加载数据...")
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# One-hot编码标签
onehot_encoder = OneHotEncoder(sparse_output=False)
Ytr_onehot = onehot_encoder.fit_transform(Ytr)
Yte_onehot = onehot_encoder.transform(Yte)

# 获取真实类别标签（用于评估）
Ytr_true = np.argmax(Ytr_onehot, axis=1)
Yte_true = np.argmax(Yte_onehot, axis=1)

print(f"   训练集形状: {Xtr.shape}, 类别数: {len(np.unique(Ytr_true))}")
print(f"   测试集形状: {Xte.shape}")

# ============ 结果存储 ============
results = {}

# ============ 模型1: RC_model (原始单层) ============
print("\n" + "=" * 80)
print("模型1: RC_model (原始单层Reservoir)")
print("=" * 80)

model1 = RC_model(
    n_internal_units=400,
    spectral_radius=0.99,
    connectivity=0.3,
    input_scaling=0.2,
    readout_type='lin',
    w_ridge=1.0,
    mts_rep='mean'
)

start_time = time.time()
model1.fit(Xtr, Ytr_onehot, verbose=True)
train_time1 = time.time() - start_time

pred1 = model1.predict(Xte)
acc1 = accuracy_score(Yte_true, pred1)
f1_1 = f1_score(Yte_true, pred1, average='weighted')

results['RC_model'] = {
    'train_time': train_time1,
    'accuracy': acc1,
    'f1': f1_1,
    'model': model1
}

print(f"   训练时间: {train_time1:.2f} 秒")
print(f"   测试准确率: {acc1:.4f}")
print(f"   F1分数: {f1_1:.4f}")

# ============ 模型2: StackedRC_model (层叠Reservoir) ============
print("\n" + "=" * 80)
print("模型2: StackedRC_model (层叠Reservoir, 2层)")
print("=" * 80)

model2 = StackedRC_model(
    n_layers=2,
    reservoir_configs=None,  # 使用默认渐进式配置
    readout_type='lin',
    w_ridge=1.0,
    mts_rep='mean'
)

start_time = time.time()
model2.fit(Xtr, Ytr_onehot, verbose=True)
train_time2 = time.time() - start_time

pred2 = model2.predict(Xte)
acc2 = accuracy_score(Yte_true, pred2)
f1_2 = f1_score(Yte_true, pred2, average='weighted')

results['StackedRC_model'] = {
    'train_time': train_time2,
    'accuracy': acc2,
    'f1': f1_2,
    'model': model2
}

print(f"   训练时间: {train_time2:.2f} 秒")
print(f"   测试准确率: {acc2:.4f}")
print(f"   F1分数: {f1_2:.4f}")

# ============ 模型3: MultiExpertStackedRC_model (多专家+残差) ============
print("\n" + "=" * 80)
print("模型3: MultiExpertStackedRC_model (多专家+残差式层叠, 2层3专家)")
print("=" * 80)

model3 = MultiExpertStackedRC_model(
    n_layers=2,
    n_experts=3,
    reservoir_configs=None,  # 使用默认渐进式配置
    readout_type='lin',
    w_ridge=1.0,
    mts_rep='mean'
)

start_time = time.time()
model3.fit(Xtr, Ytr_onehot, verbose=True)
train_time3 = time.time() - start_time

pred3 = model3.predict(Xte)
acc3 = accuracy_score(Yte_true, pred3)
f1_3 = f1_score(Yte_true, pred3, average='weighted')

results['MultiExpertStackedRC_model'] = {
    'train_time': train_time3,
    'accuracy': acc3,
    'f1': f1_3,
    'model': model3
}

print(f"   训练时间: {train_time3:.2f} 秒")
print(f"   测试准确率: {acc3:.4f}")
print(f"   F1分数: {f1_3:.4f}")

# ============ 模型4: MoEStackedRC_model (混合专家MoE) ============
print("\n" + "=" * 80)
print("模型4: MoEStackedRC_model (混合专家MoE, 2层3专家)")
print("=" * 80)

model4 = MoEStackedRC_model(
    n_layers=2,
    n_experts=3,
    reservoir_configs=None,  # 使用默认渐进式配置
    gate_lr=0.01,
    gate_epochs=100,  # 门控网络训练轮数（可调整）
    gate_reg=1e-4,
    intra_gate_input='mean',
    readout_type='lin',
    w_ridge=1.0,
    mts_rep='mean'
)

start_time = time.time()
model4.fit(Xtr, Ytr_onehot, verbose=True)
train_time4 = time.time() - start_time

pred4 = model4.predict(Xte)
acc4 = accuracy_score(Yte_true, pred4)
f1_4 = f1_score(Yte_true, pred4, average='weighted')

results['MoEStackedRC_model'] = {
    'train_time': train_time4,
    'accuracy': acc4,
    'f1': f1_4,
    'model': model4
}

print(f"   训练时间: {train_time4:.2f} 秒")
print(f"   测试准确率: {acc4:.4f}")
print(f"   F1分数: {f1_4:.4f}")

# ============ 结果汇总 ============
print("\n" + "=" * 80)
print("实验结果汇总")
print("=" * 80)

print(f"\n{'模型':<30} {'训练时间(秒)':<15} {'准确率':<12} {'F1分数':<12}")
print("-" * 80)

for model_name, result in results.items():
    print(f"{model_name:<30} {result['train_time']:<15.2f} {result['accuracy']:<12.4f} {result['f1']:<12.4f}")

# 找出最佳模型
best_acc_model = max(results.items(), key=lambda x: x[1]['accuracy'])
best_f1_model = max(results.items(), key=lambda x: x[1]['f1'])
fastest_model = min(results.items(), key=lambda x: x[1]['train_time'])

print("\n" + "-" * 80)
print(f"最佳准确率模型: {best_acc_model[0]} (准确率: {best_acc_model[1]['accuracy']:.4f})")
print(f"最佳F1分数模型: {best_f1_model[0]} (F1: {best_f1_model[1]['f1']:.4f})")
print(f"最快训练模型: {fastest_model[0]} (时间: {fastest_model[1]['train_time']:.2f}秒)")

# ============ 性能提升分析 ============
print("\n" + "=" * 80)
print("性能提升分析（相对于RC_model基准）")
print("=" * 80)

baseline_acc = results['RC_model']['accuracy']
baseline_f1 = results['RC_model']['f1']
baseline_time = results['RC_model']['train_time']

for model_name, result in results.items():
    if model_name == 'RC_model':
        continue
    acc_improve = (result['accuracy'] - baseline_acc) / baseline_acc * 100
    f1_improve = (result['f1'] - baseline_f1) / baseline_f1 * 100
    time_ratio = result['train_time'] / baseline_time
    
    print(f"\n{model_name}:")
    print(f"  准确率提升: {acc_improve:+.2f}%")
    print(f"  F1分数提升: {f1_improve:+.2f}%")
    print(f"  训练时间倍数: {time_ratio:.2f}x")

# ============ MoE模型门控权重可视化（可选） ============
print("\n" + "=" * 80)
print("MoE模型门控权重分析（前5个测试样本）")
print("=" * 80)

try:
    intra_w, inter_w = model4.get_gate_weights(Xte[:5])
    print(f"\n层内门控权重（每层专家权重，形状: {[w.shape for w in intra_w]}）")
    for layer_idx, w in enumerate(intra_w):
        print(f"  层{layer_idx}专家权重（前3个样本）:")
        print(f"    {w[:3].round(4)}")
        print(f"    均值: {w.mean(axis=0).round(4)}")
    
    print(f"\n层间门控权重（各层权重，形状: {inter_w.shape}）")
    print(f"  前3个样本:")
    print(f"    {inter_w[:3].round(4)}")
    print(f"    均值: {inter_w.mean(axis=0).round(4)}")
    print(f"    说明: 均值显示各层的相对重要性")
except Exception as e:
    print(f"  无法获取门控权重: {e}")

print("\n" + "=" * 80)
print("实验完成！")
print("=" * 80)
