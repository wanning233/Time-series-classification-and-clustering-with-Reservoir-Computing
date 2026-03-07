"""
分类任务对比实验：原始 RC vs 串联层叠 RC vs 多专家层叠 RC

本脚本对比了三种模型在分类任务上的表现：
1. 原始 RC_model
2. 串联层叠 StackedRC_model（2、3、4、5层）
3. 多专家 + 残差层叠 MultiExpertStackedRC_model（3、5层，不同专家数）

使用相同的评估指标（Accuracy, F1），便于直接对比。
"""

import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from reservoir_computing.modules import RC_model, StackedRC_model, MultiExpertStackedRC_model
from reservoir_computing.utils import compute_test_scores
from reservoir_computing.datasets import ClfLoader

# 设置随机种子以确保可重复性
np.random.seed(0)

print("=" * 80)
print("分类任务对比实验：原始 RC vs 串联层叠 RC vs 多专家层叠 RC")
print("=" * 80)

# 加载数据
print("\n1. 加载数据...")
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
Ytr = onehot_encoder.fit_transform(Ytr)
Yte = onehot_encoder.transform(Yte)

print(f"   训练集形状: {Xtr.shape}")
print(f"   测试集形状: {Xte.shape}")
print(f"   类别数量: {Ytr.shape[1]}")

# ============ 原始RC模型 ============
print("\n2. 训练原始RC模型...")
start_time = time.time()
rcm_original = RC_model(n_internal_units=400, readout_type='lin', w_ridge=1.0)
rcm_original.fit(Xtr, Ytr, verbose=True)
train_time_original = time.time() - start_time

pred_original = rcm_original.predict(Xte)
accuracy_original, f1_original = compute_test_scores(pred_original, Yte)
print(f"   训练时间: {train_time_original:.2f} 秒")
print(f"   Accuracy: {accuracy_original:.4f}")
print(f"   F1 Score: {f1_original:.4f}")

# ============ 串联层叠RC模型 ============
print("\n3. 训练串联层叠RC模型（2、3、4、5层）...")
stacked_results = {}
stacked_train_times = {}

for n_layers in [2, 3, 4, 5]:
    print(f"\n   训练 {n_layers} 层层叠模型...")
    start_time = time.time()
    rcm_stacked = StackedRC_model(
        n_layers=n_layers,
        reservoir_configs=None,  # 使用默认渐进式配置
        readout_type='lin',
        w_ridge=1.0
    )
    rcm_stacked.fit(Xtr, Ytr, verbose=False)
    train_time = time.time() - start_time
    
    pred_stacked = rcm_stacked.predict(Xte)
    accuracy, f1 = compute_test_scores(pred_stacked, Yte)
    
    stacked_results[n_layers] = {
        'accuracy': accuracy,
        'f1': f1
    }
    stacked_train_times[n_layers] = train_time
    
    print(f"   训练时间: {train_time:.2f} 秒")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")

# ============ 多专家 + 残差层叠RC模型 ============
print("\n4. 训练多专家 + 残差层叠RC模型（多种层数与专家数组合）...")
multi_expert_results = {}
multi_expert_train_times = {}

# 实验组合：[(层数, 专家数), ...]
multi_expert_settings = [
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 5),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 5),
]

for n_layers, n_experts in multi_expert_settings:
    print(f"\n   训练 多专家 {n_layers} 层层叠模型（n_experts={n_experts}）...")
    start_time = time.time()
    rcm_me = MultiExpertStackedRC_model(
        n_layers=n_layers,
        n_experts=n_experts,
        reservoir_configs=None,  # 使用默认渐进式配置
        mts_rep='mean',
        readout_type='lin',
        w_ridge=1.0
    )
    rcm_me.fit(Xtr, Ytr, verbose=False)
    train_time = time.time() - start_time
    
    pred_me = rcm_me.predict(Xte)
    accuracy, f1 = compute_test_scores(pred_me, Yte)
    
    multi_expert_results[(n_layers, n_experts)] = {
        'accuracy': accuracy,
        'f1': f1
    }
    multi_expert_train_times[(n_layers, n_experts)] = train_time
    
    print(f"   训练时间: {train_time:.2f} 秒")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")

# ============ 结果对比 ============
print("\n" + "=" * 100)
print("结果对比总结（原始 RC 与串联层叠 RC）")
print("=" * 100)

# 表头
header = f"{'指标':<15} {'原始RC':<12} {'2层':<12} {'3层':<12} {'4层':<12} {'5层':<12}"
print(f"\n{header}")
print("-" * 100)

# Accuracy对比
acc_row = f"{'Accuracy':<15} {accuracy_original:<12.4f}"
for n_layers in [2, 3, 4, 5]:
    acc_val = stacked_results[n_layers]['accuracy']
    acc_row += f" {acc_val:<11.4f}"
print(acc_row)

# F1对比
f1_row = f"{'F1 Score':<15} {f1_original:<12.4f}"
for n_layers in [2, 3, 4, 5]:
    f1_val = stacked_results[n_layers]['f1']
    f1_row += f" {f1_val:<11.4f}"
print(f1_row)

# 训练时间对比
time_row = f"{'训练时间(秒)':<15} {train_time_original:<12.2f}"
for n_layers in [2, 3, 4, 5]:
    time_val = stacked_train_times[n_layers]
    time_row += f" {time_val:<11.2f}"
print(time_row)

# 多专家层叠模型结果单独汇总
print("\n" + "=" * 100)
print("多专家 + 残差层叠 RC 模型结果汇总")
print("=" * 100)
print(f"{'层数':<10} {'专家数':<10} {'Accuracy':<12} {'F1 Score':<12} {'训练时间(秒)':<15}")
print("-" * 100)
for n_layers, n_experts in multi_expert_settings:
    res = multi_expert_results[(n_layers, n_experts)]
    time_val = multi_expert_train_times[(n_layers, n_experts)]
    print(f"{n_layers:<10} {n_experts:<10} {res['accuracy']:<12.4f} {res['f1']:<12.4f} {time_val:<15.2f}")

# 改进情况分析
print("\n" + "=" * 100)
print("改进情况分析（相对于原始RC模型）")
print("=" * 100)

for n_layers in [2, 3, 4, 5]:
    acc_val = stacked_results[n_layers]['accuracy']
    f1_val = stacked_results[n_layers]['f1']
    
    print(f"\n{n_layers}层层叠模型:")
    
    if acc_val > accuracy_original:
        acc_improvement = ((acc_val/accuracy_original-1)*100)
        print(f"  ✓ Accuracy: 提升了 {acc_improvement:+.2f}% ({acc_val:.4f} vs {accuracy_original:.4f})")
    else:
        acc_decrease = ((1-acc_val/accuracy_original)*100)
        print(f"  ✗ Accuracy: 降低了 {acc_decrease:+.2f}% ({acc_val:.4f} vs {accuracy_original:.4f})")
    
    if f1_val > f1_original:
        f1_improvement = ((f1_val/f1_original-1)*100)
        print(f"  ✓ F1 Score: 提升了 {f1_improvement:+.2f}% ({f1_val:.4f} vs {f1_original:.4f})")
    else:
        f1_decrease = ((1-f1_val/f1_original)*100)
        print(f"  ✗ F1 Score: 降低了 {f1_decrease:+.2f}% ({f1_val:.4f} vs {f1_original:.4f})")

# 找出串联层叠模型的最佳层数
best_acc_layer = max([2, 3, 4, 5], key=lambda x: stacked_results[x]['accuracy'])
best_f1_layer = max([2, 3, 4, 5], key=lambda x: stacked_results[x]['f1'])

print("\n" + "=" * 100)
print("串联层叠 RC 模型最佳配置")
print("=" * 100)
print(f"最佳Accuracy: {best_acc_layer}层 (Accuracy = {stacked_results[best_acc_layer]['accuracy']:.4f})")
print(f"最佳F1 Score: {best_f1_layer}层 (F1 = {stacked_results[best_f1_layer]['f1']:.4f})")

# 多专家层叠模型的最佳配置
best_me_acc_key = max(multi_expert_results.keys(), key=lambda k: multi_expert_results[k]['accuracy'])
best_me_f1_key = max(multi_expert_results.keys(), key=lambda k: multi_expert_results[k]['f1'])
best_me_acc = multi_expert_results[best_me_acc_key]
best_me_f1 = multi_expert_results[best_me_f1_key]

print("\n" + "=" * 100)
print("多专家 + 残差层叠 RC 模型最佳配置")
print("=" * 100)
print(f"最佳Accuracy: 层数 = {best_me_acc_key[0]}, 专家数 = {best_me_acc_key[1]} (Accuracy = {best_me_acc['accuracy']:.4f})")
print(f"最佳F1 Score: 层数 = {best_me_f1_key[0]}, 专家数 = {best_me_f1_key[1]} (F1 = {best_me_f1['f1']:.4f})")

# 综合对比：原始 RC vs 串联最佳 vs 多专家最佳
print("\n" + "=" * 100)
print("综合对比：原始 RC vs 最佳串联层叠 vs 最佳多专家层叠")
print("=" * 100)
print(f"{'模型':<25} {'Accuracy':<12} {'F1 Score':<12} {'训练时间(秒)':<15}")
print("-" * 80)
print(f"{'原始RC':<25} {accuracy_original:<12.4f} {f1_original:<12.4f} {train_time_original:<15.2f}")
print(f"{'串联最佳层叠RC':<25} {stacked_results[best_acc_layer]['accuracy']:<12.4f} {stacked_results[best_f1_layer]['f1']:<12.4f} {stacked_train_times[best_acc_layer]:<15.2f}")
print(f"{'多专家最佳层叠RC':<25} {best_me_acc['accuracy']:<12.4f} {best_me_f1['f1']:<12.4f} {multi_expert_train_times[best_me_acc_key]:<15.2f}")

# 层数趋势分析
print("\n" + "=" * 100)
print("串联层叠模型层数趋势分析")
print("=" * 100)
print(f"{'层数':<10} {'Accuracy':<12} {'Accuracy变化':<18} {'F1 Score':<12} {'F1变化':<18}")
print("-" * 100)
prev_acc = accuracy_original
prev_f1 = f1_original
for n_layers in [2, 3, 4, 5]:
    acc_val = stacked_results[n_layers]['accuracy']
    f1_val = stacked_results[n_layers]['f1']
    acc_change = acc_val - prev_acc
    f1_change = f1_val - prev_f1
    change_symbol_acc = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "→"
    change_symbol_f1 = "↑" if f1_change > 0 else "↓" if f1_change < 0 else "→"
    print(f"{n_layers:<10} {acc_val:<12.4f} {change_symbol_acc} {acc_change:+.4f} ({acc_change*100:+.2f}%){'':<8} {f1_val:<12.4f} {change_symbol_f1} {f1_change:+.4f} ({f1_change*100:+.2f}%)")
    prev_acc = acc_val
    prev_f1 = f1_val

print("\n实验完成！")
