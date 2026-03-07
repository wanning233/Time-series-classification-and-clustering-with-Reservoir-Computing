"""
四种Reservoir计算模型对比实验（聚类评估）

本脚本对比了以下四种模型在时间序列聚类任务上的表现：
1. RC_model: 原始单层Reservoir模型
2. StackedRC_model: 层叠Reservoir模型
3. MultiExpertStackedRC_model: 多专家+残差式层叠Reservoir模型
4. MoEStackedRC_model: 混合专家（MoE）层叠Reservoir模型

评估指标：
- 训练时间
- 标准化互信息 (NMI)
- 调整兰德指数 (ARI)
- 找到的聚类数
"""

import numpy as np
import time
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import v_measure_score, adjusted_rand_score

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
print("四种Reservoir计算模型对比实验（聚类评估）")
print("=" * 80)

# ============ 加载数据 ============
print("\n1. 加载数据...")
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# 由于进行聚类，不需要训练/测试分割，合并所有数据
X = np.concatenate((Xtr, Xte), axis=0)
Y = np.concatenate((Ytr, Yte), axis=0)
true_labels = Y[:, 0]

print(f"   数据形状: {X.shape}")
print(f"   类别数量: {len(np.unique(true_labels))}")

# ============ 聚类评估函数 ============
def evaluate_clustering(representations, true_labels, model_name):
    """评估聚类效果"""
    # 计算相似度矩阵
    Dist = cosine_distances(representations)
    distArray = ssd.squareform(Dist)
    
    # 层次聚类
    Z = linkage(distArray, 'ward')
    
    # 尝试不同的聚类数量
    n_clusters_true = len(np.unique(true_labels))
    clust = fcluster(Z, t=n_clusters_true, criterion="maxclust")
    
    # 评估指标
    nmi = v_measure_score(true_labels, clust)
    ari = adjusted_rand_score(true_labels, clust)
    n_clusters_found = len(np.unique(clust))
    
    print(f"\n   {model_name}:")
    print(f"   找到的聚类数: {n_clusters_found}")
    print(f"   标准化互信息 (NMI): {nmi:.4f}")
    print(f"   调整兰德指数 (ARI): {ari:.4f}")
    
    return nmi, ari, n_clusters_found, clust

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
    readout_type=None,  # 设置为None以存储输入表示用于聚类
    mts_rep='mean'
)

start_time = time.time()
model1.fit(X, verbose=True)
train_time1 = time.time() - start_time

repr1 = model1.input_repr
nmi1, ari1, n_clust1, clust1 = evaluate_clustering(
    repr1, true_labels, "RC_model"
)

results['RC_model'] = {
    'train_time': train_time1,
    'nmi': nmi1,
    'ari': ari1,
    'n_clusters': n_clust1,
    'representations': repr1,
    'model': model1
}

print(f"   训练时间: {train_time1:.2f} 秒")

# ============ 模型2: StackedRC_model (层叠Reservoir) ============
print("\n" + "=" * 80)
print("模型2: StackedRC_model (层叠Reservoir, 2层)")
print("=" * 80)

model2 = StackedRC_model(
    n_layers=2,
    reservoir_configs=None,  # 使用默认渐进式配置
    readout_type=None,  # 设置为None以存储输入表示用于聚类
    mts_rep='mean'
)

start_time = time.time()
model2.fit(X, verbose=True)
train_time2 = time.time() - start_time

repr2 = model2.input_repr
nmi2, ari2, n_clust2, clust2 = evaluate_clustering(
    repr2, true_labels, "StackedRC_model"
)

results['StackedRC_model'] = {
    'train_time': train_time2,
    'nmi': nmi2,
    'ari': ari2,
    'n_clusters': n_clust2,
    'representations': repr2,
    'model': model2
}

print(f"   训练时间: {train_time2:.2f} 秒")

# ============ 模型3: MultiExpertStackedRC_model (多专家+残差) ============
print("\n" + "=" * 80)
print("模型3: MultiExpertStackedRC_model (多专家+残差式层叠, 2层3专家)")
print("=" * 80)

model3 = MultiExpertStackedRC_model(
    n_layers=2,
    n_experts=3,
    reservoir_configs=None,  # 使用默认渐进式配置
    readout_type=None,  # 设置为None以存储输入表示用于聚类
    mts_rep='mean'
)

start_time = time.time()
model3.fit(X, verbose=True)
train_time3 = time.time() - start_time

repr3 = model3.input_repr
nmi3, ari3, n_clust3, clust3 = evaluate_clustering(
    repr3, true_labels, "MultiExpertStackedRC_model"
)

results['MultiExpertStackedRC_model'] = {
    'train_time': train_time3,
    'nmi': nmi3,
    'ari': ari3,
    'n_clusters': n_clust3,
    'representations': repr3,
    'model': model3
}

print(f"   训练时间: {train_time3:.2f} 秒")

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
    readout_type=None,  # 设置为None以存储输入表示用于聚类
    mts_rep='mean'
)

start_time = time.time()
model4.fit(X, Y=None, verbose=True)  # 聚类任务不需要标签
train_time4 = time.time() - start_time

repr4 = model4.input_repr
nmi4, ari4, n_clust4, clust4 = evaluate_clustering(
    repr4, true_labels, "MoEStackedRC_model"
)

results['MoEStackedRC_model'] = {
    'train_time': train_time4,
    'nmi': nmi4,
    'ari': ari4,
    'n_clusters': n_clust4,
    'representations': repr4,
    'model': model4
}

print(f"   训练时间: {train_time4:.2f} 秒")

# ============ 结果汇总 ============
print("\n" + "=" * 80)
print("实验结果汇总")
print("=" * 80)

print(f"\n{'模型':<35} {'训练时间(秒)':<15} {'NMI':<12} {'ARI':<12} {'聚类数':<10}")
print("-" * 100)

for model_name, result in results.items():
    print(f"{model_name:<35} {result['train_time']:<15.2f} {result['nmi']:<12.4f} {result['ari']:<12.4f} {result['n_clusters']:<10}")

# 找出最佳模型
best_nmi_model = max(results.items(), key=lambda x: x[1]['nmi'])
best_ari_model = max(results.items(), key=lambda x: x[1]['ari'])
fastest_model = min(results.items(), key=lambda x: x[1]['train_time'])

print("\n" + "-" * 100)
print(f"最佳NMI模型: {best_nmi_model[0]} (NMI: {best_nmi_model[1]['nmi']:.4f})")
print(f"最佳ARI模型: {best_ari_model[0]} (ARI: {best_ari_model[1]['ari']:.4f})")
print(f"最快训练模型: {fastest_model[0]} (时间: {fastest_model[1]['train_time']:.2f}秒)")

# ============ 性能提升分析 ============
print("\n" + "=" * 80)
print("性能提升分析（相对于RC_model基准）")
print("=" * 80)

baseline_nmi = results['RC_model']['nmi']
baseline_ari = results['RC_model']['ari']
baseline_time = results['RC_model']['train_time']

for model_name, result in results.items():
    if model_name == 'RC_model':
        continue
    nmi_improve = (result['nmi'] - baseline_nmi) / baseline_nmi * 100
    ari_improve = (result['ari'] - baseline_ari) / baseline_ari * 100
    time_ratio = result['train_time'] / baseline_time
    
    print(f"\n{model_name}:")
    if nmi_improve > 0:
        print(f"  ✓ NMI提升: {nmi_improve:+.2f}% ({result['nmi']:.4f} vs {baseline_nmi:.4f})")
    else:
        print(f"  ✗ NMI降低: {nmi_improve:+.2f}% ({result['nmi']:.4f} vs {baseline_nmi:.4f})")
    if ari_improve > 0:
        print(f"  ✓ ARI提升: {ari_improve:+.2f}% ({result['ari']:.4f} vs {baseline_ari:.4f})")
    else:
        print(f"  ✗ ARI降低: {ari_improve:+.2f}% ({result['ari']:.4f} vs {baseline_ari:.4f})")
    print(f"  训练时间倍数: {time_ratio:.2f}x")

# ============ MoE模型门控权重可视化（可选） ============
print("\n" + "=" * 80)
print("MoE模型门控权重分析（前5个样本）")
print("=" * 80)

try:
    intra_w, inter_w = model4.get_gate_weights(X[:5])
    print(f"\n层内门控权重（每层专家权重，形状: {[w.shape for w in intra_w]}）")
    for layer_idx, w in enumerate(intra_w):
        print(f"  层{layer_idx}专家权重（前3个样本）:")
        print(f"    {w[:3].round(4)}")
        print(f"    均值: {w.mean(axis=0).round(4)}")
        print(f"    说明: 均值显示各专家在该层的相对重要性")
    
    print(f"\n层间门控权重（各层权重，形状: {inter_w.shape}）")
    print(f"  前3个样本:")
    print(f"    {inter_w[:3].round(4)}")
    print(f"    均值: {inter_w.mean(axis=0).round(4)}")
    print(f"    说明: 均值显示各层的相对重要性（值越大表示该层贡献越大）")
except Exception as e:
    print(f"  无法获取门控权重: {e}")

print("\n" + "=" * 80)
print("实验完成！")
print("=" * 80)
