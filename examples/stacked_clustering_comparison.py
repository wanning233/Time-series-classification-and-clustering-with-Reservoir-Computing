"""
层叠Reservoir与原始Reservoir的聚类效果对比实验

本脚本对比了原始RC_model和层叠StackedRC_model在聚类任务上的表现。
使用相同的聚类方法和评估指标，便于直接对比两种模型的聚类效果。
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import v_measure_score, adjusted_rand_score

from reservoir_computing.modules import RC_model, StackedRC_model
from reservoir_computing.datasets import ClfLoader

# 设置随机种子以确保可重复性
np.random.seed(0)

print("=" * 60)
print("层叠Reservoir与原始Reservoir聚类效果对比实验")
print("=" * 60)

# 加载数据
print("\n1. 加载数据...")
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# 由于进行聚类，不需要训练/测试分割
X = np.concatenate((Xtr, Xte), axis=0)
Y = np.concatenate((Ytr, Yte), axis=0)

print(f"   数据形状: {X.shape}")
print(f"   类别数量: {len(np.unique(Y[:,0]))}")

# ============ 原始RC模型 ============
print("\n2. 训练原始RC模型...")
rcm_original = RC_model(
    n_internal_units=400,
    readout_type=None  # 设置为None以存储输入表示
)

rcm_original.fit(X, verbose=True)
mts_representations_original = rcm_original.input_repr
print(f"   表示维度: {mts_representations_original.shape}")

# ============ 层叠RC模型 ============
print("\n3. 训练层叠RC模型（2层）...")
rcm_stacked = StackedRC_model(
    n_layers=2,
    reservoir_configs=None,  # 使用默认渐进式配置
    readout_type=None  # 设置为None以存储输入表示
)

rcm_stacked.fit(X, verbose=True)
mts_representations_stacked = rcm_stacked.input_repr
print(f"   表示维度: {mts_representations_stacked.shape}")

# ============ 聚类评估 ============
print("\n4. 进行聚类评估...")

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

# 评估原始模型
true_labels = Y[:, 0]
nmi_original, ari_original, n_clust_original, clust_original = evaluate_clustering(
    mts_representations_original, true_labels, "原始RC模型"
)

# 评估层叠模型
nmi_stacked, ari_stacked, n_clust_stacked, clust_stacked = evaluate_clustering(
    mts_representations_stacked, true_labels, "层叠RC模型"
)

# ============ 结果对比 ============
print("\n" + "=" * 60)
print("结果对比总结")
print("=" * 60)
print(f"\n{'指标':<20} {'原始RC模型':<15} {'层叠RC模型':<15} {'改进':<10}")
print("-" * 60)
print(f"{'NMI':<20} {nmi_original:<15.4f} {nmi_stacked:<15.4f} {nmi_stacked-nmi_original:>+8.4f}")
print(f"{'ARI':<20} {ari_original:<15.4f} {ari_stacked:<15.4f} {ari_stacked-ari_original:>+8.4f}")
print(f"{'聚类数':<20} {n_clust_original:<15} {n_clust_stacked:<15} {'-':>10}")

# 判断改进情况
if nmi_stacked > nmi_original:
    print(f"\n✓ 层叠模型在NMI指标上提升了 {((nmi_stacked/nmi_original-1)*100):.2f}%")
else:
    print(f"\n✗ 层叠模型在NMI指标上降低了 {((1-nmi_stacked/nmi_original)*100):.2f}%")

if ari_stacked > ari_original:
    print(f"✓ 层叠模型在ARI指标上提升了 {((ari_stacked/ari_original-1)*100):.2f}%")
else:
    print(f"✗ 层叠模型在ARI指标上降低了 {((1-ari_stacked/ari_original)*100):.2f}%")

print("\n实验完成！")
