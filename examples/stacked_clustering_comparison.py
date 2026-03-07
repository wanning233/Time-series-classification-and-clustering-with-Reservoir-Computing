"""
四种Reservoir计算模型聚类效果对比实验

本脚本对比了以下四种模型在聚类任务上的表现：
1. RC_model: 原始单层Reservoir模型
2. StackedRC_model: 层叠Reservoir模型
3. MultiExpertStackedRC_model: 多专家+残差式层叠Reservoir模型
4. MoEStackedRC_model: 混合专家（MoE）层叠Reservoir模型

使用相同的聚类方法和评估指标，便于直接对比四种模型的聚类效果。
"""

import numpy as np
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
np.random.seed(0)

print("=" * 60)
print("四种Reservoir计算模型聚类效果对比实验")
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
print("\n3. 训练层叠RC模型（2、3、4、5、6层）...")
stacked_models = {}
stacked_representations = {}

for n_layers in [2, 3, 4, 5, 6]:
    print(f"\n   训练 {n_layers} 层层叠模型...")
    rcm_stacked = StackedRC_model(
        n_layers=n_layers,
        reservoir_configs=None,  # 使用默认渐进式配置
        readout_type=None  # 设置为None以存储输入表示
    )
    
    rcm_stacked.fit(X, verbose=False)
    mts_representations_stacked = rcm_stacked.input_repr
    print(f"   表示维度: {mts_representations_stacked.shape}")
    
    stacked_models[n_layers] = rcm_stacked
    stacked_representations[n_layers] = mts_representations_stacked


# ============ 多专家 + 残差层叠RC模型 ============
print("\n3b. 训练多专家 + 残差层叠RC模型（多种层数与专家数组合）...")
multi_expert_models = {}
multi_expert_representations = {}

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
    rcm_me = MultiExpertStackedRC_model(
        n_layers=n_layers,
        n_experts=n_experts,
        reservoir_configs=None,  # 使用默认渐进式配置
        mts_rep='mean',
        readout_type=None  # 仅存储输入表示用于聚类
    )

    rcm_me.fit(X, verbose=False)
    mts_repr_me = rcm_me.input_repr
    print(f"   表示维度: {mts_repr_me.shape}")

    multi_expert_models[(n_layers, n_experts)] = rcm_me
    multi_expert_representations[(n_layers, n_experts)] = mts_repr_me

# ============ MoE混合专家层叠RC模型 ============
print("\n3c. 训练MoE混合专家层叠RC模型（多种层数与专家数组合）...")
moe_models = {}
moe_representations = {}

# 实验组合：[(层数, 专家数), ...]
moe_settings = [
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
]

for n_layers, n_experts in moe_settings:
    print(f"\n   训练 MoE {n_layers} 层层叠模型（n_experts={n_experts}）...")
    rcm_moe = MoEStackedRC_model(
        n_layers=n_layers,
        n_experts=n_experts,
        reservoir_configs=None,  # 使用默认渐进式配置
        gate_lr=0.01,
        gate_epochs=100,  # 门控网络训练轮数
        gate_reg=1e-4,
        intra_gate_input='mean',
        readout_type=None,  # 仅存储输入表示用于聚类
        mts_rep='mean'
    )

    rcm_moe.fit(X, Y=None, verbose=False)
    mts_repr_moe = rcm_moe.input_repr
    print(f"   表示维度: {mts_repr_moe.shape}")

    moe_models[(n_layers, n_experts)] = rcm_moe
    moe_representations[(n_layers, n_experts)] = mts_repr_moe

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

# 评估各层层叠模型
stacked_results = {}
for n_layers in [2, 3, 4, 5, 6]:
    nmi, ari, n_clust, clust = evaluate_clustering(
        stacked_representations[n_layers], 
        true_labels, 
        f"层叠RC模型（{n_layers}层）"
    )
    stacked_results[n_layers] = {
        'nmi': nmi,
        'ari': ari,
        'n_clust': n_clust,
        'clust': clust
    }

# 评估多专家层叠模型
multi_expert_results = {}
for n_layers, n_experts in multi_expert_settings:
    nmi, ari, n_clust, clust = evaluate_clustering(
        multi_expert_representations[(n_layers, n_experts)],
        true_labels,
        f"多专家层叠RC模型（{n_layers}层, {n_experts}专家）"
    )
    multi_expert_results[(n_layers, n_experts)] = {
        'nmi': nmi,
        'ari': ari,
        'n_clust': n_clust,
        'clust': clust
    }

# 评估MoE层叠模型
moe_results = {}
for n_layers, n_experts in moe_settings:
    nmi, ari, n_clust, clust = evaluate_clustering(
        moe_representations[(n_layers, n_experts)],
        true_labels,
        f"MoE层叠RC模型（{n_layers}层, {n_experts}专家）"
    )
    moe_results[(n_layers, n_experts)] = {
        'nmi': nmi,
        'ari': ari,
        'n_clust': n_clust,
        'clust': clust
    }

# ============ 结果对比 ============
print("\n" + "=" * 80)
print("结果对比总结（原始 RC 与串联层叠 RC）")
print("=" * 80)

# 表头（暂不把多专家列进来，避免过宽，仅对比 baseline 串联）
header = f"{'指标':<15} {'原始RC':<12} {'2层':<12} {'3层':<12} {'4层':<12} {'5层':<12} {'6层':<12}"
print(f"\n{header}")
print("-" * 100)

# NMI对比
nmi_row = f"{'NMI':<15} {nmi_original:<12.4f}"
for n_layers in [2, 3, 4, 5, 6]:
    nmi_val = stacked_results[n_layers]['nmi']
    nmi_row += f" {nmi_val:<11.4f}"
print(nmi_row)

# ARI对比
ari_row = f"{'ARI':<15} {ari_original:<12.4f}"
for n_layers in [2, 3, 4, 5, 6]:
    ari_val = stacked_results[n_layers]['ari']
    ari_row += f" {ari_val:<11.4f}"
print(ari_row)

# 聚类数对比
clust_row = f"{'聚类数':<15} {n_clust_original:<12}"
for n_layers in [2, 3, 4, 5, 6]:
    n_clust_val = stacked_results[n_layers]['n_clust']
    clust_row += f" {n_clust_val:<12}"
print(clust_row)

# 多专家层叠模型结果单独汇总
print("\n" + "=" * 80)
print("多专家 + 残差层叠 RC 模型结果汇总")
print("=" * 80)
print(f"{'层数':<10} {'专家数':<10} {'NMI':<12} {'ARI':<12} {'聚类数':<10}")
print("-" * 80)
for n_layers, n_experts in multi_expert_settings:
    res = multi_expert_results[(n_layers, n_experts)]
    print(f"{n_layers:<10} {n_experts:<10} {res['nmi']:<12.4f} {res['ari']:<12.4f} {res['n_clust']:<10}")

# MoE层叠模型结果单独汇总
print("\n" + "=" * 80)
print("MoE混合专家层叠 RC 模型结果汇总")
print("=" * 80)
print(f"{'层数':<10} {'专家数':<10} {'NMI':<12} {'ARI':<12} {'聚类数':<10}")
print("-" * 80)
for n_layers, n_experts in moe_settings:
    res = moe_results[(n_layers, n_experts)]
    print(f"{n_layers:<10} {n_experts:<10} {res['nmi']:<12.4f} {res['ari']:<12.4f} {res['n_clust']:<10}")

# 改进情况分析
print("\n" + "=" * 100)
print("改进情况分析（相对于原始RC模型）")
print("=" * 100)

for n_layers in [2, 3, 4, 5, 6]:
    nmi_val = stacked_results[n_layers]['nmi']
    ari_val = stacked_results[n_layers]['ari']
    
    print(f"\n{n_layers}层层叠模型:")
    
    if nmi_val > nmi_original:
        nmi_improvement = ((nmi_val/nmi_original-1)*100)
        print(f"  ✓ NMI: 提升了 {nmi_improvement:+.2f}% ({nmi_val:.4f} vs {nmi_original:.4f})")
    else:
        nmi_decrease = ((1-nmi_val/nmi_original)*100)
        print(f"  ✗ NMI: 降低了 {nmi_decrease:+.2f}% ({nmi_val:.4f} vs {nmi_original:.4f})")
    
    if ari_val > ari_original:
        ari_improvement = ((ari_val/ari_original-1)*100)
        print(f"  ✓ ARI: 提升了 {ari_improvement:+.2f}% ({ari_val:.4f} vs {ari_original:.4f})")
    else:
        ari_decrease = ((1-ari_val/ari_original)*100)
        print(f"  ✗ ARI: 降低了 {ari_decrease:+.2f}% ({ari_val:.4f} vs {ari_original:.4f})")

# 找出串联层叠模型的最佳层数
best_nmi_layer = max([2, 3, 4, 5, 6], key=lambda x: stacked_results[x]['nmi'])
best_ari_layer = max([2, 3, 4, 5, 6], key=lambda x: stacked_results[x]['ari'])

print("\n" + "=" * 100)
print("串联层叠 RC 模型最佳配置")
print("=" * 100)
print(f"最佳NMI: {best_nmi_layer}层 (NMI = {stacked_results[best_nmi_layer]['nmi']:.4f})")
print(f"最佳ARI: {best_ari_layer}层 (ARI = {stacked_results[best_ari_layer]['ari']:.4f})")

# 多专家层叠模型的最佳配置
best_me_nmi_key = max(multi_expert_results.keys(), key=lambda k: multi_expert_results[k]['nmi'])
best_me_ari_key = max(multi_expert_results.keys(), key=lambda k: multi_expert_results[k]['ari'])
best_me_nmi = multi_expert_results[best_me_nmi_key]
best_me_ari = multi_expert_results[best_me_ari_key]

print("\n" + "=" * 100)
print("多专家 + 残差层叠 RC 模型最佳配置")
print("=" * 100)
print(f"最佳NMI: 层数 = {best_me_nmi_key[0]}, 专家数 = {best_me_nmi_key[1]} (NMI = {best_me_nmi['nmi']:.4f})")
print(f"最佳ARI: 层数 = {best_me_ari_key[0]}, 专家数 = {best_me_ari_key[1]} (ARI = {best_me_ari['ari']:.4f})")

# MoE层叠模型的最佳配置
best_moe_nmi_key = max(moe_results.keys(), key=lambda k: moe_results[k]['nmi'])
best_moe_ari_key = max(moe_results.keys(), key=lambda k: moe_results[k]['ari'])
best_moe_nmi = moe_results[best_moe_nmi_key]
best_moe_ari = moe_results[best_moe_ari_key]

print("\n" + "=" * 100)
print("MoE混合专家层叠 RC 模型最佳配置")
print("=" * 100)
print(f"最佳NMI: 层数 = {best_moe_nmi_key[0]}, 专家数 = {best_moe_nmi_key[1]} (NMI = {best_moe_nmi['nmi']:.4f})")
print(f"最佳ARI: 层数 = {best_moe_ari_key[0]}, 专家数 = {best_moe_ari_key[1]} (ARI = {best_moe_ari['ari']:.4f})")

# 综合对比：原始 RC vs 串联最佳 vs 多专家最佳 vs MoE最佳
print("\n" + "=" * 100)
print("综合对比：四种模型最佳配置")
print("=" * 100)
print(f"{'模型':<25} {'NMI':<12} {'ARI':<12}")
print("-" * 60)
print(f"{'原始RC':<25} {nmi_original:<12.4f} {ari_original:<12.4f}")
print(f"{'串联最佳层叠RC':<25} {stacked_results[best_nmi_layer]['nmi']:<12.4f} {stacked_results[best_ari_layer]['ari']:<12.4f}")
print(f"{'多专家最佳层叠RC':<25} {best_me_nmi['nmi']:<12.4f} {best_me_ari['ari']:<12.4f}")
print(f"{'MoE最佳层叠RC':<25} {best_moe_nmi['nmi']:<12.4f} {best_moe_ari['ari']:<12.4f}")

# 层数趋势分析
print("\n" + "=" * 100)
print("层数趋势分析")
print("=" * 100)
print(f"{'层数':<10} {'NMI':<12} {'NMI变化':<15} {'ARI':<12} {'ARI变化':<15}")
print("-" * 100)
prev_nmi = nmi_original
prev_ari = ari_original
for n_layers in [2, 3, 4, 5, 6]:
    nmi_val = stacked_results[n_layers]['nmi']
    ari_val = stacked_results[n_layers]['ari']
    nmi_change = nmi_val - prev_nmi
    ari_change = ari_val - prev_ari
    change_symbol_nmi = "↑" if nmi_change > 0 else "↓" if nmi_change < 0 else "→"
    change_symbol_ari = "↑" if ari_change > 0 else "↓" if ari_change < 0 else "→"
    print(f"{n_layers:<10} {nmi_val:<12.4f} {change_symbol_nmi} {nmi_change:+.4f} ({nmi_change*100:+.2f}%){'':<5} {ari_val:<12.4f} {change_symbol_ari} {ari_change:+.4f} ({ari_change*100:+.2f}%)")
    prev_nmi = nmi_val
    prev_ari = ari_val

# ============ 四模型综合排名 ============
print("\n" + "=" * 100)
print("四模型综合排名（按NMI和ARI）")
print("=" * 100)

# 收集所有模型的最佳结果
all_best_results = [
    ('原始RC', nmi_original, ari_original, '单层'),
    (f'串联层叠RC ({best_nmi_layer}层)', stacked_results[best_nmi_layer]['nmi'], stacked_results[best_ari_layer]['ari'], f'{best_nmi_layer}层'),
    (f'多专家层叠RC ({best_me_nmi_key[0]}层,{best_me_nmi_key[1]}专家)', best_me_nmi['nmi'], best_me_ari['ari'], f'{best_me_nmi_key[0]}层{best_me_nmi_key[1]}专家'),
    (f'MoE层叠RC ({best_moe_nmi_key[0]}层,{best_moe_nmi_key[1]}专家)', best_moe_nmi['nmi'], best_moe_ari['ari'], f'{best_moe_nmi_key[0]}层{best_moe_nmi_key[1]}专家'),
]

# 按NMI排序
all_best_results_sorted_nmi = sorted(all_best_results, key=lambda x: x[1], reverse=True)
print("\n按NMI排名:")
print(f"{'排名':<6} {'模型':<35} {'NMI':<12} {'配置':<20}")
print("-" * 80)
for rank, (name, nmi, ari, config) in enumerate(all_best_results_sorted_nmi, 1):
    print(f"{rank:<6} {name:<35} {nmi:<12.4f} {config:<20}")

# 按ARI排序
all_best_results_sorted_ari = sorted(all_best_results, key=lambda x: x[2], reverse=True)
print("\n按ARI排名:")
print(f"{'排名':<6} {'模型':<35} {'ARI':<12} {'配置':<20}")
print("-" * 80)
for rank, (name, nmi, ari, config) in enumerate(all_best_results_sorted_ari, 1):
    print(f"{rank:<6} {name:<35} {ari:<12.4f} {config:<20}")

# 找出总体最佳模型
overall_best_nmi = max(all_best_results, key=lambda x: x[1])
overall_best_ari = max(all_best_results, key=lambda x: x[2])

print("\n" + "=" * 100)
print("总体最佳模型")
print("=" * 100)
print(f"最佳NMI: {overall_best_nmi[0]} (NMI = {overall_best_nmi[1]:.4f})")
print(f"最佳ARI: {overall_best_ari[0]} (ARI = {overall_best_ari[2]:.4f})")

print("\n实验完成！")
