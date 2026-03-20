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
import matplotlib.pyplot as plt
import umap
import os
import matplotlib.font_manager as fm

# 设置中文字体支持 - 尝试多种方式
# 方式1: 尝试使用系统默认中文字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 
                 'Source Han Sans CN', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS']

# 获取系统中可用的字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = None

for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
    print(f"使用字体: {selected_font}")
else:
    # 方式2: 使用matplotlib的默认字体，但避免中文
    print("警告: 未找到中文字体，将使用英文标签")

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
print("\n3. 训练层叠RC模型（6层）...")
stacked_models = {}
stacked_representations = {}

n_layers = 6
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
print("\n3b. 训练多专家 + 残差层叠RC模型（6层，专家数递增：5, 10, 15, 20）...")
multi_expert_models = {}
multi_expert_representations = {}

# 实验组合：固定6层，每层专家数分别为 5, 10, 15, 20
multi_expert_settings = [
    (6, 5),
    (6, 10),
    (6, 15),
    (6, 20),
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
print("\n3c. 训练MoE混合专家层叠RC模型（6层，专家数递增：5, 10, 15, 20）...")
moe_models = {}
moe_representations = {}

# 实验组合：固定6层，每层专家数分别为 5, 10, 15, 20
moe_settings = [
    (6, 5),
    (6, 10),
    (6, 15),
    (6, 20),
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

def visualize_with_umap(representations, true_labels, cluster_labels, model_name, save_dir='umap_visualizations'):
    """使用UMAP将高维表示降维到2D并可视化"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # UMAP降维
    print(f"\n   正在为{model_name}生成UMAP可视化...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(representations)
    
    # 检查是否支持中文
    use_chinese = selected_font is not None
    
    if use_chinese:
        # 中文标签
        title_true = f'{model_name} - 真实标签分布'
        title_cluster = f'{model_name} - 聚类结果分布'
        xlabel = 'UMAP维度1'
        ylabel = 'UMAP维度2'
        label_true = '类别标签'
        label_cluster = '聚类标签'
    else:
        # 英文标签（避免乱码）
        title_true = f'{model_name} - True Labels'
        title_cluster = f'{model_name} - Clustering Results'
        xlabel = 'UMAP Dimension 1'
        ylabel = 'UMAP Dimension 2'
        label_true = 'True Class'
        label_cluster = 'Cluster'
    
    # 创建图形，增加高度以确保标题和标签完整显示
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图：真实标签
    scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                               c=true_labels, cmap='tab10', s=30, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].set_title(title_true, fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel(xlabel, fontsize=12, labelpad=10)
    axes[0].set_ylabel(ylabel, fontsize=12, labelpad=10)
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], label=label_true, pad=0.02)
    cbar1.ax.tick_params(labelsize=10)
    
    # 右图：聚类结果
    scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], 
                               c=cluster_labels, cmap='tab10', s=30, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1].set_title(title_cluster, fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel(xlabel, fontsize=12, labelpad=10)
    axes[1].set_ylabel(ylabel, fontsize=12, labelpad=10)
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1], label=label_cluster, pad=0.02)
    cbar2.ax.tick_params(labelsize=10)
    
    # 使用更大的padding确保所有元素都显示完整
    plt.tight_layout(pad=3.0)
    
    # 保存图片（使用安全的文件名）
    safe_model_name = model_name.replace(' ', '_').replace('（', '_').replace('）', '').replace('，', '_').replace(',', '_')
    save_path = os.path.join(save_dir, f'umap_{safe_model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   已保存UMAP可视化图到: {save_path}")
    plt.close()
    
    return embedding

# 评估原始模型
true_labels = Y[:, 0]
nmi_original, ari_original, n_clust_original, clust_original = evaluate_clustering(
    mts_representations_original, true_labels, "原始RC模型"
)
# UMAP可视化
visualize_with_umap(mts_representations_original, true_labels, clust_original, "原始RC模型")

# 评估层叠模型（6层）
stacked_results = {}
n_layers = 6
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
# UMAP可视化
visualize_with_umap(stacked_representations[n_layers], true_labels, clust, f"层叠RC模型_{n_layers}层")

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
    # UMAP可视化
    visualize_with_umap(
        multi_expert_representations[(n_layers, n_experts)],
        true_labels,
        clust,
        f"多专家层叠RC模型_{n_layers}层_{n_experts}专家"
    )

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
    # UMAP可视化
    visualize_with_umap(
        moe_representations[(n_layers, n_experts)],
        true_labels,
        clust,
        f"MoE层叠RC模型_{n_layers}层_{n_experts}专家"
    )

# ============ 结果对比 ============
print("\n" + "=" * 80)
print("结果对比总结（原始 RC 与串联层叠 RC）")
print("=" * 80)

# 表头（仅对比原始RC和6层层叠RC）
header = f"{'指标':<15} {'原始RC':<12} {'6层':<12}"
print(f"\n{header}")
print("-" * 60)

# NMI对比
nmi_row = f"{'NMI':<15} {nmi_original:<12.4f} {stacked_results[6]['nmi']:<12.4f}"
print(nmi_row)

# ARI对比
ari_row = f"{'ARI':<15} {ari_original:<12.4f} {stacked_results[6]['ari']:<12.4f}"
print(ari_row)

# 聚类数对比
clust_row = f"{'聚类数':<15} {n_clust_original:<12} {stacked_results[6]['n_clust']:<12}"
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

n_layers = 6
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

print("\n" + "=" * 100)
print("串联层叠 RC 模型配置")
print("=" * 100)
print(f"6层 (NMI = {stacked_results[6]['nmi']:.4f}, ARI = {stacked_results[6]['ari']:.4f})")

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

# 综合对比：原始 RC vs 串联6层 vs 多专家最佳 vs MoE最佳
print("\n" + "=" * 100)
print("综合对比：四种模型最佳配置")
print("=" * 100)
print(f"{'模型':<25} {'NMI':<12} {'ARI':<12}")
print("-" * 60)
print(f"{'原始RC':<25} {nmi_original:<12.4f} {ari_original:<12.4f}")
print(f"{'串联6层层叠RC':<25} {stacked_results[6]['nmi']:<12.4f} {stacked_results[6]['ari']:<12.4f}")
print(f"{'多专家最佳层叠RC':<25} {best_me_nmi['nmi']:<12.4f} {best_me_ari['ari']:<12.4f}")
print(f"{'MoE最佳层叠RC':<25} {best_moe_nmi['nmi']:<12.4f} {best_moe_ari['ari']:<12.4f}")

# 专家数趋势分析（6层，专家数递增）
print("\n" + "=" * 100)
print("专家数趋势分析（6层，专家数递增：5, 10, 15, 20）")
print("=" * 100)
print(f"{'专家数':<10} {'NMI':<12} {'NMI变化':<15} {'ARI':<12} {'ARI变化':<15}")
print("-" * 100)
prev_nmi = stacked_results[6]['nmi']
prev_ari = stacked_results[6]['ari']
for n_experts in [5, 10, 15, 20]:
    if (6, n_experts) in multi_expert_results:
        nmi_val = multi_expert_results[(6, n_experts)]['nmi']
        ari_val = multi_expert_results[(6, n_experts)]['ari']
        nmi_change = nmi_val - prev_nmi
        ari_change = ari_val - prev_ari
        change_symbol_nmi = "↑" if nmi_change > 0 else "↓" if nmi_change < 0 else "→"
        change_symbol_ari = "↑" if ari_change > 0 else "↓" if ari_change < 0 else "→"
        print(f"{n_experts:<10} {nmi_val:<12.4f} {change_symbol_nmi} {nmi_change:+.4f} ({nmi_change*100:+.2f}%){'':<5} {ari_val:<12.4f} {change_symbol_ari} {ari_change:+.4f} ({ari_change*100:+.2f}%)")
        prev_nmi = nmi_val
        prev_ari = ari_val

# ============ 四模型综合排名 ============
print("\n" + "=" * 100)
print("四模型综合排名（按NMI和ARI）")
print("=" * 100)

# 收集所有模型的最佳结果
all_best_results = [
    ('原始RC', nmi_original, ari_original, '单层'),
    (f'串联层叠RC (6层)', stacked_results[6]['nmi'], stacked_results[6]['ari'], '6层'),
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

# ============ 综合对比可视化 ============
print("\n" + "=" * 100)
print("生成综合对比UMAP可视化...")
print("=" * 100)

def create_comparison_visualization():
    """创建四种模型最佳配置的综合对比可视化"""
    os.makedirs('umap_visualizations', exist_ok=True)
    
    # 检查是否支持中文
    use_chinese = selected_font is not None
    
    if use_chinese:
        # 中文模型名称
        models_data = [
            ('原始RC', mts_representations_original, clust_original, nmi_original, ari_original),
            ('串联6层层叠RC', stacked_representations[6], stacked_results[6]['clust'], 
             stacked_results[6]['nmi'], stacked_results[6]['ari']),
            (f'多专家层叠RC({best_me_nmi_key[0]}层,{best_me_nmi_key[1]}专家)', 
             multi_expert_representations[best_me_nmi_key], 
             multi_expert_results[best_me_nmi_key]['clust'],
             best_me_nmi['nmi'], best_me_ari['ari']),
            (f'MoE层叠RC({best_moe_nmi_key[0]}层,{best_moe_nmi_key[1]}专家)',
             moe_representations[best_moe_nmi_key],
             moe_results[best_moe_nmi_key]['clust'],
             best_moe_nmi['nmi'], best_moe_ari['ari'])
        ]
        title_true_suffix = '真实标签分布'
        title_cluster_suffix = '聚类结果'
        xlabel = 'UMAP维度1'
        ylabel = 'UMAP维度2'
        label_true = '类别'
        label_cluster = '聚类'
    else:
        # 英文模型名称（避免乱码）
        models_data = [
            ('Original RC', mts_representations_original, clust_original, nmi_original, ari_original),
            ('Stacked RC (6 layers)', stacked_representations[6], stacked_results[6]['clust'], 
             stacked_results[6]['nmi'], stacked_results[6]['ari']),
            (f'Multi-Expert RC ({best_me_nmi_key[0]}L,{best_me_nmi_key[1]}E)', 
             multi_expert_representations[best_me_nmi_key], 
             multi_expert_results[best_me_nmi_key]['clust'],
             best_me_nmi['nmi'], best_me_ari['ari']),
            (f'MoE RC ({best_moe_nmi_key[0]}L,{best_moe_nmi_key[1]}E)',
             moe_representations[best_moe_nmi_key],
             moe_results[best_moe_nmi_key]['clust'],
             best_moe_nmi['nmi'], best_moe_ari['ari'])
        ]
        title_true_suffix = 'True Labels'
        title_cluster_suffix = 'Clustering Results'
        xlabel = 'UMAP Dimension 1'
        ylabel = 'UMAP Dimension 2'
        label_true = 'True Class'
        label_cluster = 'Cluster'
    
    # 创建2x4的子图（每行2个模型，共4个模型，每个模型显示真实标签和聚类结果）
    # 增加图片尺寸和调整间距
    fig, axes = plt.subplots(4, 2, figsize=(18, 28))
    
    for idx, (model_name, representations, cluster_labels, nmi, ari) in enumerate(models_data):
        # UMAP降维
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(representations)
        
        # 左图：真实标签
        scatter1 = axes[idx, 0].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=true_labels, cmap='tab10', s=30, alpha=0.6, 
                                        edgecolors='k', linewidths=0.5)
        axes[idx, 0].set_title(f'{model_name}\n{title_true_suffix}', fontsize=13, fontweight='bold', pad=12)
        axes[idx, 0].set_xlabel(xlabel, fontsize=11, labelpad=8)
        axes[idx, 0].set_ylabel(ylabel, fontsize=11, labelpad=8)
        axes[idx, 0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes[idx, 0], label=label_true, pad=0.02)
        cbar1.ax.tick_params(labelsize=9)
        
        # 右图：聚类结果
        scatter2 = axes[idx, 1].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=cluster_labels, cmap='tab10', s=30, alpha=0.6, 
                                        edgecolors='k', linewidths=0.5)
        # 缩短标题以避免显示不全
        title_text = f'{model_name}\n{title_cluster_suffix}\nNMI={nmi:.4f}, ARI={ari:.4f}'
        axes[idx, 1].set_title(title_text, fontsize=13, fontweight='bold', pad=12)
        axes[idx, 1].set_xlabel(xlabel, fontsize=11, labelpad=8)
        axes[idx, 1].set_ylabel(ylabel, fontsize=11, labelpad=8)
        axes[idx, 1].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[idx, 1], label=label_cluster, pad=0.02)
        cbar2.ax.tick_params(labelsize=9)
    
    # 使用更大的padding确保所有元素都显示完整
    plt.tight_layout(pad=4.0, h_pad=2.0, w_pad=2.0)
    save_path = os.path.join('umap_visualizations', 'comparison_all_models.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存综合对比可视化图到: {save_path}")
    plt.close()

create_comparison_visualization()

print("\n实验完成！所有UMAP可视化图已保存到 umap_visualizations/ 目录")
