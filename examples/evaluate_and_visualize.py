"""
加载已训练的模型表示向量，进行聚类评估和可视化

本脚本从保存的文件中加载所有模型的表示向量，然后：
1. 进行聚类评估（NMI, ARI）
2. 生成UMAP可视化图
3. 输出对比分析结果
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import v_measure_score, adjusted_rand_score
import matplotlib.pyplot as plt
import umap
import os
import pickle

# 设置中文字体支持
import matplotlib
# 尝试多种中文字体，优先使用系统可用的字体
try:
    # macOS 系统字体
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保字体设置生效
matplotlib.rcParams.update({'font.size': 10})

# 设置随机种子以确保可重复性
np.random.seed(0)

print("=" * 60)
print("加载模型表示向量并进行聚类评估与可视化")
print("=" * 60)

# 加载保存的数据
save_dir = 'saved_representations'

if not os.path.exists(save_dir):
    print(f"\n错误：找不到保存目录 {save_dir}")
    print("请先运行 train_models.py 训练模型并保存表示向量")
    exit(1)

print("\n1. 加载保存的表示向量...")

# 加载真实标签
true_labels = np.load(os.path.join(save_dir, 'true_labels.npy'))
print(f"   已加载真实标签: {true_labels.shape}")

# 加载原始RC模型表示
mts_representations_original = np.load(os.path.join(save_dir, 'original_rc_representation.npy'))
print(f"   已加载原始RC模型表示: {mts_representations_original.shape}")

# 加载层叠RC模型表示（6层）
n_layers = 6
mts_representations_stacked = np.load(os.path.join(save_dir, f'stacked_rc_{n_layers}layers_representation.npy'))
print(f"   已加载层叠RC模型表示（{n_layers}层）: {mts_representations_stacked.shape}")

# 加载多专家模型配置和表示
with open(os.path.join(save_dir, 'multi_expert_settings.pkl'), 'rb') as f:
    multi_expert_settings = pickle.load(f)

multi_expert_representations = {}
for n_layers, n_experts in multi_expert_settings:
    filename = f'multi_expert_{n_layers}layers_{n_experts}experts_representation.npy'
    multi_expert_representations[(n_layers, n_experts)] = np.load(os.path.join(save_dir, filename))
    print(f"   已加载多专家模型表示（{n_layers}层, {n_experts}专家）: {multi_expert_representations[(n_layers, n_experts)].shape}")

# 加载MoE模型配置和表示
with open(os.path.join(save_dir, 'moe_settings.pkl'), 'rb') as f:
    moe_settings = pickle.load(f)

moe_representations = {}
for n_layers, n_experts in moe_settings:
    filename = f'moe_{n_layers}layers_{n_experts}experts_representation.npy'
    moe_representations[(n_layers, n_experts)] = np.load(os.path.join(save_dir, filename))
    print(f"   已加载MoE模型表示（{n_layers}层, {n_experts}专家）: {moe_representations[(n_layers, n_experts)].shape}")

# ============ 聚类评估 ============
print("\n2. 进行聚类评估...")

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
    
    # 创建图形，增加宽度和高度以确保标题和标签完整显示
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：真实标签
    scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                               c=true_labels, cmap='tab10', s=30, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].set_title(f'{model_name} - 真实标签分布', fontsize=14, fontweight='bold', pad=15)
    # 大幅增加 labelpad 确保轴标签完整显示
    axes[0].set_xlabel('UMAP维度1', fontsize=12, labelpad=15)
    axes[0].set_ylabel('UMAP维度2', fontsize=12, labelpad=15)
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], label='类别标签', pad=0.05)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.set_label('类别标签', fontsize=11)
    
    # 右图：聚类结果
    scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], 
                               c=cluster_labels, cmap='tab10', s=30, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1].set_title(f'{model_name} - 聚类结果分布', fontsize=14, fontweight='bold', pad=15)
    # 大幅增加 labelpad 确保轴标签完整显示
    axes[1].set_xlabel('UMAP维度1', fontsize=12, labelpad=15)
    axes[1].set_ylabel('UMAP维度2', fontsize=12, labelpad=15)
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1], label='聚类标签', pad=0.05)
    cbar2.ax.tick_params(labelsize=10)
    cbar2.set_label('聚类标签', fontsize=11)
    
    # 使用 subplots_adjust 手动调整布局，确保轴标签完整显示
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.92, wspace=0.25)
    
    # 保存图片（使用安全的文件名）
    safe_model_name = model_name.replace(' ', '_').replace('（', '_').replace('）', '').replace('，', '_').replace(',', '_')
    save_path = os.path.join(save_dir, f'umap_{safe_model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    print(f"   已保存UMAP可视化图到: {save_path}")
    plt.close()
    
    return embedding

# 评估原始模型
nmi_original, ari_original, n_clust_original, clust_original = evaluate_clustering(
    mts_representations_original, true_labels, "原始RC模型"
)
# UMAP可视化
visualize_with_umap(mts_representations_original, true_labels, clust_original, "原始RC模型")

# 评估层叠模型（6层）
stacked_results = {}
nmi, ari, n_clust, clust = evaluate_clustering(
    mts_representations_stacked, 
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
visualize_with_umap(mts_representations_stacked, true_labels, clust, f"层叠RC模型_{n_layers}层")

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
    
    # 准备数据
    models_data = [
        ('原始RC', mts_representations_original, clust_original, nmi_original, ari_original),
        ('串联6层层叠RC', mts_representations_stacked, stacked_results[6]['clust'], 
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
    
    # 创建2x4的子图（每行2个模型，共4个模型，每个模型显示真实标签和聚类结果）
    # 增加图片宽度，给轴标签更多空间，并增加高度
    fig, axes = plt.subplots(4, 2, figsize=(20, 30))
    
    for idx, (model_name, representations, cluster_labels, nmi, ari) in enumerate(models_data):
        # UMAP降维
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(representations)
        
        # 左图：真实标签
        scatter1 = axes[idx, 0].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=true_labels, cmap='tab10', s=30, alpha=0.6, 
                                        edgecolors='k', linewidths=0.5)
        axes[idx, 0].set_title(f'{model_name}\n真实标签分布', fontsize=13, fontweight='bold', pad=15)
        # 大幅增加 labelpad 确保轴标签完整显示
        axes[idx, 0].set_xlabel('UMAP维度1', fontsize=12, labelpad=15)
        axes[idx, 0].set_ylabel('UMAP维度2', fontsize=12, labelpad=15)
        axes[idx, 0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=axes[idx, 0], label='类别', pad=0.05)
        cbar1.ax.tick_params(labelsize=9)
        # 设置 colorbar 标签字体
        cbar1.set_label('类别', fontsize=10)
        
        # 右图：聚类结果
        scatter2 = axes[idx, 1].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=cluster_labels, cmap='tab10', s=30, alpha=0.6, 
                                        edgecolors='k', linewidths=0.5)
        # 缩短标题以避免显示不全
        title_text = f'{model_name}\n聚类结果\nNMI={nmi:.4f}, ARI={ari:.4f}'
        axes[idx, 1].set_title(title_text, fontsize=13, fontweight='bold', pad=15)
        # 大幅增加 labelpad 确保轴标签完整显示
        axes[idx, 1].set_xlabel('UMAP维度1', fontsize=12, labelpad=15)
        axes[idx, 1].set_ylabel('UMAP维度2', fontsize=12, labelpad=15)
        axes[idx, 1].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=axes[idx, 1], label='聚类', pad=0.05)
        cbar2.ax.tick_params(labelsize=9)
        # 设置 colorbar 标签字体
        cbar2.set_label('聚类', fontsize=10)
    
    # 使用 subplots_adjust 手动调整布局，确保轴标签完整显示
    # 增加底部和左侧边距，为轴标签留出空间
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.95, top=0.97, hspace=0.35, wspace=0.25)
    
    save_path = os.path.join('umap_visualizations', 'comparison_all_models.png')
    # 使用更大的 bbox_inches 确保所有内容都保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    print(f"已保存综合对比可视化图到: {save_path}")
    plt.close()

create_comparison_visualization()

print("\n实验完成！所有UMAP可视化图已保存到 umap_visualizations/ 目录")
