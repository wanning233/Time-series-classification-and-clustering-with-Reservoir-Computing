"""
层叠Reservoir超参数调优脚本

基于实验结果，5层表现最佳。本脚本针对5层架构进行超参数调优：
1. 调整单元数配置
2. 尝试不同表示方法
3. 调整谱半径
4. 尝试降维参数
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import v_measure_score, adjusted_rand_score

from reservoir_computing.modules import StackedRC_model
from reservoir_computing.datasets import ClfLoader

# 设置随机种子以确保可重复性
np.random.seed(0)

print("=" * 80)
print("层叠Reservoir超参数调优实验（基于5层架构）")
print("=" * 80)

# 加载数据
print("\n1. 加载数据...")
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')
X = np.concatenate((Xtr, Xte), axis=0)
Y = np.concatenate((Ytr, Yte), axis=0)
true_labels = Y[:, 0]

print(f"   数据形状: {X.shape}")
print(f"   类别数量: {len(np.unique(true_labels))}")

# 基准：默认5层配置
print("\n2. 基准测试：默认5层配置...")
baseline_model = StackedRC_model(
    n_layers=5,
    reservoir_configs=None,  # 默认配置：[400, 350, 300, 200, 150]
    readout_type=None
)
baseline_model.fit(X, verbose=False)
baseline_repr = baseline_model.input_repr

def evaluate_clustering(representations, true_labels):
    """评估聚类效果"""
    Dist = cosine_distances(representations)
    distArray = ssd.squareform(Dist)
    Z = linkage(distArray, 'ward')
    n_clusters_true = len(np.unique(true_labels))
    clust = fcluster(Z, t=n_clusters_true, criterion="maxclust")
    nmi = v_measure_score(true_labels, clust)
    ari = adjusted_rand_score(true_labels, clust)
    return nmi, ari, clust

baseline_nmi, baseline_ari, _ = evaluate_clustering(baseline_repr, true_labels)
print(f"   基准NMI: {baseline_nmi:.4f}, ARI: {baseline_ari:.4f}")

# ============ 调优实验1：不同单元数配置 ============
print("\n" + "=" * 80)
print("调优实验1：不同单元数配置")
print("=" * 80)

unit_configs = [
    {"name": "默认配置", "units": [400, 350, 300, 200, 150]},
    {"name": "更宽配置", "units": [500, 400, 350, 300, 200]},
    {"name": "更窄配置", "units": [300, 250, 200, 150, 100]},
    {"name": "均匀配置", "units": [300, 300, 300, 300, 300]},
    {"name": "递增配置", "units": [200, 250, 300, 350, 400]},
    {"name": "递减更缓", "units": [400, 380, 360, 340, 320]},
]

unit_results = {}
for config in unit_configs:
    print(f"\n   测试配置: {config['name']} - {config['units']}")
    reservoir_configs = [
        {
            'n_internal_units': units,
            'spectral_radius': 0.99,
            'leak': None,
            'connectivity': 0.3,
            'input_scaling': 0.2,
            'noise_level': 0.0,
            'circle': False
        }
        for units in config['units']
    ]
    
    model = StackedRC_model(
        n_layers=5,
        reservoir_configs=reservoir_configs,
        readout_type=None
    )
    model.fit(X, verbose=False)
    nmi, ari, _ = evaluate_clustering(model.input_repr, true_labels)
    unit_results[config['name']] = {'nmi': nmi, 'ari': ari, 'units': config['units']}
    print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}")

# ============ 调优实验2：不同表示方法 ============
print("\n" + "=" * 80)
print("调优实验2：不同表示方法")
print("=" * 80)

# 使用最佳单元配置
best_unit_config = max(unit_results.items(), key=lambda x: x[1]['nmi'])[1]
best_units = best_unit_config['units']
print(f"   使用最佳单元配置: {best_units}")

representation_methods = ['mean', 'last', 'reservoir']
repr_results = {}

for mts_rep in representation_methods:
    print(f"\n   测试表示方法: {mts_rep}")
    reservoir_configs = [
        {
            'n_internal_units': units,
            'spectral_radius': 0.99,
            'leak': None,
            'connectivity': 0.3,
            'input_scaling': 0.2,
            'noise_level': 0.0,
            'circle': False
        }
        for units in best_units
    ]
    
    model = StackedRC_model(
        n_layers=5,
        reservoir_configs=reservoir_configs,
        mts_rep=mts_rep,
        readout_type=None
    )
    model.fit(X, verbose=False)
    nmi, ari, _ = evaluate_clustering(model.input_repr, true_labels)
    repr_results[mts_rep] = {'nmi': nmi, 'ari': ari}
    print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}")

# ============ 调优实验3：不同谱半径 ============
print("\n" + "=" * 80)
print("调优实验3：不同谱半径")
print("=" * 80)

# 使用最佳表示方法
best_repr_method = max(repr_results.items(), key=lambda x: x[1]['nmi'])[0]
print(f"   使用最佳表示方法: {best_repr_method}")

spectral_radii = [0.90, 0.95, 0.99, 0.999]
spectral_results = {}

for sr in spectral_radii:
    print(f"\n   测试谱半径: {sr}")
    reservoir_configs = [
        {
            'n_internal_units': units,
            'spectral_radius': sr,
            'leak': None,
            'connectivity': 0.3,
            'input_scaling': 0.2,
            'noise_level': 0.0,
            'circle': False
        }
        for units in best_units
    ]
    
    model = StackedRC_model(
        n_layers=5,
        reservoir_configs=reservoir_configs,
        mts_rep=best_repr_method,
        readout_type=None
    )
    model.fit(X, verbose=False)
    nmi, ari, _ = evaluate_clustering(model.input_repr, true_labels)
    spectral_results[sr] = {'nmi': nmi, 'ari': ari}
    print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}")

# ============ 调优实验4：不同连接度 ============
print("\n" + "=" * 80)
print("调优实验4：不同连接度")
print("=" * 80)

# 使用最佳谱半径
best_sr = max(spectral_results.items(), key=lambda x: x[1]['nmi'])[0]
print(f"   使用最佳谱半径: {best_sr}")

connectivities = [0.2, 0.3, 0.4, 0.5]
connectivity_results = {}

for conn in connectivities:
    print(f"\n   测试连接度: {conn}")
    reservoir_configs = [
        {
            'n_internal_units': units,
            'spectral_radius': best_sr,
            'leak': None,
            'connectivity': conn,
            'input_scaling': 0.2,
            'noise_level': 0.0,
            'circle': False
        }
        for units in best_units
    ]
    
    model = StackedRC_model(
        n_layers=5,
        reservoir_configs=reservoir_configs,
        mts_rep=best_repr_method,
        readout_type=None
    )
    model.fit(X, verbose=False)
    nmi, ari, _ = evaluate_clustering(model.input_repr, true_labels)
    connectivity_results[conn] = {'nmi': nmi, 'ari': ari}
    print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}")

# ============ 结果总结 ============
print("\n" + "=" * 80)
print("调优结果总结")
print("=" * 80)

print(f"\n基准配置（默认5层）:")
print(f"  NMI: {baseline_nmi:.4f}, ARI: {baseline_ari:.4f}")

print(f"\n最佳单元配置:")
best_unit_name = max(unit_results.items(), key=lambda x: x[1]['nmi'])[0]
best_unit_result = unit_results[best_unit_name]
print(f"  {best_unit_name}: {best_unit_result['units']}")
print(f"  NMI: {best_unit_result['nmi']:.4f} (提升 {((best_unit_result['nmi']/baseline_nmi-1)*100):+.2f}%)")
print(f"  ARI: {best_unit_result['ari']:.4f} (提升 {((best_unit_result['ari']/baseline_ari-1)*100):+.2f}%)")

print(f"\n最佳表示方法:")
best_repr_result = repr_results[best_repr_method]
print(f"  {best_repr_method}")
print(f"  NMI: {best_repr_result['nmi']:.4f} (提升 {((best_repr_result['nmi']/baseline_nmi-1)*100):+.2f}%)")
print(f"  ARI: {best_repr_result['ari']:.4f} (提升 {((best_repr_result['ari']/baseline_ari-1)*100):+.2f}%)")

print(f"\n最佳谱半径:")
best_sr_result = spectral_results[best_sr]
print(f"  {best_sr}")
print(f"  NMI: {best_sr_result['nmi']:.4f} (提升 {((best_sr_result['nmi']/baseline_nmi-1)*100):+.2f}%)")
print(f"  ARI: {best_sr_result['ari']:.4f} (提升 {((best_sr_result['ari']/baseline_ari-1)*100):+.2f}%)")

print(f"\n最佳连接度:")
best_conn = max(connectivity_results.items(), key=lambda x: x[1]['nmi'])[0]
best_conn_result = connectivity_results[best_conn]
print(f"  {best_conn}")
print(f"  NMI: {best_conn_result['nmi']:.4f} (提升 {((best_conn_result['nmi']/baseline_nmi-1)*100):+.2f}%)")
print(f"  ARI: {best_conn_result['ari']:.4f} (提升 {((best_conn_result['ari']/baseline_ari-1)*100):+.2f}%)")

# 最终最佳配置
print("\n" + "=" * 80)
print("最终最佳配置")
print("=" * 80)
print(f"单元数: {best_units}")
print(f"表示方法: {best_repr_method}")
print(f"谱半径: {best_sr}")
print(f"连接度: {best_conn}")
print(f"\n预期性能:")
print(f"  NMI: {best_conn_result['nmi']:.4f}")
print(f"  ARI: {best_conn_result['ari']:.4f}")

print("\n调优完成！")
