#!/usr/bin/env python3
"""
层叠 Reservoir 实验生成器
根据 experiment_config.json 生成 Kaggle Notebook 可运行的实验代码

用法:
    python generate_experiment.py --exp exp_002 --output kaggle_exp_002.py
"""

import json
import argparse
import os
from datetime import datetime

def load_config():
    """加载实验配置"""
    config_path = os.path.join(os.path.dirname(__file__), 'experiment_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_experiment(exp_id):
    """获取指定实验配置"""
    config = load_config()
    for exp in config['experiment_queue']:
        if exp['id'] == exp_id:
            return exp, config
    return None, config

def generate_kaggle_code(exp, config):
    """生成 Kaggle Notebook 代码"""
    
    exp_config = exp['config']
    datasets = exp.get('datasets', ['Japanese_Vowels'])
    
    code = f'''#!/usr/bin/env python3
"""
实验：{exp['name']}
实验 ID: {exp['id']}
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

配置说明:
{json.dumps(exp_config, indent=2, ensure_ascii=False)}
"""

# ============================================================================
# 步骤 1: 安装依赖
# ============================================================================
!pip install numpy scikit-learn scipy matplotlib pandas -q

# ============================================================================
# 步骤 2: 导入库
# ============================================================================
import numpy as np
import sys
import os
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import time

# 设置项目路径
PROJECT_PATH = '/kaggle/working/Time-series-classification-and-clustering-with-Reservoir-Computing'
sys.path.insert(0, PROJECT_PATH)

# ============================================================================
# 步骤 3: 实验配置
# ============================================================================
EXPERIMENT_ID = "{exp['id']}"
EXPERIMENT_NAME = "{exp['name']}"

# 层叠 Reservoir 配置
STACKED_CONFIG = {{
    'n_layers': {exp_config['n_layers']},
    'units_per_layer': {exp_config['units_per_layer']},
    'spectral_radius': {exp_config['spectral_radius']},
    'connectivity': {exp_config['connectivity']},
    'input_scaling': {exp_config['input_scaling']},
    'leak': {exp_config['leak']},
    'noise_level': {exp_config['noise_level']},
    'representation': '{exp_config['representation']}'
}}

# 数据集列表
DATASETS = {datasets}

print("=" * 70)
print(f"实验：{{EXPERIMENT_NAME}}")
print(f"实验 ID: {{EXPERIMENT_ID}}")
print("=" * 70)
print("\\n层叠 Reservoir 配置:")
for key, value in STACKED_CONFIG.items():
    print(f"  {{key}}: {{value}}")
print(f"\\n数据集：{{DATASETS}}")
print("=" * 70)

# ============================================================================
# 步骤 4: 导入项目模块
# ============================================================================
try:
    from reservoir_computing.modules import RC_model, StackedRCModel
    from reservoir_computing.utils import compute_test_scores
    from reservoir_computing.datasets import ClfLoader
    print("✓ 项目模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败：{{e}}")
    print("提示：请确保项目文件已正确解压到 {{PROJECT_PATH}}")
    # 尝试从 GitHub 克隆
    print("\\n尝试从 GitHub 克隆项目...")
    !git clone https://github.com/wanning233/Time-series-classification-and-clustering-with-Reservoir-Computing.git /kaggle/working/Time-series-classification-and-clustering-with-Reservoir-Computing
    from reservoir_computing.modules import RC_model, StackedRCModel
    from reservoir_computing.utils import compute_test_scores
    from reservoir_computing.datasets import ClfLoader
    print("✓ 克隆完成，模块导入成功")

# ============================================================================
# 步骤 5: 定义层叠聚类模型
# ============================================================================
class StackedReservoirClustering:
    """层叠 Reservoir 聚类模型"""
    
    def __init__(self, n_layers=5, units_per_layer=None, **rc_params):
        self.n_layers = n_layers
        self.units_per_layer = units_per_layer or [400, 350, 300, 200, 150]
        self.rc_params = rc_params
        self.layers = []
        self.input_repr = None
        
    def fit(self, X_list):
        """训练层叠模型"""
        print(f"训练层叠 Reservoir 模型 ({{self.n_layers}}层)...")
        
        # 初始化各层
        self.layers = []
        current_input = X_list
        
        for i in range(self.n_layers):
            n_units = self.units_per_layer[i]
            print(f"  第 {{i+1}} 层：{{n_units}} 单元")
            
            layer = RC_model(
                n_internal_units=n_units,
                **self.rc_params
            )
            
            # 训练当前层
            layer.fit(current_input)
            self.layers.append(layer)
            
            # 下一层的输入是当前层的表示
            current_input = layer.input_repr
            
        # 保存最终表示
        self.input_repr = current_input
        print(f"✓ 层叠模型训练完成，最终表示形状：{{self.input_repr.shape}}")
        return self
    
    def get_representation(self, X_list, representation='mean'):
        """获取时间序列表示"""
        if representation == 'mean':
            # 各层状态的均值
            reprs = []
            current_input = X_list
            for layer in self.layers:
                layer_repr = layer.get_representation(current_input, representation='mean')
                reprs.append(layer_repr)
                current_input = layer.input_repr
            # 拼接各层表示
            return np.hstack(reprs)
        elif representation == 'last':
            reprs = []
            current_input = X_list
            for layer in self.layers:
                layer_repr = layer.get_representation(current_input, representation='last')
                reprs.append(layer_repr)
                current_input = layer.input_repr
            return np.hstack(reprs)
        else:
            # 使用最后一层的表示
            return self.input_repr
    
    def cluster(self, n_clusters, random_state=0):
        """执行聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.input_repr)
        return labels, kmeans

# ============================================================================
# 步骤 6: 定义评估函数
# ============================================================================
def evaluate_clustering(true_labels, pred_labels, dataset_name):
    """评估聚类结果"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    print(f"\\n数据集：{{dataset_name}}")
    print(f"  NMI: {{nmi:.4f}}")
    print(f"  ARI: {{ari:.4f}}")
    
    return {{'NMI': nmi, 'ARI': ari}}

# ============================================================================
# 步骤 7: 运行实验
# ============================================================================
results = []

for dataset_name in DATASETS:
    print(f"\\n{'='*70}")
    print(f"数据集：{{dataset_name}}")
    print('='*70)
    
    try:
        # 加载数据
        print("\\n加载数据...")
        Xtr, Ytr, Xte, Yte = ClfLoader().get_data(dataset_name)
        print(f"✓ 数据加载完成")
        print(f"  训练集：{{len(Xtr)}} 个序列")
        print(f"  测试集：{{len(Xte)}} 个序列")
        print(f"  类别数：{{len(np.unique(Ytr))}}")
        
        # 创建层叠模型
        print("\\n创建层叠 Reservoir 模型...")
        model = StackedReservoirClustering(
            n_layers=STACKED_CONFIG['n_layers'],
            units_per_layer=STACKED_CONFIG['units_per_layer'],
            spectral_radius=STACKED_CONFIG['spectral_radius'],
            connectivity=STACKED_CONFIG['connectivity'],
            input_scaling=STACKED_CONFIG['input_scaling'],
            leak=STACKED_CONFIG['leak'],
            noise_level=STACKED_CONFIG['noise_level']
        )
        
        # 训练模型
        print("\\n训练模型...")
        start_time = time.time()
        model.fit(Xtr)
        train_time = time.time() - start_time
        print(f"✓ 训练完成，用时：{{train_time:.2f}} 秒")
        
        # 获取表示并聚类
        print("\\n获取时间序列表示...")
        n_clusters = len(np.unique(Ytr))
        pred_labels, kmeans = model.cluster(n_clusters)
        
        # 评估结果
        print("\\n评估聚类结果...")
        metrics = evaluate_clustering(Ytr, pred_labels, dataset_name)
        metrics['dataset'] = dataset_name
        metrics['train_time'] = train_time
        metrics['config'] = STACKED_CONFIG.copy()
        results.append(metrics)
        
    except Exception as e:
        print(f"✗ 实验失败：{{e}}")
        import traceback
        traceback.print_exc()
        results.append({{
            'dataset': dataset_name,
            'error': str(e),
            'config': STACKED_CONFIG.copy()
        }})

# ============================================================================
# 步骤 8: 汇总结果
# ============================================================================
print(f"\\n{'='*70}")
print("实验结果汇总")
print('='*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 保存结果
output_file = f'/kaggle/working/results_{{EXPERIMENT_ID}}_{{datetime.now().strftime("%Y%m%d_%H%M%S")}}.csv'
results_df.to_csv(output_file, index=False)
print(f"\\n✓ 结果已保存：{{output_file}}")

# 保存配置
config_file = f'/kaggle/working/config_{{EXPERIMENT_ID}}_{{datetime.now().strftime("%Y%m%d_%H%M%S")}}.json'
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump({{
        'experiment_id': EXPERIMENT_ID,
        'experiment_name': EXPERIMENT_NAME,
        'config': STACKED_CONFIG,
        'datasets': DATASETS,
        'timestamp': datetime.now().isoformat()
    }}, f, indent=2, ensure_ascii=False)
print(f"✓ 配置已保存：{{config_file}}")

# ============================================================================
# 步骤 9: 可视化
# ============================================================================
if len(results) > 0 and 'NMI' in results[0]:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    nmi_data = results_df[results_df['NMI'].notna()]
    if len(nmi_data) > 0:
        plt.bar(nmi_data['dataset'], nmi_data['NMI'])
        plt.xlabel('Dataset')
        plt.ylabel('NMI Score')
        plt.title(f'NMI by Dataset ({{EXPERIMENT_ID}})')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    ari_data = results_df[results_df['ARI'].notna()]
    if len(ari_data) > 0:
        plt.bar(ari_data['dataset'], ari_data['ARI'])
        plt.xlabel('Dataset')
        plt.ylabel('ARI Score')
        plt.title(f'ARI by Dataset ({{EXPERIMENT_ID}})')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(f'/kaggle/working/visualization_{{EXPERIMENT_ID}}.png', dpi=150, bbox_inches='tight')
    print(f"✓ 可视化已保存：/kaggle/working/visualization_{{EXPERIMENT_ID}}.png")
    plt.show()

print(f"\\n{'='*70}")
print(f"实验 {{EXPERIMENT_ID}} 完成!")
print('='*70)
'''
    
    return code

def list_experiments():
    """列出所有可用实验"""
    config = load_config()
    print(f"\n{'='*70}")
    print(f"可用实验列表 ({len(config['experiment_queue'])} 个)")
    print('='*70)
    
    for exp in config['experiment_queue']:
        status_icon = '⏳' if exp['status'] == 'pending' else '✅' if exp['status'] == 'completed' else '🔄'
        priority = exp.get('priority', 99)
        print(f"{status_icon} {exp['id']} (优先级：{priority})")
        print(f"   名称：{exp['name']}")
        print(f"   配置：{exp['config']['n_layers']}层，单元={exp['config']['units_per_layer']}")
        print(f"   数据集：{exp.get('datasets', ['Japanese_Vowels'])}")
        print(f"   说明：{exp.get('note', 'N/A')}")
        print()

def main():
    parser = argparse.ArgumentParser(description='层叠 Reservoir 实验生成器')
    parser.add_argument('--exp', type=str, help='实验 ID (如：exp_002)')
    parser.add_argument('--list', action='store_true', help='列出所有实验')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if not args.exp:
        print("错误：请指定实验 ID (--exp) 或使用 --list 查看所有实验")
        parser.print_help()
        return
    
    exp, config = get_experiment(args.exp)
    if not exp:
        print(f"错误：未找到实验 {args.exp}")
        return
    
    print(f"生成实验代码：{exp['name']}")
    code = generate_kaggle_code(exp, config)
    
    output_path = args.output or f"kaggle_{args.exp}.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"✓ 代码已生成：{output_path}")
    print(f"\n使用方法:")
    print(f"  1. 在 Kaggle Notebook 中创建新单元格")
    print(f"  2. 复制 {output_path} 的内容")
    print(f"  3. 运行所有单元格")

if __name__ == "__main__":
    main()
