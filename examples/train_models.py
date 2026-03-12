"""
训练所有Reservoir计算模型并保存表示向量

本脚本训练以下四种模型：
1. RC_model: 原始单层Reservoir模型
2. StackedRC_model: 层叠Reservoir模型
3. MultiExpertStackedRC_model: 多专家+残差式层叠Reservoir模型
4. MoEStackedRC_model: 混合专家（MoE）层叠Reservoir模型

训练完成后，将所有模型的表示向量保存到文件中，供后续评估和可视化使用。
"""

import numpy as np
import os
import pickle

from reservoir_computing.modules import (
    RC_model, 
    StackedRC_model, 
    MultiExpertStackedRC_model,
    MoEStackedRC_model
)
from reservoir_computing.datasets import ClfLoader

# 设置随机种子以确保可重复性
np.random.seed(0)

# 创建保存目录
save_dir = 'saved_representations'
os.makedirs(save_dir, exist_ok=True)

print("=" * 60)
print("训练所有Reservoir计算模型")
print("=" * 60)

# 加载数据
print("\n1. 加载数据...")
Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')

# 由于进行聚类，不需要训练/测试分割
X = np.concatenate((Xtr, Xte), axis=0)
Y = np.concatenate((Ytr, Yte), axis=0)

print(f"   数据形状: {X.shape}")
print(f"   类别数量: {len(np.unique(Y[:,0]))}")

# 保存数据标签
np.save(os.path.join(save_dir, 'true_labels.npy'), Y[:, 0])
print(f"\n   已保存真实标签到: {os.path.join(save_dir, 'true_labels.npy')}")

# ============ 原始RC模型 ============
print("\n2. 训练原始RC模型...")
rcm_original = RC_model(
    n_internal_units=400,
    readout_type=None  # 设置为None以存储输入表示
)

rcm_original.fit(X, verbose=True)
mts_representations_original = rcm_original.input_repr
print(f"   表示维度: {mts_representations_original.shape}")

# 保存表示向量
np.save(os.path.join(save_dir, 'original_rc_representation.npy'), mts_representations_original)
print(f"   已保存表示向量到: {os.path.join(save_dir, 'original_rc_representation.npy')}")

# ============ 层叠RC模型 ============
print("\n3. 训练层叠RC模型（6层）...")
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

# 保存表示向量
np.save(os.path.join(save_dir, f'stacked_rc_{n_layers}layers_representation.npy'), mts_representations_stacked)
print(f"   已保存表示向量到: {os.path.join(save_dir, f'stacked_rc_{n_layers}layers_representation.npy')}")

# ============ 多专家 + 残差层叠RC模型 ============
print("\n4. 训练多专家 + 残差层叠RC模型（6层，专家数递增：5, 10, 15, 20）...")

# 实验组合：固定6层，每层专家数分别为 5, 10, 15, 20
multi_expert_settings = [
    (6, 5),
    (6, 10),
    (6, 15),
    (6, 20),
]

multi_expert_representations = {}

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

    multi_expert_representations[(n_layers, n_experts)] = mts_repr_me
    
    # 保存表示向量
    filename = f'multi_expert_{n_layers}layers_{n_experts}experts_representation.npy'
    np.save(os.path.join(save_dir, filename), mts_repr_me)
    print(f"   已保存表示向量到: {os.path.join(save_dir, filename)}")

# 保存配置信息
with open(os.path.join(save_dir, 'multi_expert_settings.pkl'), 'wb') as f:
    pickle.dump(multi_expert_settings, f)

# ============ MoE混合专家层叠RC模型 ============
print("\n5. 训练MoE混合专家层叠RC模型（6层，专家数递增：5, 10, 15, 20）...")

# 实验组合：固定6层，每层专家数分别为 5, 10, 15, 20
moe_settings = [
    (6, 5),
    (6, 10),
    (6, 15),
    (6, 20),
]

moe_representations = {}

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

    moe_representations[(n_layers, n_experts)] = mts_repr_moe
    
    # 保存表示向量
    filename = f'moe_{n_layers}layers_{n_experts}experts_representation.npy'
    np.save(os.path.join(save_dir, filename), mts_repr_moe)
    print(f"   已保存表示向量到: {os.path.join(save_dir, filename)}")

# 保存配置信息
with open(os.path.join(save_dir, 'moe_settings.pkl'), 'wb') as f:
    pickle.dump(moe_settings, f)

print("\n" + "=" * 60)
print("所有模型训练完成！")
print(f"所有表示向量已保存到: {save_dir}/")
print("=" * 60)
