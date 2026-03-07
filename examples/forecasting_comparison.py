"""
预测任务对比实验：原始 RC vs 串联层叠 RC vs 多专家层叠 RC

本脚本对比了三种模型在时间序列预测任务上的表现：
1. 原始 RC_forecaster
2. 串联层叠预测模型（需要实现）
3. 多专家 + 残差层叠预测模型（需要实现）

注意：目前代码库中只有 RC_forecaster，层叠预测模型需要基于 StackedRC_model 的结构进行适配实现。
本脚本先对比原始模型，层叠版本将在后续版本中添加。

使用相同的评估指标（MSE, MAE, RMSE），便于直接对比。
"""

import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from reservoir_computing.modules import RC_forecaster
from reservoir_computing.utils import make_forecasting_dataset
from reservoir_computing.datasets import PredLoader

# 设置随机种子以确保可重复性
np.random.seed(0)

print("=" * 80)
print("预测任务对比实验：原始 RC vs 串联层叠 RC vs 多专家层叠 RC")
print("=" * 80)
print("\n注意：目前只实现了原始 RC_forecaster，层叠预测模型需要单独实现。")
print("本脚本先展示原始模型的性能，作为后续层叠模型的 baseline。\n")

# 加载数据
print("1. 加载数据...")
ts_full = PredLoader().get_data('ElecRome')

# Resample the time series to hourly frequency
ts_hourly = np.mean(ts_full.reshape(-1, 6), axis=1)

# Use only the first 3000 time steps
ts_small = ts_hourly[0:3000, None]

print(f"   时间序列长度: {ts_small.shape[0]}")
print(f"   变量数量: {ts_small.shape[1]}")

# Generate training and testing datasets
print("\n2. 生成训练和测试数据集...")
Xtr, Ytr, Xte, Yte, scaler = make_forecasting_dataset(
    ts_small, 
    horizon=24,  # forecast horizon of 24h ahead
    test_percent=0.1
)

print(f"   训练集形状: {Xtr.shape}")
print(f"   测试集形状: {Xte.shape}")
print(f"   预测步长: 24")

# ============ 原始RC预测模型 ============
print("\n3. 训练原始RC预测模型...")
start_time = time.time()
forecaster_original = RC_forecaster(n_internal_units=900, w_ridge=1.0)
forecaster_original.fit(Xtr, Ytr, verbose=True)
train_time_original = time.time() - start_time

# 预测
Yhat_original = forecaster_original.predict(Xte)
Yhat_original = scaler.inverse_transform(Yhat_original)  # Revert the scaling
Yte_unscaled = scaler.inverse_transform(Yte)  # Revert the scaling for ground truth

# 计算指标
mse_original = mean_squared_error(Yte_unscaled, Yhat_original)
mae_original = mean_absolute_error(Yte_unscaled, Yhat_original)
rmse_original = np.sqrt(mse_original)

print(f"   训练时间: {train_time_original:.2f} 秒")
print(f"   MSE: {mse_original:.4f}")
print(f"   MAE: {mae_original:.4f}")
print(f"   RMSE: {rmse_original:.4f}")

# ============ 串联层叠预测模型 ============
print("\n4. 串联层叠预测模型（待实现）...")
print("   注意：层叠预测模型需要基于 StackedRC_model 的结构进行适配。")
print("   预测任务的输入格式为 [T, V]（单条时间序列），")
print("   而分类任务的输入格式为 [N, T, V]（多个样本）。")
print("   需要实现 StackedRC_forecaster 类来支持层叠预测。")

# TODO: 实现层叠预测模型后，取消以下注释并添加实现
# stacked_forecasters = {}
# stacked_results = {}
# stacked_train_times = {}
# 
# for n_layers in [2, 3, 4, 5]:
#     print(f"\n   训练 {n_layers} 层层叠预测模型...")
#     start_time = time.time()
#     forecaster_stacked = StackedRC_forecaster(
#         n_layers=n_layers,
#         reservoir_configs=None,
#         w_ridge=1.0
#     )
#     forecaster_stacked.fit(Xtr, Ytr, verbose=False)
#     train_time = time.time() - start_time
#     
#     Yhat_stacked = forecaster_stacked.predict(Xte)
#     Yhat_stacked = scaler.inverse_transform(Yhat_stacked)
#     
#     mse = mean_squared_error(Yte_unscaled, Yhat_stacked)
#     mae = mean_absolute_error(Yte_unscaled, Yhat_stacked)
#     rmse = np.sqrt(mse)
#     
#     stacked_results[n_layers] = {
#         'mse': mse,
#         'mae': mae,
#         'rmse': rmse
#     }
#     stacked_train_times[n_layers] = train_time

# ============ 多专家层叠预测模型 ============
print("\n5. 多专家 + 残差层叠预测模型（待实现）...")
print("   需要实现 MultiExpertStackedRC_forecaster 类来支持多专家预测。")

# TODO: 实现多专家预测模型后，取消以下注释并添加实现
# multi_expert_settings = [
#     (3, 1), (3, 2), (3, 3), (3, 5),
#     (5, 1), (5, 2), (5, 3), (5, 5),
# ]
# 
# multi_expert_results = {}
# multi_expert_train_times = {}
# 
# for n_layers, n_experts in multi_expert_settings:
#     print(f"\n   训练 多专家 {n_layers} 层层叠预测模型（n_experts={n_experts}）...")
#     start_time = time.time()
#     forecaster_me = MultiExpertStackedRC_forecaster(
#         n_layers=n_layers,
#         n_experts=n_experts,
#         reservoir_configs=None,
#         mts_rep='mean',
#         w_ridge=1.0
#     )
#     forecaster_me.fit(Xtr, Ytr, verbose=False)
#     train_time = time.time() - start_time
#     
#     Yhat_me = forecaster_me.predict(Xte)
#     Yhat_me = scaler.inverse_transform(Yhat_me)
#     
#     mse = mean_squared_error(Yte_unscaled, Yhat_me)
#     mae = mean_absolute_error(Yte_unscaled, Yhat_me)
#     rmse = np.sqrt(mse)
#     
#     multi_expert_results[(n_layers, n_experts)] = {
#         'mse': mse,
#         'mae': mae,
#         'rmse': rmse
#     }
#     multi_expert_train_times[(n_layers, n_experts)] = train_time

# ============ 结果汇总 ============
print("\n" + "=" * 100)
print("预测任务结果汇总（当前只有原始 RC 模型）")
print("=" * 100)
print(f"{'模型':<25} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'训练时间(秒)':<15}")
print("-" * 100)
print(f"{'原始RC':<25} {mse_original:<12.4f} {mae_original:<12.4f} {rmse_original:<12.4f} {train_time_original:<15.2f}")

print("\n" + "=" * 100)
print("后续工作")
print("=" * 100)
print("1. 实现 StackedRC_forecaster 类，支持串联层叠预测")
print("2. 实现 MultiExpertStackedRC_forecaster 类，支持多专家层叠预测")
print("3. 完成三种模型的完整对比实验")
print("\n实现建议：")
print("- 参考 StackedRC_model 的结构，适配预测任务的输入格式 [T, V]")
print("- 每层处理前一层输出的状态序列，最后一层用于预测")
print("- 保持与 RC_forecaster 相同的接口（fit, predict）")

print("\n实验完成！")
