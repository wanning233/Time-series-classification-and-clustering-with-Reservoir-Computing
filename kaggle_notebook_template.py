"""
Kaggle Notebook 快速开始模板
将此代码复制到 Kaggle Notebook 中即可快速开始使用本项目

注意: 在 Kaggle Notebook 中，每个以 ! 开头的命令需要在单独的代码单元格中运行
"""

# ============================================================================
# 步骤 1: 安装依赖包
# 在 Kaggle Notebook 的第一个单元格中运行以下命令:
# !pip install numpy scikit-learn scipy requests matplotlib
# ============================================================================

# ============================================================================
# 步骤 2: 设置项目路径
# ============================================================================
import sys
import os

# 根据您的文件上传方式选择以下之一：
# 选项 A: 如果项目文件在 /kaggle/input/ 目录（推荐使用 Kaggle Datasets）
PROJECT_PATH = '/kaggle/input/reservoir-computing-project'  # 请修改为您的实际路径

# 选项 B: 如果项目文件在 /kaggle/working/ 目录
# PROJECT_PATH = '/kaggle/working/Time-series-classification-and-clustering-with-Reservoir-Computing'

# 选项 C: 如果从 GitHub 克隆（需要网络访问）
# !git clone https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing.git
# PROJECT_PATH = '/kaggle/working/Time-series-classification-and-clustering-with-Reservoir-Computing'

sys.path.insert(0, PROJECT_PATH)

# ============================================================================
# 步骤 3: 导入项目模块
# ============================================================================
import numpy as np
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores
from reservoir_computing.datasets import ClfLoader
from sklearn.preprocessing import OneHotEncoder

# ============================================================================
# 步骤 4: 设置随机种子（确保结果可复现）
# ============================================================================
np.random.seed(0)

# ============================================================================
# 步骤 5: 加载数据
# ============================================================================
print("=" * 60)
print("正在加载数据...")
print("=" * 60)

# 可用的数据集包括：
# 'Japanese_Vowels', 'Libras', 'UWaveGestureLibrary', 'CharacterTrajectories', 
# 'ECG200', 'ECG5000', 'FordA', 'FordB', 'Wafer', 'Yoga', 'StarLightCurves'
dataset_name = 'Japanese_Vowels'  # 可以修改为其他数据集

try:
    Xtr, Ytr, Xte, Yte = ClfLoader().get_data(dataset_name)
    print(f"✓ 成功加载数据集: {dataset_name}")
    print(f"  训练集形状: {Xtr.shape}")
    print(f"  测试集形状: {Xte.shape}")
    print(f"  类别数: {len(np.unique(Ytr))}")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    print("提示: 请确保 Kaggle Notebook 已启用互联网访问")

# ============================================================================
# 步骤 6: 准备标签（One-hot 编码）
# ============================================================================
print("\n" + "=" * 60)
print("正在准备标签...")
print("=" * 60)

onehot_encoder = OneHotEncoder(sparse_output=False)
Ytr_encoded = onehot_encoder.fit_transform(Ytr.reshape(-1, 1))
Yte_encoded = onehot_encoder.transform(Yte.reshape(-1, 1))

print(f"✓ 标签编码完成")
print(f"  训练标签形状: {Ytr_encoded.shape}")
print(f"  测试标签形状: {Yte_encoded.shape}")

# ============================================================================
# 步骤 7: 创建和训练模型
# ============================================================================
print("\n" + "=" * 60)
print("正在初始化模型...")
print("=" * 60)

# 可以调整的参数：
# - n_internal_units: 储备池内部单元数（默认 500，可以降低以节省内存）
# - spectral_radius: 谱半径（默认 0.9）
# - leaky: 是否使用泄漏积分器（默认 False）
# - connectivity: 连接密度（默认 1.0）
# - input_scaling: 输入缩放（默认 1.0）
# - n_drop: 丢弃的前 n 个状态（默认 0）
# - bidir: 是否使用双向储备池（默认 False）
# - readout_type: 读出层类型（默认 'ridge'）

classifier = RC_model(
    n_internal_units=500,  # 如果内存不足，可以降低到 200 或 300
    spectral_radius=0.9,
    leaky=False,
    connectivity=1.0,
    input_scaling=1.0,
    n_drop=0,
    bidir=False,
    readout_type='ridge'
)

print("✓ 模型初始化完成")

print("\n" + "=" * 60)
print("正在训练模型...")
print("=" * 60)

tr_time = classifier.fit(Xtr, Ytr_encoded)
print(f"✓ 训练完成")
print(f"  训练时间: {tr_time:.2f} 秒")

# ============================================================================
# 步骤 8: 预测和评估
# ============================================================================
print("\n" + "=" * 60)
print("正在进行预测和评估...")
print("=" * 60)

pred_class = classifier.predict(Xte)
accuracy, f1 = compute_test_scores(pred_class, Yte_encoded)

print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"F1 分数: {f1:.4f}")
print("=" * 60)

# ============================================================================
# 可选: 保存结果
# ============================================================================
# 如果需要保存结果到 /kaggle/working/ 目录：
# import pickle
# with open('/kaggle/working/model_results.pkl', 'wb') as f:
#     pickle.dump({
#         'accuracy': accuracy,
#         'f1': f1,
#         'predictions': pred_class
#     }, f)
# print("\n结果已保存到 /kaggle/working/model_results.pkl")
