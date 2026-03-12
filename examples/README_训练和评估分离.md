# 训练和评估分离说明

为了节省时间，避免每次运行都重新训练模型，已将训练和评估分离为两个独立的脚本。

## 文件说明

1. **`train_models.py`** - 训练所有模型并保存表示向量
   - 训练所有四种模型（原始RC、层叠RC、多专家层叠RC、MoE层叠RC）
   - 将所有模型的表示向量保存到 `saved_representations/` 目录
   - 只需运行一次，除非需要重新训练模型

2. **`evaluate_and_visualize.py`** - 加载保存的表示向量，进行聚类评估和可视化
   - 从 `saved_representations/` 目录加载已保存的表示向量
   - 进行聚类评估（计算NMI、ARI等指标）
   - 生成UMAP可视化图
   - 输出对比分析结果
   - 可以多次运行，快速调整可视化参数或重新生成图表

3. **`stacked_clustering_comparison.py`** - 原始完整脚本（保留作为参考）
   - 包含训练和评估的完整流程
   - 如果需要一次性运行所有步骤，可以使用此脚本

## 使用流程

### 第一次运行（训练模型）

```bash
cd examples
python train_models.py
```

这将：
- 训练所有模型
- 保存表示向量到 `saved_representations/` 目录
- 保存配置信息（专家数设置等）

### 后续运行（评估和可视化）

```bash
python evaluate_and_visualize.py
```

这将：
- 加载已保存的表示向量
- 进行聚类评估
- 生成所有UMAP可视化图
- 输出分析结果

## 保存的文件结构

```
saved_representations/
├── true_labels.npy                                    # 真实标签
├── original_rc_representation.npy                    # 原始RC模型表示
├── stacked_rc_6layers_representation.npy             # 层叠RC模型表示（6层）
├── multi_expert_6layers_5experts_representation.npy  # 多专家模型表示
├── multi_expert_6layers_10experts_representation.npy
├── multi_expert_6layers_15experts_representation.npy
├── multi_expert_6layers_20experts_representation.npy
├── moe_6layers_5experts_representation.npy           # MoE模型表示
├── moe_6layers_10experts_representation.npy
├── moe_6layers_15experts_representation.npy
├── moe_6layers_20experts_representation.npy
├── multi_expert_settings.pkl                        # 多专家模型配置
└── moe_settings.pkl                                  # MoE模型配置
```

## 优势

1. **节省时间**：训练模型通常需要较长时间，分离后只需训练一次
2. **快速迭代**：可以快速调整可视化参数、重新生成图表，无需重新训练
3. **灵活性**：可以单独运行评估脚本，方便调试和实验
4. **数据持久化**：表示向量保存后可以随时使用，不会丢失

## 注意事项

- 如果修改了模型配置（如层数、专家数），需要重新运行 `train_models.py`
- 如果 `saved_representations/` 目录不存在或文件缺失，`evaluate_and_visualize.py` 会提示错误
- 表示向量文件使用 `.npy` 格式（NumPy数组），配置信息使用 `.pkl` 格式（Pickle）
