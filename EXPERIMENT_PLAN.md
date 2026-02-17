# 第 2 轮实验计划 - 层叠 Reservoir 调优

**基准结果**: 5 层配置 NMI=0.8295, ARI=0.7856  
**目标**: 系统性地探索超参数空间，寻找更优配置  
**更新时间**: 2026-02-17

---

## 📊 实验总览

| 阶段 | 实验数 | 目标 |
|------|--------|------|
| Phase 1 | 4 个 | 单元配置调优 |
| Phase 2 | 2 个 | 谱半径调优 |
| Phase 3 | 2 个 | 连接度调优 |
| Phase 4 | 1 个 | 噪声正则化 |
| Phase 5 | 1 个 | 最佳组合验证 |
| **总计** | **10 个** | 系统性探索 |

---

## 🎯 实验列表

### Phase 1: 单元配置调优（优先级 1-3, 8）

| ID | 配置 | 数据集 | 假设 |
|----|------|--------|------|
| **exp_001** | `[400,350,300,200,150]` 基准 | Japanese_Vowels, Libras | 复现 NMI≈0.83 |
| **exp_002** | `[500,400,350,300,200]` 宽配置 | Japanese_Vowels, Libras | 更强表达能力 |
| **exp_003** | `[400,380,360,340,320]` 缓递减 | Japanese_Vowels, Libras | 保留更多信息 |
| **exp_008** | `[350,350,350,350,350]` 均匀 | Japanese_Vowels, Libras | 测试均匀配置 |

**预期**: 宽配置可能提升性能，但需警惕过拟合

---

### Phase 2: 谱半径调优（优先级 4-5）

| ID | 谱半径 | 数据集 | 假设 |
|----|--------|--------|------|
| **exp_004** | 0.999 | Japanese_Vowels, Libras | 更强记忆，适合长序列 |
| **exp_005** | 0.95 | Japanese_Vowels, Libras | 更稳定，减少过拟合 |

**预期**: 0.999 可能提升长序列任务，0.95 可能更稳定

---

### Phase 3: 连接度调优（优先级 6-7）

| ID | 连接度 | 数据集 | 假设 |
|----|--------|--------|------|
| **exp_006** | 0.4 密集 | Japanese_Vowels, Libras | 更强表达能力 |
| **exp_007** | 0.2 稀疏 | Japanese_Vowels, Libras | 更稳定，泛化更好 |

**预期**: 0.4 可能提升性能，0.2 可能提高泛化

---

### Phase 4: 噪声正则化（优先级 9）

| ID | 噪声水平 | 数据集 | 假设 |
|----|----------|--------|------|
| **exp_009** | 0.05 | Japanese_Vowels, Libras | 提高泛化能力 |

**预期**: 轻微噪声可能作为正则化，减少过拟合

---

### Phase 5: 最佳组合验证（优先级 10）

| ID | 配置 | 数据集 | 目标 |
|----|------|--------|------|
| **exp_010** | 宽配置 +0.999+0.4 | Japanese_Vowels, Libras, UWave | 验证激进组合 |

**预期**: 组合各阶段最优参数，测试性能上限

---

## 🚀 快速开始

### 方式 A: 使用生成脚本（推荐）

```bash
# 1. 查看可用实验
python generate_experiment.py --list

# 2. 生成指定实验代码
python generate_experiment.py --exp exp_002 --output kaggle_exp_002.py

# 3. 复制生成的代码到 Kaggle Notebook 运行
```

### 方式 B: 手动配置

编辑 `experiment_config.json` 中的 `experiment_queue`，然后：

```python
# Kaggle Notebook 代码模板
EXPERIMENT_CONFIG = {
    'n_layers': 5,
    'units_per_layer': [500, 400, 350, 300, 200],  # 修改这里
    'spectral_radius': 0.99,
    'connectivity': 0.3,
    'input_scaling': 0.2,
    'noise_level': 0.0,
}
```

---

## 📋 执行顺序建议

```
第 1 天: exp_001 (基准复现) → 验证环境正确
第 2 天: exp_002 (宽配置) → 如果有提升，继续
第 3 天: exp_003, exp_008 (其他单元配置)
第 4 天: exp_004, exp_005 (谱半径)
第 5 天: exp_006, exp_007 (连接度)
第 6 天: exp_009 (噪声)
第 7 天: exp_010 (最佳组合)
```

---

## 📊 结果记录模板

每次实验后更新 `experiment_results.md`:

```markdown
## exp_002 - 宽配置测试

**日期**: 2026-02-17  
**状态**: ⏳ 待运行 / 🔄 运行中 / ✅ 完成 / ❌ 失败

### 配置
- 层数：5
- 单元：[500, 400, 350, 300, 200]
- 谱半径：0.99
- 连接度：0.3

### 结果
| 数据集 | NMI | ARI | 训练时间 |
|--------|-----|-----|----------|
| Japanese_Vowels | - | - | - |
| Libras | - | - | - |

### 分析
- 相比基准：NMI ±X.XX, ARI ±X.XX
- 观察：...
- 结论：...
```

---

## 🎯 成功标准

| 指标 | 基准 | 目标 |
|------|------|------|
| NMI (Japanese_Vowels) | 0.8295 | >0.85 |
| ARI (Japanese_Vowels) | 0.7856 | >0.80 |
| 稳定性 | - | 多次运行方差<0.01 |

---

## ⚠️ 注意事项

1. **Kaggle 限制**: 每次运行最长 9 小时，合理安排实验
2. **内存管理**: 宽配置可能占用更多内存，注意监控
3. **结果保存**: 每次实验后及时下载 CSV 和可视化结果
4. **版本控制**: 每次实验前 commit 代码变更

---

## 📁 输出文件

每次实验生成：
- `results_exp_XXX_timestamp.csv` - 数值结果
- `config_exp_XXX_timestamp.json` - 实验配置
- `visualization_exp_XXX.png` - 可视化图表

---

## 🔗 相关链接

- [WORKFLOW.md](./WORKFLOW.md) - 完整工作流
- [experiment_config.json](./experiment_config.json) - 实验配置
- [层叠 Reservoir 调优指南.md](./层叠 Reservoir 调优指南.md) - 调优参数说明
