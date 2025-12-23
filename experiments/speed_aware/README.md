# Speed-Aware Flow Matching for CALVIN

**独立实验** - 与 `freq_flow` 完全解耦，可以同时存在

## 核心思想

不同于频域专家分解，本方案通过**学习运动速度**来自适应调整动作平滑度：

- **快速运动**（push, slide）→ 保留高频细节
- **精细操作**（lift, place）→ 强低频平滑

### 与 freq_flow 的区别

| 方面 | freq_flow | speed_aware |
|------|-----------|-------------|
| **核心机制** | 频域专家MoE分解 | 速度感知自适应平滑 |
| **训练损失** | 子带一致性损失（已禁用） | 速度预测辅助损失 |
| **与FM兼容性** | 冲突（对x_t做DCT无意义） | ✓ 完全兼容 |
| **实现复杂度** | 高（4个expert + 频域损失） | 低（单头 + 轻量预测器） |
| **物理解释** | 超低频/低频/高频？ | 快速/精细操作 ✓ |

---

## 快速开始

### 1. 训练 Speed-Aware 模型

```bash
cd /mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/speed_aware/scripts
./train_speed_aware.sh
```

训练日志：`experiments/speed_aware/logs/`

### 2. 监控训练

WandB项目: `MoDE_SpeedAware`

关键指标：
- `train/fm_loss`: Flow Matching主损失
- `train/speed_loss`: 速度预测辅助损失
- `train/speed_label_avg`: 真实速度标签
- `train/speed_pred_avg`: 预测速度

**预期**：
- push/slide任务: speed ≈ 0.7-0.9（高速）
- lift/place任务: speed ≈ 0.2-0.4（低速）
- rotate任务: speed ≈ 0.4-0.6（中速）

### 3. 评估模型

```bash
cd /mnt/nvme-fast/zbs/fp/MoDE_Diffusion_Policy/experiments/speed_aware/scripts
./eval_speed_aware.sh experiments/speed_aware/logs/checkpoints/last.ckpt 1000
```

---

## 文件结构

```
experiments/speed_aware/
├── models/
│   ├── speed_aware_modedit.py    # SpeedAwareMoDeDiT (独立模型)
│   └── speed_aware_agent.py      # SpeedAwareFlowAgent (独立agent)
├── configs/
│   └── config_speed_aware_calvin.yaml  # 独立配置
├── scripts/
│   ├── train_speed_aware.sh      # 训练脚本
│   └── eval_speed_aware.sh       # 评估脚本
├── logs/                          # 训练日志和checkpoint
└── README.md                      # 本文件
```

依赖的共享模块：
- `experiments/freq_flow/models/speed_aware_head.py`: 核心速度感知头
- `experiments/freq_flow/models/freq_utils.py`: DCT/IDCT工具

---

## 实现细节

### 模型架构

```
Input (observation + goal)
    ↓
MoDeDiT Transformer (标准MoE backbone)
    ↓
Action Hidden States (B, T, embed_dim)
    ↓
AdaptiveSmoothingHead
    ├─ Action Predictor → Raw Actions (B, T, da)
    └─ Speed Predictor → Speed Score (B, 1)
    ↓
Frequency Domain Processing:
    1. DCT(Raw Actions)
    2. Adaptive Filter(Speed) → Filter Weights (B, T, 1)
    3. Filtered Spectrum = Spectrum * Weights
    4. IDCT → Smooth Actions (B, T, da)
    ↓
Output: Smoothed Actions
```

### 速度计算

从真实动作序列计算速度标签：

```python
# 一阶导数（速度）
velocity = actions[:, 1:] - actions[:, :-1]

# 二阶导数（加速度）
acceleration = velocity[:, 1:] - velocity[:, :-1]

# 速度 = 平均加速度幅度
speed = acceleration.abs().mean(dim=(1, 2))

# 归一化到 [0, 1]
speed_normalized = (speed - speed.min()) / (speed.max() - speed.min())
```

**直觉**：
- 高加速度 = 快速运动（push, slide）
- 低加速度 = 精细操作（lift, place）

### 自适应滤波

根据预测速度动态调整频域滤波器：

```python
# 速度 → 截止频率
cutoff = max_smoothing - speed * (max_smoothing - min_smoothing)

# 例如：speed=0.8 (快) → cutoff=0.3 (保留30%频谱)
#       speed=0.2 (慢) → cutoff=0.9 (只保留10%频谱)

# 指数衰减滤波器
freq_indices = [0, 0.1, 0.2, ..., 1.0]  # 归一化频率
filter_weights = exp(-sharpness * max(0, freq_indices - cutoff))
```

### 训练损失

```python
# 主损失：标准 Flow Matching
fm_loss = MSE(velocity_pred, velocity_target)

# 辅助损失：速度预测
speed_label = compute_speed_label(target_actions)
speed_loss = MSE(speed_pred, speed_label)

# 总损失
total_loss = fm_loss + 0.1 * speed_loss
```

权重 0.1 确保辅助损失不会干扰主要的Flow Matching训练。

---

## 配置说明

### 核心参数

在 `config_speed_aware_calvin.yaml` 中：

```yaml
model:
  # Speed-aware loss weight
  speed_loss_weight: 0.1  # 辅助损失权重（建议 0.05-0.2）

  model:
    inner_model:
      # Enable/disable speed-aware head
      enable_speed_head: true  # false = 回退到baseline

      # Smoothing parameters
      min_smoothing: 0.3  # 快速运动最小平滑（保留30%频谱）
      max_smoothing: 0.9  # 慢速运动最大平滑（保留10%频谱）
      smoothing_sharpness: 3.0  # 滤波器陡度（越大越陡）
```

### 调参建议

**如果轨迹太抖动**：
- 增大 `max_smoothing`（0.9 → 0.95）
- 减小 `min_smoothing`（0.3 → 0.2）

**如果轨迹太平滑（反应慢）**：
- 减小 `max_smoothing`（0.9 → 0.8）
- 增大 `min_smoothing`（0.3 → 0.4）

**如果速度预测不准**：
- 增大 `speed_loss_weight`（0.1 → 0.2）

---

## 调试技巧

### 1. 验证速度标签是否合理

训练前先测试：

```python
from experiments.freq_flow.models.speed_aware_head import compute_speed_label
from torch.utils.data import DataLoader

# 加载训练数据
train_loader = ...

# 采样batch并打印速度
for batch in train_loader:
    actions = batch['actions']
    speeds = compute_speed_label(actions)

    for i in range(min(5, len(actions))):
        print(f"Task: {batch['language'][i]}")
        print(f"Speed: {speeds[i]:.3f}")
        print()
```

**预期**：
- "push" 任务 → 高速度 (>0.6)
- "lift" 任务 → 低速度 (<0.4)

### 2. 可视化频域滤波器

在推理时保存滤波器形状：

```python
import matplotlib.pyplot as plt

# 推理时
actions, speed, freq_filter = model.inner_model.out(
    hidden,
    return_speed=True,
    return_filter=True
)

# 绘图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(freq_filter[0].squeeze().cpu())
plt.title(f"Filter (Speed={speed[0].item():.2f})")
plt.xlabel("Frequency")
plt.ylabel("Weight")

plt.subplot(1, 2, 2)
plt.plot(actions[0, :, 0].cpu())  # 第一个维度的动作
plt.title("Filtered Action (dim 0)")
plt.xlabel("Time")
plt.ylabel("Action")

plt.tight_layout()
plt.savefig("speed_aware_debug.png")
```

**预期**：
- 高速度 → 平坦的滤波器（保留所有频率）
- 低速度 → 陡降的滤波器（只保留低频）

### 3. 监控WandB日志

关键诊断指标：

```
train/speed_label_avg vs train/speed_pred_avg
```

如果这两个值相差很大（>0.3），说明速度预测不准，需要：
1. 增大 `speed_loss_weight`
2. 检查数据加载是否正确
3. 确认动作序列没有被截断

---

## 对比实验

### Baseline (无速度感知)

设置 `enable_speed_head: false` 训练baseline：

```bash
# 修改config
vim experiments/speed_aware/configs/config_speed_aware_calvin.yaml
# 改为：enable_speed_head: false

# 训练
./scripts/train_speed_aware.sh
```

### 与 freq_flow 对比

两个实验可以**同时运行**：

```bash
# Terminal 1: freq_flow
cd experiments/freq_flow/scripts
./train_multi_gpu.sh

# Terminal 2: speed_aware
cd experiments/speed_aware/scripts
./train_speed_aware.sh
```

它们的日志和checkpoint完全独立。

---

## 预期效果

### 训练收敛

- **FM Loss**: 应该与baseline相当
- **Speed Loss**: 前几百步快速下降，然后稳定在 0.01-0.05
- **Speed Prediction**: 500步后 `speed_label` 和 `speed_pred` 应该接近

### 评估提升

如果速度感知有效，应该看到：

1. **轨迹平滑度提升**
   - 精细操作（lift, place）更稳定
   - 减少高频抖动

2. **任务成功率**
   - Lift任务提升（需要稳定）
   - Push任务不下降（仍需保持快速）

3. **平均链长度**
   - 如果baseline是 2.5，期望提升到 2.7-2.9

---

## 常见问题

### Q1: Speed loss不下降？

**可能原因**：
1. 速度标签全是相同值 → 检查数据多样性
2. 权重太小 → 增大 `speed_loss_weight`
3. 网络容量不够 → 检查 `SpeedPredictor` 是否被正确初始化

**解决**：
```python
# 在训练循环中添加debug打印
print(f"Speed label range: [{speed_label.min():.3f}, {speed_label.max():.3f}]")
print(f"Speed pred range: [{speed_pred.min():.3f}, {speed_pred.max():.3f}]")
```

### Q2: 效果不如baseline？

**可能原因**：
1. 过度平滑 → 减小 `max_smoothing`
2. 速度预测不准 → 检查速度标签是否合理
3. 超参数不匹配 → 尝试调整 `smoothing_sharpness`

**解决**：先用 `enable_speed_head: false` 确认baseline性能。

### Q3: 如何回退到baseline？

只需设置：
```yaml
enable_speed_head: false
```

模型会自动使用标准线性输出层，完全等价于baseline MoDeDiT。

---

## 下一步优化

如果基础版本work，可以尝试：

### 1. 任务条件速度先验

```python
# 根据任务类型调整速度预测
task_speed_prior = {
    'push': 0.8,
    'lift': 0.3,
    'rotate': 0.5,
}

# 混合预测和先验
final_speed = 0.7 * pred_speed + 0.3 * task_prior[task_type]
```

### 2. Per-dimension 速度

不同动作维度可能需要不同平滑：

```python
# XYZ平移 → 需要平滑
# 末端执行器 → 可以快速切换

speed_per_dim = [slow, slow, slow, slow, slow, slow, fast]  # 7-dim actions
```

### 3. 时间变化速度

一个轨迹内速度可能变化：

```python
# 开始：快速接近 (high speed)
# 中间：精细操作 (low speed)
# 结束：快速撤离 (high speed)

speed_curve = predict_speed_over_time(hidden)  # (B, T, 1)
```

---

## 总结

### ✅ 优势

1. **与Flow Matching完全兼容** - 不干扰主训练流程
2. **物理意义清晰** - 快/慢运动，直观易懂
3. **实现简单** - 单个输出头，易于调试
4. **独立实验** - 不影响 freq_flow 代码

### 🎯 适用场景

- 需要同时执行快速和精细操作的任务
- 轨迹抖动问题严重
- 希望保持baseline性能同时提升稳定性

### 📊 何时使用

**推荐**：
- 作为 freq_flow 的替代方案
- 验证"频域平滑"是否有效的简单基线
- 需要快速迭代实验

**不推荐**：
- 如果baseline已经足够smooth
- 计算资源极度受限（虽然开销很小）

---

## 联系与反馈

实验日期：2025-12-19
实现者：Claude Code

如有问题，检查：
1. WandB日志中的速度曲线
2. `logs/` 目录下的训练日志
3. 速度标签分布是否合理
