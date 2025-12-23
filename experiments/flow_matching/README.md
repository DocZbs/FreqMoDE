# Flow Matching for MoDE Diffusion Policy

将扩散模型替换为Flow Matching，实现更快的采样速度。

## 模型架构

**主干网络**: MoDeDiT (Mixture of Experts Diffusion Transformer)
- 12层 Transformer, 4专家, top-k=2
- ResNet50 (2048维) + CLIP ViT-B/32 (512维)
- 参数量: ~307M

## 核心差异

| 方面 | 扩散模型 | Flow Matching |
|------|---------|---------------|
| 训练 | 预测噪声 | 预测速度场 |
| 损失 | `||ε_θ - ε||²` | `||v_θ - u_t||²` |
| 采样 | 10-50步 | 5-20步 |
| 速度 | 基准 | ~2x faster |

## 快速开始

### 1. 测试
```bash
python experiments/flow_matching/test_flow_matching.py
```

### 2. 训练
```bash
# 基础训练
python experiments/flow_matching/training/train_flow_calvin.py

# 自定义配置
python experiments/flow_matching/training/train_flow_calvin.py \
    flow_type=conditional \
    sampling_method=heun \
    num_sampling_steps=10 \
    trainer.devices=4
```

## 配置选项

### Flow类型
- `conditional`: 条件流匹配 (OT-CFM, 推荐)
- `rectified`: 修正流 (更简单)

### 采样方法
- `euler`: 一阶，最快
- `heun`: 二阶，推荐
- `rk4`: 四阶，最准确

### 关键参数
```yaml
flow_type: conditional
sampling_method: heun
num_sampling_steps: 10
batch_size: 128
learning_rate: 1e-4
```

## 日志

- **WandB**: `MoDE_FlowMatching` project
- **本地日志**: `logs/{date}/{time}/seed_{seed}/training.log`
- **Checkpoints**: `logs/checkpoints/`

## 文件结构

```
experiments/flow_matching/
├── models/
│   ├── flow_core/           # 核心算法
│   ├── flow_wrappers/       # 模型包装器
│   └── flow_agent.py        # Agent类
├── configs/                 # 配置文件
├── training/                # 训练脚本
├── test_flow_matching.py    # 单元测试
└── run_experiment.sh        # 快速启动
```

## 引用

```bibtex
@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  booktitle={ICLR},
  year={2023}
}
```
