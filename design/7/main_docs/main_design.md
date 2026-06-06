# 主设计文档：train_main.py 支持指定初始学习率

| 字段 | 值 |
|------|-----|
| intention_id | 7 |
| PRD | requirements/7/PRD.md |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

本次变更在 `train_main.py` 新增 `--lr` 命令行参数，并打通从 CLI → 训练器构造 → 优化器初始化的完整参数传递链路，使用户可通过命令行指定初始学习率。

**修改范围**：3 个已有文件，无新增文件。

交互图见 [lr_param_flow.puml](lr_param_flow.puml)。

---

## 2. 模块变更

### 2.1 变更清单

| 文件 | 变更类别 | 说明 |
|------|---------|------|
| `ai/base_trainer.py` | 接口扩展 | `BaseTrainer.__init__` 新增 `lr` 参数 |
| `ai/train_main.py` | CLI 扩展 | 新增 `--lr` 参数，解析校验后传递给训练器 |
| `ai/runners.py` | 日志扩展 | `run_train_from_dir` 打印实际初始学习率 |

### 2.2 BaseTrainer 构造器变更

**当前签名**：

```
BaseTrainer(model, fresh, gen, device, loss_logger,
            lr_scheduler_type, lr_step_size, lr_gamma,
            lr_patience, lr_factor, weight_decay)
```

**变更后签名**：

```
BaseTrainer(model, fresh, gen, device, loss_logger,
            lr_scheduler_type, lr_step_size, lr_gamma,
            lr_patience, lr_factor, weight_decay, lr)
```

| 新参数 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `lr` | `float` | `None` | 初始学习率，None 时使用类属性 `DEFAULT_LR`（0.001） |

**行为**：
- `lr=None`（默认）：使用 `self.DEFAULT_LR`，与当前行为完全一致
- `lr` 显式传入：替代 `DEFAULT_LR`，用于创建 AdamW 优化器
- 校验：若 `lr` 不是 None 但 ≤ 0，在构造器中抛 `ValueError`

### 2.3 train_main.py CLI 变更

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr` | `float` | `0.001` | 初始学习率，必须为正数 |

**校验策略**：参数解析通过 argparse 的 `type=float` 完成类型转换后，在 `main()` 中执行后置校验 `lr > 0`。不满足时调用 `parser.error()` 退出。

**传递路径**：`args.lr` 无条件加入 kwargs（与 scheduler 是否存在解耦），通过 `get_trainer(..., **kwargs)` 传递给构造器。

```
kwargs = {'lr': args.lr}                    # lr 始终传递，无条件
if args.lr_scheduler:
    kwargs.update({lr_scheduler_type: ...,   # scheduler 参数条件追加
                   lr_step_size: ..., ...})
trainer = get_trainer(args.model, fresh=args.fresh, **kwargs)
```

### 2.4 runners.py 日志变更

`run_train_from_dir` 的训练参数打印区域新增一行：

```
初始学习率 (lr): {lr_value}
```

其中 `lr_value` 从 `trainer.optimizer.param_groups[0]['lr']` 读取（保证训练信息与实际一致）。

---

## 3. 数据流

```
train_main.py CLI
    │
    │  argparse 解析 --lr (float, default=0.001)
    │  main() 后置校验 lr > 0
    │
    ▼
kwargs = {'lr': args.lr}                       ← 无条件加入
    │  if args.lr_scheduler: kwargs.update({...})  ← scheduler 条件追加
    │
    │  get_trainer(args.model, fresh=args.fresh, **kwargs)
    │  → registry.py: 透传 **kwargs 到具体 Trainer 类构造器
    │
    ▼
BaseTrainer.__init__(..., lr=lr, ...)
    │
    │  lr = lr if lr is not None else self.DEFAULT_LR
    │  optimizer = AdamW(params, lr=lr, ...)
    │
    ▼
run_train_from_dir(args, trainer)
    │
    │  读取 trainer.optimizer.param_groups[0]['lr']
    │  打印 "初始学习率 (lr): {value}"
    │
    ▼
trainer.train_auto(...) → 训练使用指定初始学习率
```

---

## 4. 与现有参数交互

### 4.1 学习率调度器

`--lr` 与 `--lr-scheduler` 独立。当两者同时指定时，学习率从 `--lr` 指定的值出发，调度器从此起点开始衰减。

```
--lr 0.01 --lr-scheduler step --lr-step-size 100 --lr-gamma 0.5
→ epoch 1~100: lr=0.01
→ epoch 101~200: lr=0.005
→ epoch 201~300: lr=0.0025
→ ...
```

### 4.2 默认值一致性

不指定 `--lr` 时，CLI 默认值 0.001 与 `BaseTrainer.DEFAULT_LR` 一致。即使 CLI 传递 0.001，训练器内部仍走 `lr is not None` 分支，结果等价。

---

## 5. 向后兼容

| 场景 | 兼容性 |
|------|--------|
| 不传 `--lr` 的现有调用 | 行为不变，优化器使用 0.001 |
| `get_trainer()` 的 Python API 调用（如 `iterative_train.py`） | 不传 `lr` 参数时行为不变 |
| `BaseTrainer` 子类（fc_v1, cnn_v1, cnn_v2, cnn_v3） | 无需修改，继承基类行为 |

---

## 6. 错误处理

| 场景 | 处理方式 |
|------|---------|
| `--lr 0` | `main()` 中 `parser.error("--lr 必须为正数")`，退出码 2 |
| `--lr -0.01` | 同上 |
| `--lr abc` | argparse `type=float` 自动报错，退出码 2 |
| 代码中传入 `lr=0`（非 CLI） | `BaseTrainer.__init__` 中抛 `ValueError` |
