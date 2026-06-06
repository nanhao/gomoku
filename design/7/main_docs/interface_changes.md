# 接口变更详细设计

---

## 1. 变更模块：ai/base_trainer.py

### 1.1 类：BaseTrainer

#### 构造器签名变更

**变更前**：

```
BaseTrainer(model=None, fresh=False, gen=None, device=None,
            loss_logger=None,
            lr_scheduler_type=None, lr_step_size=500,
            lr_gamma=0.9, lr_patience=100,
            lr_factor=0.5, weight_decay=None)
```

**变更后**：

```
BaseTrainer(model=None, fresh=False, gen=None, device=None,
            loss_logger=None,
            lr_scheduler_type=None, lr_step_size=500,
            lr_gamma=0.9, lr_patience=100,
            lr_factor=0.5, weight_decay=None, lr=None)
```

#### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr` | `float` 或 `None` | `None` | 初始学习率。`None` 时使用类属性 `DEFAULT_LR = 0.001`；显式传入时必须为正数 |

#### 新增异常

| 异常类型 | 抛出条件 |
|----------|---------|
| `ValueError` | `lr` 不为 `None` 且 `lr ≤ 0` |

#### 兼容性

- `lr=None`（默认）：行为完全不变，优化器使用 `DEFAULT_LR = 0.001`
- 所有现有调用 `BaseTrainer(...)` / `get_trainer(...)` 不传 `lr` 时行为不变
- 子类构造器无需修改，`**kwargs` 透传

---

## 2. 变更模块：ai/train_main.py

### 2.1 函数：main()

无签名变更。内部新增参数解析与校验逻辑。

#### 新增 CLI 参数

| 长参数 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--lr` | `float` | `0.001` | 初始学习率，必须为正数（> 0） |

#### 参数校验

- `type=float`：argparse 自动拒绝非数字字符串
- 后置校验 `args.lr > 0`：不满足时调用 `parser.error("--lr 必须为正数")` 退出（退出码 2）

#### 传递方式

`args.lr` 无条件加入 kwargs 字典（与 `--lr-scheduler` 是否存在解耦），通过 `get_trainer(model, fresh=fresh, **kwargs)` 传递给训练器构造器。

```
# lr 始终传递，scheduler 参数仅当 --lr-scheduler 指定时才追加
kwargs = {'lr': args.lr}
if args.lr_scheduler:
    kwargs.update({
        'lr_scheduler_type': args.lr_scheduler,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'lr_patience': args.lr_patience,
        'lr_factor': args.lr_factor
    })
trainer = get_trainer(args.model, fresh=args.fresh, **kwargs)
```

---

## 3. 变更模块：ai/runners.py

### 3.1 函数：run_train_from_dir()

签名不变，函数体仅在训练参数打印区新增一行。

#### 新增输出

在"训练参数"打印块末尾新增：

```
初始学习率 (lr): {lr_value}
```

`lr_value` 来源：`trainer.optimizer.param_groups[0]['lr']`。

---

## 4. 未变更模块

以下模块无需修改：

| 模块 | 原因 |
|------|------|
| `ai/registry.py` | `get_trainer` 已将 `**kwargs` 透传给训练器构造器，无需改代码 |
| `ai/models/*/trainer.py`（fc_v1, cnn_v1, cnn_v2, cnn_v3） | 继承 `BaseTrainer`，构造器中 `**kwargs` 自动透传 |
| `ai/iterative_train.py` | 不传 `lr` 参数，使用默认值，行为不变 |
