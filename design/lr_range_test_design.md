# LR Range Test 设计文档

## 1. 需求概述

实现一个独立脚本，用于对指定模型和数据执行 **LR Range Test**：

- 学习率从一个极小值线性递增至一个较大值，每个 **round** 对应一个固定 lr
- 每个 round 包含 **epochs** 个训练 step，全量数据作为一个 batch
- 每个 round 开始前，**重置模型到初始状态**，确保 loss 只反映 lr 的影响
- 每个 round 的 loss 为该 round 内所有 epoch 的**全量平均 loss**
- 最终绘制 lr-loss 曲线图（每个 round 一个数据点）
- 不保存模型

## 2. 接口设计

### 2.1 命令行入口

脚本路径：`temp/lr_range_test.py`

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model` | str | 是 | - | 模型名称，当前支持 `cnn_v2` |
| `--data` | str | 是 | - | 训练数据路径（`.json` 文件或目录） |
| `--rounds` | int | 否 | 50 | lr 递增次数（round 数） |
| `--epochs` | int | 否 | 100 | 每个 round 内的训练 step 数 |
| `--min-lr` | float | 否 | 1e-7 | 起始学习率 |
| `--max-lr` | float | 否 | 1e-1 | 终止学习率 |
| `--output` | str | 否 | `temp/lr_range_test.png` | 输出图片路径 |

示例：
```bash
python temp/lr_range_test.py --model cnn_v2 --data ai/merge_samples/merge_cnn_v1_100.json --rounds 50 --epochs 100
```

### 2.2 核心函数接口

```python
def run_lr_range_test(
    model_name: str,
    data_path: str,
    rounds: int = 50,
    epochs: int = 100,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    device: torch.device = None
) -> Tuple[List[float], List[float]]:
    """执行 LR Range Test

    Args:
        model_name: 模型名称
        data_path: 数据文件或目录路径
        rounds: lr 递增次数
        epochs: 每个 round 内的训练 step 数
        min_lr: 起始学习率
        max_lr: 终止学习率
        device: 训练设备，None 则自动检测

    Returns:
        (lrs, losses): 学习率列表和对应 round 平均 loss 列表，长度均为 rounds
    """

def plot_lr_loss_curve(
    lrs: List[float],
    losses: List[float],
    output_path: str
) -> None:
    """绘制 lr-loss 曲线并保存

    Args:
        lrs: 学习率列表
        losses: loss 列表
        output_path: 图片保存路径
    """

def load_model(model_name: str, device: torch.device) -> BaseModel:
    """根据名称加载模型实例（fresh=True，不加载已有参数）

    Args:
        model_name: 模型名称
        device: 目标设备

    Returns:
        模型实例

    Raises:
        ValueError: 不支持的模型名称
    """

def load_data(data_path: str, sample_loader: SampleLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载训练数据并转换为张量

    Args:
        data_path: 数据文件或目录路径
        sample_loader: SampleLoader 实例

    Returns:
        (inputs, targets, masks) 张量三元组

    Raises:
        FileNotFoundError: 数据路径不存在
    """
```

## 3. 主流程（伪代码）

```python
def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 获取设备
    device = get_device()

    # 3. 创建模型（fresh=True，随机初始化）
    model = load_model(args.model, device)

    # 4. 加载数据
    sample_loader = SampleLoader(device=device, input_converter=get_input_converter(model))
    inputs, targets, masks = load_data(args.data, sample_loader)

    # 5. 保存初始模型状态（每个 round 开始前恢复）
    initial_state = copy.deepcopy(model.network.state_dict())

    # 6. 执行 range test
    criterion = L1Loss(reduction='none')
    lrs = []
    losses = []

    for round_idx in range(args.rounds):
        # 6.1 计算当前 round 的 lr（线性递增）
        lr = args.min_lr + (args.max_lr - args.min_lr) * round_idx / max(1, args.rounds - 1)

        # 6.2 重置模型到初始状态
        model.network.load_state_dict(initial_state)

        # 6.3 创建优化器（当前 round 的固定 lr）
        optimizer = Adam(model.network.parameters(), lr=lr)

        # 6.4 训练 epochs 个 step
        round_loss_sum = 0.0
        for epoch in range(args.epochs):
            model.network.train()
            optimizer.zero_grad()
            outputs = model.network(inputs)

            # 计算 masked loss（与 BaseTrainer 一致）
            loss = criterion(outputs, targets)
            loss = (loss * masks).sum() / masks.sum()

            loss.backward()
            optimizer.step()

            round_loss_sum += loss.item()

        # 6.5 记录该 round 的平均 loss
        avg_loss = round_loss_sum / args.epochs
        lrs.append(lr)
        losses.append(avg_loss)
        print(f"Round {round_idx+1}/{args.rounds}, LR: {lr:.2e}, Avg Loss: {avg_loss:.6f}")

    # 7. 绘制并保存曲线
    plot_lr_loss_curve(lrs, losses, args.output)
    print(f"曲线图已保存: {args.output}")
```

## 4. 测试用例

### 4.1 单元测试

**文件**: `tests/unit/test_lr_range_test.py`

| 用例 | 描述 |
|------|------|
| `test_linear_lr_increase` | 验证 `run_lr_range_test` 返回的 lr 列表严格线性递增 |
| `test_losses_length_matches_rounds` | 验证返回的 losses 长度等于 rounds |
| `test_plot_lr_loss_curve_creates_file` | 验证 `plot_lr_loss_curve` 能在临时目录生成 png 文件 |
| `test_load_model_cnn_v2` | 验证 `load_model("cnn_v2")` 返回 CNNV2Model 实例且 network 不为 None |
| `test_load_data_from_merge_file` | 验证 `load_data` 能从 merge json 文件加载并返回三个张量 |
| `test_load_data_from_directory` | 验证 `load_data` 能从目录扫描并加载数据 |
| `test_model_reset_each_round` | 验证每个 round 开始时模型参数被重置到初始状态 |

### 4.2 集成测试

**文件**: `tests/integration/test_lr_range_test.py`

| 用例 | 描述 |
|------|------|
| `test_end_to_end_cnn_v2` | 端到端测试：CNNV2Model + 模拟数据，验证 lrs/losses 非空且长度正确 |
| `test_invalid_model_name` | 验证传入不支持模型名称时抛出 ValueError |
| `test_nonexistent_data_path` | 验证传入不存在的数据路径时抛出 FileNotFoundError |

### 4.3 边界条件

- `rounds=1`：只执行一个 round，lr = min_lr
- `epochs=1`：每个 round 只训练一个 step
- `min_lr == max_lr`：所有 round 使用相同 lr
