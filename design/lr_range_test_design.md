# LR Range Test 设计文档

## 1. 需求概述

实现一个独立脚本，用于对指定模型和数据执行 **LR Range Test**：

- 学习率从一个极小值线性递增至一个较大值
- 全量数据作为一个 batch，每个 step 训练一次并更新参数
- 记录每个 step 的学习率与全量平均 loss
- 最终绘制 lr-loss 曲线图
- 不保存模型

## 2. 接口设计

### 2.1 命令行入口

脚本路径：`temp/lr_range_test.py`

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--model` | str | 是 | - | 模型名称，当前支持 `cnn_v2` |
| `--data` | str | 是 | - | 训练数据路径（`.json` 文件或目录） |
| `--epochs` | int | 否 | 100 | 总 step 数 |
| `--min-lr` | float | 否 | 1e-7 | 起始学习率 |
| `--max-lr` | float | 否 | 1e-1 | 终止学习率 |
| `--output` | str | 否 | `temp/lr_range_test.png` | 输出图片路径 |

示例：
```bash
python temp/lr_range_test.py --model cnn_v2 --data ai/merge_samples/merge_cnn_v1_100.json --epochs 100
```

### 2.2 核心类/函数接口

```python
def run_lr_range_test(
    model_name: str,
    data_path: str,
    epochs: int = 100,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    device: torch.device = None
) -> Tuple[List[float], List[float]]:
    """执行 LR Range Test

    Args:
        model_name: 模型名称
        data_path: 数据文件或目录路径
        epochs: 总 step 数
        min_lr: 起始学习率
        max_lr: 终止学习率
        device: 训练设备，None 则自动检测

    Returns:
        (lrs, losses): 学习率列表和对应 loss 列表
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
    """

def load_data(data_path: str, sample_loader: SampleLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载训练数据并转换为张量

    Args:
        data_path: 数据文件或目录路径
        sample_loader: SampleLoader 实例

    Returns:
        (inputs, targets, masks) 张量三元组
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
    model = load_model(args.model_name, device)

    # 4. 加载数据
    sample_loader = SampleLoader(device=device, input_converter=get_input_converter(model))
    inputs, targets, masks = load_data(args.data, sample_loader)

    # 5. 创建优化器（初始 lr = min_lr）
    optimizer = Adam(model.network.parameters(), lr=args.min_lr)
    criterion = L1Loss(reduction='none')

    # 6. 执行 range test
    lrs = []
    losses = []
    for step in range(args.epochs):
        # 6.1 计算当前 step 的 lr（线性递增）
        lr = args.min_lr + (args.max_lr - args.min_lr) * step / max(1, args.epochs - 1)
        optimizer.param_groups[0]['lr'] = lr

        # 6.2 前向传播
        model.network.train()
        optimizer.zero_grad()
        outputs = model.network(inputs)

        # 6.3 计算 masked loss（与 BaseTrainer 一致）
        loss = criterion(outputs, targets)
        loss = (loss * masks).sum() / masks.sum()

        # 6.4 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 6.5 记录
        lrs.append(lr)
        losses.append(loss.item())
        print(f"Step {step+1}/{args.epochs}, LR: {lr:.2e}, Loss: {loss.item():.6f}")

    # 7. 绘制并保存曲线
    plot_lr_loss_curve(lrs, losses, args.output)
    print(f"曲线图已保存: {args.output}")
```

## 4. 测试用例

### 4.1 单元测试

**文件**: `tests/unit/test_lr_range_test.py`

| 用例 | 描述 |
|------|------|
| `test_linear_lr_increase` | 验证 `run_lr_range_test` 返回的 lr 列表是否严格线性递增 |
| `test_losses_length_matches_epochs` | 验证返回的 losses 长度等于 epochs |
| `test_plot_lr_loss_curve_creates_file` | 验证 `plot_lr_loss_curve` 能在临时目录生成 png 文件 |
| `test_load_model_cnn_v2` | 验证 `load_model("cnn_v2")` 返回 CNNV2Model 实例且 network 不为 None |
| `test_load_data_from_merge_file` | 验证 `load_data` 能从 merge json 文件加载并返回三个张量 |
| `test_load_data_from_directory` | 验证 `load_data` 能从目录扫描并加载数据 |

### 4.2 集成测试

**文件**: `tests/integration/test_lr_range_test.py`

| 用例 | 描述 |
|------|------|
| `test_end_to_end_cnn_v2` | 端到端测试：使用真实 CNNV2Model + 模拟数据（内存中构造）运行完整流程，验证返回的 lrs/losses 非空且长度正确，输出文件被创建 |
| `test_invalid_model_name` | 验证传入不支持模型名称时抛出 ValueError |
| `test_nonexistent_data_path` | 验证传入不存在的数据路径时抛出 FileNotFoundError |

### 4.3 边界条件

- `epochs=1`：只执行一个 step，lr = min_lr
- `min_lr == max_lr`：所有 step 使用相同 lr
- 空数据：应抛出异常（由 SampleLoader 或数据加载逻辑保证）
