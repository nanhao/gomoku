"""LR Range Test 脚本

对指定模型和数据执行 LR Range Test：
- 学习率从 min_lr 线性递增至 max_lr
- 全量数据作为一个 batch，每个 step 训练一次
- 记录每个 step 的学习率与全量平均 loss
- 绘制 lr-loss 曲线图
- 不保存模型
"""

import argparse
import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai.base_model import BaseModel
from ai.sample_loader import SampleLoader


# 模型注册表：名称 -> 模型类
MODEL_REGISTRY = {
    "cnn_v2": None,  # 延迟导入，在 load_model 中设置
}


def _get_model_registry():
    """获取模型注册表（延迟导入）"""
    from ai.models.cnn_v2 import CNNV2Model
    return {
        "cnn_v2": CNNV2Model,
    }


def load_model(model_name: str, device: torch.device) -> BaseModel:
    """根据名称加载模型实例（fresh=True，不加载已有参数）

    Args:
        model_name: 模型名称
        device: 目标设备

    Returns:
        模型实例（已移至指定设备）

    Raises:
        ValueError: 不支持的模型名称
    """
    registry = _get_model_registry()
    if model_name not in registry:
        raise ValueError(
            f"不支持的模型名称: '{model_name}'，"
            f"可选值: {list(registry.keys())}"
        )
    model_cls = registry[model_name]
    model = model_cls(fresh=True)
    if model.network is not None:
        model.network.to(device)
    return model


def _get_input_converter(model: BaseModel):
    """获取模型对应的输入转换函数

    Args:
        model: 模型实例

    Returns:
        输入转换函数，或 None（FC 模型无需转换）
    """
    # CNN 模型有 _vector_to_tensor 方法
    if hasattr(model, '_vector_to_tensor'):
        def convert_inputs(inputs: list) -> torch.Tensor:
            """将批量 226 维向量转换为 CNN 输入张量 (N, 3, 15, 15)"""
            batch = [model._vector_to_tensor(v) for v in inputs]
            return torch.cat(batch, dim=0)
        return convert_inputs
    return None


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
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")

    if os.path.isfile(data_path):
        # 单个 json 文件
        sample_data = sample_loader.load_merge_file(data_path)
    else:
        # 目录：扫描并加载
        sample_data = sample_loader.discover_and_load(data_path)

    if not sample_data:
        raise ValueError(f"数据为空: {data_path}")

    return sample_loader.prepare_tensors(sample_data)


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
    # 获取设备
    if device is None:
        from ai.base_trainer import get_device
        device = get_device()

    # 创建模型
    model = load_model(model_name, device)

    # 加载数据
    sample_loader = SampleLoader(
        device=device,
        input_converter=_get_input_converter(model)
    )
    inputs, targets, masks = load_data(data_path, sample_loader)

    # 创建优化器和损失函数
    optimizer = optim.Adam(model.network.parameters(), lr=min_lr)
    criterion = nn.L1Loss(reduction='none')

    # 执行 range test
    lrs = []
    losses = []

    for step in range(epochs):
        # 线性递增 lr
        lr = min_lr + (max_lr - min_lr) * step / max(1, epochs - 1)
        optimizer.param_groups[0]['lr'] = lr

        # 前向传播
        model.network.train()
        optimizer.zero_grad()
        outputs = model.network(inputs)

        # 计算 masked loss（与 BaseTrainer._masked_loss 一致）
        loss = criterion(outputs, targets)
        loss = (loss * masks).sum() / masks.sum()

        # 反向传播与参数更新
        loss.backward()
        optimizer.step()

        # 记录
        lrs.append(lr)
        losses.append(loss.item())

    return lrs, losses


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
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lrs, losses, linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('LR Range Test')
    ax.grid(True, alpha=0.3)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LR Range Test')
    parser.add_argument('--model', type=str, required=True,
                        help='模型名称 (cnn_v2)')
    parser.add_argument('--data', type=str, required=True,
                        help='训练数据路径 (.json 文件或目录)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='总 step 数 (默认: 100)')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                        help='起始学习率 (默认: 1e-7)')
    parser.add_argument('--max-lr', type=float, default=1e-1,
                        help='终止学习率 (默认: 1e-1)')
    parser.add_argument('--output', type=str, default='temp/lr_range_test.png',
                        help='输出图片路径 (默认: temp/lr_range_test.png)')
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    print(f"LR Range Test: model={args.model}, data={args.data}")
    print(f"  epochs={args.epochs}, min_lr={args.min_lr:.2e}, max_lr={args.max_lr:.2e}")

    lrs, losses = run_lr_range_test(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        min_lr=args.min_lr,
        max_lr=args.max_lr
    )

    # 打印每个 step 的结果
    for i, (lr, loss) in enumerate(zip(lrs, losses)):
        print(f"Step {i+1}/{args.epochs}, LR: {lr:.2e}, Loss: {loss:.6f}")

    # 绘制并保存曲线
    plot_lr_loss_curve(lrs, losses, args.output)
    print(f"曲线图已保存: {args.output}")


if __name__ == '__main__':
    main()
