"""CNN v11 off-policy 监督训练脚本 CLI 入口

对静态样本集（如 renjunet_samples 人类棋谱）做多 epoch 全量 mini-batch 监督训练：
加载指定 gen 模型 → 全量遍历样本训练 → 保存 gen{N+1}（pth + TorchScript）。

loss 曲线默认每 50 次 update 记录一次窗口均值（--log-every-updates 可调，
0 恢复 epoch 级记录），X 轴为累计 update 序号；
val_total_loss 仍每 epoch 评估一次。

用法示例：
    python -m ai.scripts.mcts_train_cnn_v11_offpolicy
    python -m ai.scripts.mcts_train_cnn_v11_offpolicy --model cnn_v11.1
    python -m ai.scripts.mcts_train_cnn_v11_offpolicy --model cnn_v11.3 --batch-size 1024 --epochs 50

--model 为可选参数。不指定时自动识别 cnn_v11 的最高代数。
Ctrl+C 不注册 os._exit 级 SIGINT handler——由 train_epochs 优雅捕获
（先生成曲线图），中断后仍保存模型。
"""
import argparse
import os
import sys

# 确保项目根目录和 cpp/python 在 sys.path 中，支持直接执行脚本
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_src_dir = os.path.join(_proj_root, 'src')
_cpp_py_dir = os.path.join(_proj_root, 'cpp', 'python')
for _p in (_proj_root, _src_dir, _cpp_py_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ai.off_policy_train_loop import OffPolicyTrainLoop


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器

    Returns:
        配置好的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        description="CNN v11 off-policy 监督训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m ai.scripts.mcts_train_cnn_v11_offpolicy
  python -m ai.scripts.mcts_train_cnn_v11_offpolicy --model cnn_v11.1
  python -m ai.scripts.mcts_train_cnn_v11_offpolicy --model cnn_v11.3 --batch-size 1024 --epochs 50
""",
    )

    # 模型参数（可选，不指定时自动识别最高代数）
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="初始模型 spec（如 cnn_v11.1）。不指定时自动识别 cnn_v11 最高代数",
    )

    # 数据与训练参数
    parser.add_argument(
        "--data-dir",
        type=str,
        default="renjunet_samples",
        help="样本目录（chunk JSON 或 sample.json 格式，默认: renjunet_samples）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="每个 mini-batch 的样本数（默认: 2048）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练 epoch 数（默认: 100）",
    )
    parser.add_argument(
        "--validate-set",
        type=float,
        default=0.1,
        help="验证集比例，值域 [0, 1)（默认: 0.1）",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学习率（默认: 0.001）",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW 权重衰减（默认: 1e-4）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备（默认: cuda，cuda 不可用时自动回退 cpu 并警告）",
    )
    parser.add_argument(
        "--log-every-updates",
        type=int,
        default=50,
        help="每 N 次 update 记录一次 loss 用于高密度曲线（默认: 50，0 为仅 epoch 级）",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """校验命令行参数

    Args:
        args: 解析后的参数命名空间

    Raises:
        SystemExit: 参数校验失败
    """
    errors = []

    if args.batch_size <= 0:
        errors.append("--batch-size 必须 > 0")

    if args.epochs <= 0:
        errors.append("--epochs 必须 > 0")

    if args.validate_set < 0:
        errors.append("--validate-set 必须 >= 0")

    if args.validate_set >= 1:
        errors.append("--validate-set 必须 < 1")

    if args.lr <= 0:
        errors.append("--lr 必须 > 0")

    if args.weight_decay < 0:
        errors.append("--weight-decay 必须 >= 0")

    if args.log_every_updates < 0:
        errors.append("--log-every-updates 必须 >= 0")

    if errors:
        print("参数校验失败:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI 入口函数"""
    parser = build_parser()
    args = parser.parse_args()

    # 参数校验
    validate_args(args)

    # 自动识别 model（未指定时查找 cnn_v11 最高代数）
    if args.model is None:
        from ai.registry import get_highest_gen
        highest_gen = get_highest_gen("cnn_v11")
        args.model = f"cnn_v11.{highest_gen}"
        print(f"[自动识别] 未指定 --model，使用 cnn_v11 最高代数: {args.model}")

    # 数据目录检查（先于模型加载，避免无意义加载）
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    # ---------- 打印完整配置 ----------
    print("=" * 60)
    print("CNN v11 off-policy 监督训练脚本")
    print("=" * 60)
    print()
    print("[训练参数]")
    print(f"  初始模型:        {args.model}")
    print(f"  数据目录:        {args.data_dir}")
    print(f"  batch 大小:      {args.batch_size}")
    print(f"  epoch 数:        {args.epochs}")
    print(f"  验证集比例:      {args.validate_set}")
    print(f"  学习率:          {args.lr}")
    print(f"  权重衰减:        {args.weight_decay}")
    print(f"  训练设备:        {args.device}")
    print(f"  记录频率:        每 {args.log_every_updates} 次 update"
          f"（--log-every-updates，0 为仅 epoch 级）")
    print()
    print("=" * 60)

    # 创建训练循环（模型不存在等错误在此捕获）
    try:
        loop = OffPolicyTrainLoop(
            model_spec=args.model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            val_split=args.validate_set,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            log_every_updates=args.log_every_updates,
        )
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)

    # 执行训练（Ctrl+C 由 train_epochs 优雅捕获，中断后仍保存模型）
    result = loop.run()

    # 打印结果摘要（曲线图路径已由 train_epochs 打印，不重复）
    print(f"\n[结果] 已保存: {result.saved_spec}（gen{result.saved_gen}）")
    print(f"完成 epoch: {result.total_epochs}/{result.requested_epochs}")
    if result.final_val_loss is not None:
        print(f"最终 total_loss: {result.final_total_loss:.4f}"
              f"，val_loss: {result.final_val_loss:.4f}")
    else:
        print(f"最终 total_loss: {result.final_total_loss:.4f}")
    if result.interrupted:
        print("[中断] 训练被 Ctrl+C 中断，已完成部分训练并保存模型")


if __name__ == "__main__":
    main()
