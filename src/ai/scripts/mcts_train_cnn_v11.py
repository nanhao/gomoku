"""CNN v11 样本缓冲区驱动训练脚本 CLI 入口

从持续运行的样本缓冲区中随机抽取 mini-batch 进行增量训练，
每 N 步产生 checkpoint，通过 C++ MCTS 对战评估决定是否晋升。

用法示例：
    python -m ai.scripts.mcts_train_cnn_v11 --model cnn_v11.1 --buffer-dir sample_buffer
"""
import argparse
import os
import signal
import sys

# 确保项目根目录和 cpp/python 在 sys.path 中，支持直接执行脚本
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_src_dir = os.path.join(_proj_root, 'src')
_cpp_py_dir = os.path.join(_proj_root, 'cpp', 'python')
for _p in (_proj_root, _src_dir, _cpp_py_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ai.cnn_v11_train_loop import CnnV11TrainLoop
from ai.scripts.mcts_config import MctsSelfPlayConfig


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器

    Returns:
        配置好的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        description="CNN v11 样本缓冲区驱动训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m ai.scripts.mcts_train_cnn_v11 --model cnn_v11.1
  python -m ai.scripts.mcts_train_cnn_v11 --model cnn_v11.3 --batch-size 256 --eval-games 30
""",
    )

    # 必填参数
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="初始模型 spec（如 cnn_v11.1），同时作为初始 best",
    )

    # 可选参数
    parser.add_argument(
        "--buffer-dir",
        type=str,
        default="sample_buffer",
        help="样本缓冲区目录（默认: sample_buffer）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="每个 mini-batch 的样本数（默认: 512）",
    )
    parser.add_argument(
        "--updates-per-checkpoint",
        type=int,
        default=50,
        help="多少步后触发 checkpoint 评估（默认: 50）",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="checkpoint 评估对战局数（默认: 50）",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="连续评估失败上限（默认: 3）",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="评估时每步 MCTS 模拟次数（默认: 800）",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="PUCT 探索系数（默认: 2.0）",
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
        help="训练设备（默认: cuda）",
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

    if args.updates_per_checkpoint <= 0:
        errors.append("--updates-per-checkpoint 必须 > 0")

    if args.eval_games < 0:
        errors.append("--eval-games 必须 >= 0")

    if args.max_failures <= 0:
        errors.append("--max-failures 必须 > 0")

    if args.simulations <= 0:
        errors.append("--simulations 必须 > 0")

    if args.lr <= 0:
        errors.append("--lr 必须 > 0")

    if args.weight_decay < 0:
        errors.append("--weight-decay 必须 >= 0")

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

    # 构建 MCTS 配置
    mcts_config = MctsSelfPlayConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        temp_steps=2,  # 评估时固定前 2 步采样，之后贪婪
        temperature=1.0,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.03,
        virtual_loss=3.0,
        use_prior=True,
    )

    # ---------- 打印所有配置 ----------
    print("=" * 60)
    print("CNN v11 样本缓冲区训练脚本")
    print("=" * 60)
    print()
    print("[训练参数]")
    print(f"  初始模型:        {args.model}")
    print(f"  缓冲区目录:      {args.buffer_dir}")
    print(f"  batch 大小:      {args.batch_size}")
    print(f"  每 checkpoint:   {args.updates_per_checkpoint} 步")
    print(f"  评估局数:        {args.eval_games}")
    print(f"  最大连续失败:    {args.max_failures}")
    print(f"  学习率:          {args.lr}")
    print(f"  权重衰减:        {args.weight_decay}")
    print(f"  训练设备:        {args.device}")
    print()
    print("[MCTS 评估参数]")
    print(f"  simulations:     {mcts_config.simulations}")
    print(f"  c_puct:          {mcts_config.c_puct}")
    print(f"  temp_steps:      {mcts_config.temp_steps}")
    print(f"  temperature:     {mcts_config.temperature}")
    print(f"  dirichlet_eps:   {mcts_config.dirichlet_epsilon}")
    print(f"  virtual_loss:    {mcts_config.virtual_loss}")
    print()
    print("[晋升规则]")
    print(f"  晋升阈值:        >= 55% 胜率")
    print(f"  停止条件:        连续 {args.max_failures} 次 < 55%")
    print()
    print("=" * 60)
    print()

    # 注册 SIGINT 信号处理器（Ctrl+C 直接终止，不等待）
    original_handler = signal.signal(signal.SIGINT, lambda signum, frame: os._exit(0))

    # 创建训练循环
    loop = CnnV11TrainLoop(
        model_spec=args.model,
        buffer_dir=args.buffer_dir,
        batch_size=args.batch_size,
        updates_per_checkpoint=args.updates_per_checkpoint,
        eval_games=args.eval_games,
        max_consecutive_failures=args.max_failures,
        mcts_config=mcts_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    # 执行训练
    result = loop.run()

    # 恢复原始信号处理器
    signal.signal(signal.SIGINT, original_handler)

    # 打印结果
    if result.stopped_by == "promoted":
        print(f"\n[晋升] checkpoint 胜率 {result.best_win_rate:.1%}，"
              f"晋升为 {result.promoted_to}")
    elif result.stopped_by == "max_failures":
        print(f"\n[停止] 连续 {args.max_failures} 次评估失败，"
              f"训练终止。最高胜率: {result.best_win_rate:.1%}")
    else:
        print(f"\n[{result.stopped_by}] 训练被中断。"
              f"已执行 {result.total_updates} 次 mini-batch，"
              f"评估 {result.checkpoints_evaluated} 个 checkpoint")

    print(f"总 mini-batch: {result.total_updates}")
    print(f"评估 checkpoint: {result.checkpoints_evaluated}")
    print(f"最佳胜率: {result.best_win_rate:.1%}")


if __name__ == "__main__":
    main()
