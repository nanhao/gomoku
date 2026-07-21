"""MCTS 训练脚本

使用 MCTS 生成的 sample.json 训练双头模型。

用法：
    python src/ai/scripts/mcts_train.py --model model.pth --train-from-dir data/ --epochs 10

设计文档 §4.5 规范实现。
"""

import argparse
import os
import sys
from typing import List

# 确保项目根在 sys.path 中
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch

from ai.registry import get_model, get_trainer
from ai.runners import discover_sample_dirs


# ---------- 参数解析 ----------

def parse_args():
    """解析 CLI 参数"""
    p = argparse.ArgumentParser(
        description="MCTS 训练脚本 — 训练双头模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="初始权重文件路径（以此为起点训练）")
    p.add_argument("--model-name", default=None,
                   help="模型注册名.代数，如 cnn_v8.1。默认从 --model 路径自动解析")
    p.add_argument("--train-from-dir", required=True, nargs='+',
                   help="训练数据目录（sample.json 所在目录），可指定多个")
    p.add_argument("--epochs", type=int, required=True,
                   help="训练轮数")
    p.add_argument("--output", default=None,
                   help="训练后权重保存路径（默认覆盖 --model）")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="训练 batch size（默认 2048）")
    p.add_argument("--lr", type=float, default=0.001,
                   help="初始学习率（默认 0.001）")
    p.add_argument("--l2-reg", type=float, default=1e-4,
                   help="AdamW weight_decay（默认 1e-4）")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="验证集比例（默认 0.1）")
    p.add_argument("--no-shuffle", action="store_true",
                   help="关闭训练数据 shuffle")
    p.add_argument("--device", default="cuda",
                   help="训练设备（默认 cuda）")
    return p.parse_args()


# ---------- 主流程 ----------

def run_train(args):
    """执行训练流水线"""
    # 确定设备
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[设备] {device}")

    # 验证输入目录：支持直接 sample.json 或子目录自动发现
    train_dirs: List[str] = []
    for d in args.train_from_dir:
        sample_path = os.path.join(d, "sample.json")
        if os.path.exists(sample_path):
            # 目录自身包含 sample.json，直接使用
            n_samples = _count_samples(sample_path)
            print(f"[数据] {d} — {n_samples} 条样本")
            train_dirs.append(d)
        else:
            # 目录自身无 sample.json，扫描子目录自动发现
            sub_dirs = discover_sample_dirs(d)
            if sub_dirs:
                total = sum(_count_samples(os.path.join(sd, "sample.json")) for sd in sub_dirs)
                print(f"[数据] {d} — 发现 {len(sub_dirs)} 个子目录，共 {total} 条样本")
                train_dirs.extend(sub_dirs)
            else:
                print(f"[警告] 目录 {d} 中未找到 sample.json（含子目录），跳过")

    if not train_dirs:
        print("[错误] 没有找到有效的训练数据目录")
        sys.exit(1)

    # 加载模型
    print(f"[加载模型] {args.model}")
    model_spec = getattr(args, 'model_name', None)
    if model_spec:
        model = get_model(model_spec)
    else:
        model_path = os.path.normpath(args.model)
        parts = model_path.split(os.sep)
        model_name = None
        model_gen = None
        for i, p in enumerate(parts):
            if p.startswith("gen"):
                model_gen = int(p[3:])
                if i >= 1:
                    model_name = parts[i - 1]
                break
        if model_name and model_gen:
            spec = f"{model_name}.{model_gen}"
            model = get_model(spec)
        else:
            raise ValueError(
                f"无法从路径 '{args.model}' 解析模型名称和代数。"
                f"请使用 --model-name 显式指定"
            )
    model.move_to_device(device)

    # 创建训练器
    trainer_spec = model_spec if model_spec else f"{model_name}.{model_gen}"
    print(f"[训练器] lr={args.lr}, weight_decay={args.l2_reg}, batch_size={args.batch_size}")
    trainer = get_trainer(
        trainer_spec,
        model=model,
        lr=args.lr,
        weight_decay=args.l2_reg,
    )

    # 执行训练
    shuffle = not args.no_shuffle
    print(f"[训练] epochs={args.epochs}, val_split={args.val_split}, shuffle={shuffle}")
    print("=" * 60)

    history = trainer.train_epochs(
        sample_dirs=train_dirs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        shuffle=shuffle,
        verbose=True,
    )

    # 保存模型
    output_path = args.output if args.output else args.model
    print("=" * 60)
    print(f"[保存模型] {output_path}")
    trainer.save_model(output_path)

    # 输出最终损失
    if history:
        final = history[-1]
        print(f"[最终损失] value_loss={final['value_loss']:.4f}, "
              f"policy_loss={final['policy_loss']:.4f}, "
              f"total_loss={final['total_loss']:.4f}")

    return output_path


def _count_samples(sample_path: str) -> int:
    """统计 sample.json 中的样本数（不加载全部内容）"""
    import json
    try:
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


# ---------- 入口 ----------

if __name__ == "__main__":
    args = parse_args()
    run_train(args)
