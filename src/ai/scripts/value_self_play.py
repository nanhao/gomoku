"""价值头自我对弈脚本

使用 ValueOnlyAiEngine 生成训练样本，所有局合并输出一个 sample.json。

用法：
    python src/ai/scripts/value_self_play.py --model model.pth --games 100

设计文档 design/44/main_docs/design.md 规范实现。
"""

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

# 确保项目根在 sys.path 中
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch

from ai.scripts.value_self_play_worker import ValueSelfPlayWorker, ValueSelfPlayConfig
from ai.registry import get_model


# ---------- 常量 ----------

# 周期保存间隔（局数），不可通过 CLI 配置
_SAVE_INTERVAL = 50

# 中断标志位，由信号处理器设置，主循环检查
_interrupted = False


def _build_hyperparams(args) -> dict:
    """从 CLI args 构建超参数字典"""
    return {
        "temperature": args.temperature,
        "temp_steps": args.temp_steps,
        "temp_side": args.temp_side,
        "top_k": args.top_k,
        "seed": args.temp_seed,
    }


def _parse_model_spec_from_path(model_path: str) -> str:
    """从权重文件路径解析模型 spec 字符串（如 cnn_v10.1）

    解析逻辑与 ValueSelfPlayWorker._load_model 中一致。
    """
    norm_path = os.path.normpath(model_path)
    parts = norm_path.split(os.sep)
    model_name = None
    model_gen = None
    for i, p in enumerate(parts):
        if p.startswith("gen"):
            model_gen = int(p[3:])
            if i >= 1:
                model_name = parts[i - 1]
            break
    if model_name and model_gen is not None:
        return f"{model_name}.{model_gen}"
    raise ValueError(
        f"无法从路径 '{model_path}' 解析模型名称和代数。"
        f"请使用 --model-name 显式指定，如 --model-name cnn_v10.1"
    )


def _signal_handler(signum, frame):
    """SIGINT 信号处理器"""
    global _interrupted
    _interrupted = True
    print("\n[中断] 收到 Ctrl+C，正在保存数据...")


def _load_existing_samples(output_root: str) -> List[dict]:
    """加载输出目录下已有的 sample.json"""
    sample_path = os.path.join(output_root, "sample.json")
    if not os.path.exists(sample_path):
        return []
    try:
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        print(f"[警告] sample.json 格式异常（非数组），将从头开始")
        return []
    except (json.JSONDecodeError, IOError) as e:
        print(f"[警告] 无法读取已有 sample.json: {e}，将从头开始")
        return []


def _save_progress(all_samples: List[dict], output_root: str,
                   args, completed_games: int) -> None:
    """原子保存当前累积样本到磁盘

    同时写入 sample.json 和 summary.json。
    采用"先写 .tmp 再 rename"原子写入策略。
    """
    hyperparams = _build_hyperparams(args)

    # 原子写入 sample.json
    sample_path = os.path.join(output_root, "sample.json")
    tmp_sample = sample_path + ".tmp"
    with open(tmp_sample, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False)
    os.replace(tmp_sample, sample_path)

    # 原子写入 summary.json
    summary_path = os.path.join(output_root, "summary.json")
    tmp_summary = summary_path + ".tmp"
    summary = {
        "model": args.model,
        "games": args.games,
        "total_samples": len(all_samples),
        "completed_games": completed_games,
        "hyperparams": hyperparams,
    }
    with open(tmp_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    os.replace(tmp_summary, summary_path)

    print(f"[保存] {completed_games} 局, {len(all_samples)} 条样本已写入磁盘")


# ---------- 参数解析 ----------

def parse_args():
    """解析 CLI 参数"""
    p = argparse.ArgumentParser(
        description="价值头自我对弈脚本 — 生成训练样本 sample.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="权重文件路径（加载此 .pth 做推理）")
    p.add_argument("--model-name", default=None,
                   help="模型注册名.代数，如 cnn_v10.1。默认从 --model 路径自动解析")
    p.add_argument("--games", type=int, required=True,
                   help="对弈局数")
    p.add_argument("--output", default=None,
                   help="输出目录名（默认 {model}_value_{timestamp}）")
    p.add_argument("--device", default="cuda",
                   help="推理设备（默认 cuda）")
    p.add_argument("--workers", type=int, default=None,
                   help="并行 worker 进程数（默认 CPU 核数，1 为单进程串行）")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="温度值（默认 1.0）")
    p.add_argument("--temp-steps", type=int, default=8,
                   help="前 N 步启用温度采样（默认 8）")
    p.add_argument("--temp-side", type=str, default="random",
                   choices=["black", "white", "random", "both"],
                   help="温度采样方（默认 random）")
    p.add_argument("--top-k", type=int, default=None,
                   help="采样截断阈值")
    p.add_argument("--temp-seed", type=int, default=None,
                   help="温度采样随机种子")
    return p.parse_args()


# ---------- 主流程 ----------

def _distribute_indices(total: int, num_chunks: int) -> List[List[int]]:
    """将 [0, total) 的整数均匀分配到 num_chunks 个列表中"""
    if num_chunks <= 0:
        num_chunks = 1
    indices = list(range(total))
    chunks: List[List[int]] = [[] for _ in range(num_chunks)]
    for i, idx in enumerate(indices):
        chunks[i % num_chunks].append(idx)
    return [c for c in chunks if c]


def run_self_play(args):
    """执行所有局自我对弈"""
    # 确定输出根目录
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        output_root = f"{model_stem}_value_{ts}"
    else:
        output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    # 确定 worker 数
    workers = getattr(args, 'workers', 1)
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    # 设备：多进程模式下固定使用 CPU
    if workers > 1:
        device = "cpu"
    else:
        device = torch.device(
            args.device if torch.cuda.is_available() and args.device == "cuda"
            else "cpu"
        )
    print(f"[设备] {device}")
    print(f"[workers] {workers}")

    # 超参数
    hyperparams = _build_hyperparams(args)
    print(f"[超参数] {hyperparams}")

    # 生成本次运行的唯一标识
    run_id = f"{int(time.time() * 1000) % 1000000:06d}"

    # 加载已有样本
    all_samples = _load_existing_samples(output_root)
    if all_samples:
        print(f"[加载] 已有 {len(all_samples)} 条样本")

    # 注册 SIGINT 信号处理器
    global _interrupted
    _interrupted = False
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    try:
        completed_games = 0

        if workers == 1:
            completed_games, all_samples = _run_single_process(
                args, hyperparams, output_root, run_id, all_samples, device
            )
        else:
            completed_games, all_samples = _run_multi_process(
                args, hyperparams, output_root, run_id, all_samples, workers
            )

        # 最终保存
        if completed_games > 0:
            _save_progress(all_samples, output_root, args, completed_games)

        print(f"\n[完成] 共 {completed_games} 局，{len(all_samples)} 条样本，输出: {output_root}")

    finally:
        signal.signal(signal.SIGINT, original_handler)


def _run_single_process(args, hyperparams: dict, output_root: str,
                        run_id: str, all_samples: List[dict], device):
    """单进程模式：加载模型并串行执行所有对局"""
    print(f"[加载模型] {args.model}")
    model_spec = getattr(args, 'model_name', None)
    if not model_spec:
        model_spec = _parse_model_spec_from_path(args.model)
    model = get_model(model_spec)
    model.move_to_device(device)

    cfg = ValueSelfPlayConfig(
        temperature=hyperparams["temperature"],
        temp_steps=hyperparams["temp_steps"],
        temp_side=hyperparams["temp_side"],
        top_k=hyperparams["top_k"],
        seed=hyperparams["temp_seed"],
    )

    completed_games = 0
    for game_idx in range(args.games):
        if _interrupted:
            print(f"\n[中断] 已停止，共完成 {completed_games} 局")
            break

        print(f"\n{'='*50}")
        print(f"[局 {game_idx + 1}/{args.games}] 开始")
        t_start = time.time()

        samples = play_one_game(model, device, cfg, game_idx, run_id)
        all_samples.extend(samples)
        completed_games += 1

        elapsed = time.time() - t_start
        result_desc = _describe_result(samples)
        print(f"[局 {game_idx + 1}/{args.games}] 完成 — {len(samples)} 步, "
              f"结果: {result_desc}, 耗时: {elapsed:.1f}s")

        if completed_games % _SAVE_INTERVAL == 0:
            _save_progress(all_samples, output_root, args, completed_games)

    return completed_games, all_samples


def _run_multi_process(args, hyperparams: dict, output_root: str,
                       run_id: str, all_samples: List[dict], workers: int):
    """多进程模式：通过 ProcessPoolExecutor 并行执行对局"""
    cfg = ValueSelfPlayConfig(
        temperature=hyperparams["temperature"],
        temp_steps=hyperparams["temp_steps"],
        temp_side=hyperparams["temp_side"],
        top_k=hyperparams["top_k"],
        seed=hyperparams["temp_seed"],
    )

    # 分配 game_indices
    model_name = getattr(args, 'model_name', None)
    chunks = _distribute_indices(args.games, workers)
    print(f"[分配] {workers} 个 worker，各负责局数: {[len(c) for c in chunks]}")

    completed_games = 0
    t_total_start = time.time()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for chunk in chunks:
                future = executor.submit(
                    ValueSelfPlayWorker.worker_entry,
                    model_path=args.model,
                    model_name=model_name,
                    device="cpu",
                    config=cfg,
                    game_indices=chunk,
                    run_id=run_id,
                )
                futures[future] = len(chunk)

            for future in as_completed(futures):
                chunk_size = futures[future]
                try:
                    chunk_samples = future.result()
                    all_samples.extend(chunk_samples)
                    completed_games += chunk_size
                    print(f"[进度] worker 完成 {chunk_size} 局，累计 {completed_games}/{args.games}")
                    if completed_games % _SAVE_INTERVAL == 0:
                        _save_progress(all_samples, output_root, args, completed_games)
                except Exception as e:
                    print(f"[错误] worker 执行失败: {e}")

    except KeyboardInterrupt:
        print("\n[中断] 等待进行中的 worker 完成...")

    elapsed_total = time.time() - t_total_start
    print(f"[耗时] 总耗时 {elapsed_total:.1f}s")

    return completed_games, all_samples


def play_one_game(model, device, cfg: ValueSelfPlayConfig,
                  game_idx: int, run_id: str) -> List[dict]:
    """执行一局自我对弈（薄封装，委托给 ValueSelfPlayWorker）

    Args:
        model: CNNV10Model 实例
        device: torch device
        cfg: 超参数配置
        game_idx: 局编号
        run_id: 本次运行唯一标识

    Returns:
        sample 列表，每项 {"state": [...], "z": int}
    """
    worker = ValueSelfPlayWorker(
        model_path="", model_name=None, device=str(device),
        config=cfg, _model=model,
    )
    return worker.run_games(game_indices=[game_idx], run_id=run_id)


def _describe_result(samples: List[dict]) -> str:
    """用人类可读方式描述对局结果"""
    if not samples:
        return "无样本"
    last_z = samples[-1]["z"]
    last_player = samples[-1]["state"][225]
    if last_z == 1:
        return f"玩家{last_player}胜"
    elif last_z == -1:
        other = 2 if last_player == 1 else 1
        return f"玩家{other}胜"
    else:
        return "平局"


# ---------- 入口 ----------

if __name__ == "__main__":
    args = parse_args()
    run_self_play(args)
