"""MCTS 自我对弈脚本

使用 MCTS 搜索生成训练样本，所有局合并输出一个 sample.json。

用法：
    python src/ai/scripts/mcts_self_play.py --model model.pth --games 100

设计文档 §4.4 规范实现。
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

from ai.scripts.mcts_self_play_worker import MctsSelfPlayWorker, MctsSelfPlayConfig
from ai.registry import get_model
from ai.engine.builder import AIEngineBuilder
from ai.constants import STRATEGY_SAMPLING
from config.mcts import MCTS_BATCH_SIZE, MCTS_VIRTUAL_LOSS
from game.game import check_winner, is_board_full


# ---------- 常量 ----------

# 周期保存间隔（局数），不可通过 CLI 配置
_SAVE_INTERVAL = 50

# 中断标志位，由信号处理器设置，主循环检查
_interrupted = False


def _signal_handler(signum, frame):
    """SIGINT 信号处理器 — 设置中断标志，主循环检测后触发保存"""
    global _interrupted
    _interrupted = True
    print("\n[中断] 收到 Ctrl+C，正在保存数据...")


def _load_existing_samples(output_root: str) -> List[dict]:
    """加载输出目录下已有的 sample.json

    若文件存在且格式正确，返回其 JSON 数组内容；
    若文件不存在或格式损坏，返回空列表。

    Args:
        output_root: 输出根目录路径

    Returns:
        已有样本列表（可能为空）
    """
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
    采用"先写 .tmp 再 rename"原子写入策略，防止写入过程中崩溃导致文件损坏。

    Args:
        all_samples: 当前累积的全部样本
        output_root: 输出根目录路径
        args: argparse.Namespace（用于获取 model、games、hyperparams）
        completed_games: 当前已完成局数
    """
    # 超参数（与 run_self_play 中定义一致）
    hyperparams = {
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "temp_steps": args.temp_steps,
        "temperature": args.temperature,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "dirichlet_alpha": args.dirichlet_alpha,
        "batch_size": args.batch_size if args.batch_size is not None else MCTS_BATCH_SIZE,
        "virtual_loss": args.virtual_loss if args.virtual_loss is not None else MCTS_VIRTUAL_LOSS,
        "use_prior": not args.no_prior,
    }

    # 原子写入 sample.json
    sample_path = os.path.join(output_root, "sample.json")
    tmp_sample = sample_path + ".tmp"
    with open(tmp_sample, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False)
    os.replace(tmp_sample, sample_path)

    # 原子写入 summary.json（新增 completed_games 字段）
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
        description="MCTS 自我对弈脚本 — 生成训练样本 sample.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="权重文件路径（加载此 .pth 做 MCTS 推理）")
    p.add_argument("--model-name", default=None,
                   help="模型注册名.代数，如 cnn_v8.1。默认从 --model 路径自动解析")
    p.add_argument("--games", type=int, required=True,
                   help="对弈局数")
    p.add_argument("--simulations", type=int, default=800,
                   help="每步 MCTS 模拟次数（默认 800）")
    p.add_argument("--c-puct", type=float, default=2.0,
                   help="PUCT 探索系数（默认 2.0）")
    p.add_argument("--temp-steps", type=int, default=8,
                   help="温度探索步数，前 N 步 τ=1.0（默认 8）")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="温度参数（默认 1.0）")
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                   help="狄利克雷噪声比例（默认 0.25）")
    p.add_argument("--dirichlet-alpha", type=float, default=0.03,
                   help="狄利克雷分布参数（默认 0.03）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="每批 simulation 数（默认读取 MCTS_BATCH_SIZE 配置）")
    p.add_argument("--virtual-loss", type=float, default=None,
                   help="virtual loss 值（默认读取 MCTS_VIRTUAL_LOSS 配置）")
    p.add_argument("--output", default=None,
                   help="输出目录名（默认 {model}_mcts_{timestamp}）")
    p.add_argument("--no-prior", action="store_true",
                   help="禁用模型先验概率，MCTS 仅依赖价值评估")
    p.add_argument("--device", default="cuda",
                   help="推理设备（默认 cuda）")
    p.add_argument("--workers", type=int, default=None,
                   help="并行 worker 进程数（默认 CPU 核数，1 为单进程串行）")
    return p.parse_args()


# ---------- 主流程 ----------

def _distribute_indices(total: int, num_chunks: int) -> List[List[int]]:
    """将 [0, total) 的整数均匀分配到 num_chunks 个列表中

    Args:
        total: 总数
        num_chunks: 分块数

    Returns:
        列表的列表，每个子列表包含分配到的索引
    """
    if num_chunks <= 0:
        num_chunks = 1
    indices = list(range(total))
    chunks: List[List[int]] = [[] for _ in range(num_chunks)]
    for i, idx in enumerate(indices):
        chunks[i % num_chunks].append(idx)
    # 移除空 chunk（当 num_chunks > total 时）
    return [c for c in chunks if c]


def run_self_play(args):
    """执行所有局自我对弈

    根据 --workers 参数选择单进程或多进程模式：
    - workers=1: 单进程串行（与原有逻辑一致）
    - workers>1: ProcessPoolExecutor 并行
    """
    # 确定输出根目录
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        output_root = f"{model_stem}_mcts_{ts}"
    else:
        output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    # 确定 worker 数
    # getattr 在 args 不含 workers 属性时回退到 1（兼容旧测试 Mock）
    workers = getattr(args, 'workers', 1)
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, workers)

    # 设备：多进程模式下固定使用 CPU
    if workers > 1:
        device = "cpu"
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[设备] {device}")
    print(f"[workers] {workers}")

    # 超参数汇总
    hyperparams = {
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "temp_steps": args.temp_steps,
        "temperature": args.temperature,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "dirichlet_alpha": args.dirichlet_alpha,
        "batch_size": args.batch_size if args.batch_size is not None else MCTS_BATCH_SIZE,
        "virtual_loss": args.virtual_loss if args.virtual_loss is not None else MCTS_VIRTUAL_LOSS,
        "use_prior": not args.no_prior,
    }
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
            # 单进程模式：保持原有逻辑
            completed_games, all_samples = _run_single_process(
                args, hyperparams, output_root, run_id, all_samples
            )
        else:
            # 多进程模式
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
                        run_id: str, all_samples: List[dict]):
    """单进程模式：加载模型并串行执行所有对局

    与原有 run_self_play 逻辑一致，保留模型加载和设备配置。
    """
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

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
                f"请使用 --model-name 显式指定，如 --model-name cnn_v8.1"
            )
    model.move_to_device(device)

    completed_games = 0
    for game_idx in range(args.games):
        if _interrupted:
            print(f"\n[中断] 已停止，共完成 {completed_games} 局")
            break

        print(f"\n{'='*50}")
        print(f"[局 {game_idx + 1}/{args.games}] 开始")
        t_start = time.time()

        samples = play_one_game(model, device, hyperparams, game_idx, run_id)
        all_samples.extend(samples)
        completed_games += 1

        elapsed = time.time() - t_start
        print(f"[局 {game_idx + 1}/{args.games}] 完成 — {len(samples)} 步, "
              f"结果: {_describe_result(samples)}, 耗时: {elapsed:.1f}s")

        if completed_games % _SAVE_INTERVAL == 0:
            _save_progress(all_samples, output_root, args, completed_games)

    return completed_games, all_samples


def _run_multi_process(args, hyperparams: dict, output_root: str,
                       run_id: str, all_samples: List[dict], workers: int):
    """多进程模式：通过 ProcessPoolExecutor 并行执行对局

    将 game_indices 均匀分配给各 worker，每个 worker 在子进程中独立加载模型并执行。
    收集完成后合并样本并保存。
    """
    # 构建 MctsSelfPlayConfig
    config = MctsSelfPlayConfig(
        simulations=hyperparams["simulations"],
        c_puct=hyperparams["c_puct"],
        temp_steps=hyperparams["temp_steps"],
        temperature=hyperparams["temperature"],
        dirichlet_epsilon=hyperparams["dirichlet_epsilon"],
        dirichlet_alpha=hyperparams["dirichlet_alpha"],
        batch_size=hyperparams["batch_size"],
        virtual_loss=hyperparams["virtual_loss"],
        use_prior=hyperparams["use_prior"],
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
                    MctsSelfPlayWorker.worker_entry,
                    model_path=args.model,
                    model_name=model_name,
                    device="cpu",
                    config=config,
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
                except Exception as e:
                    print(f"[错误] worker 执行失败: {e}")

    except KeyboardInterrupt:
        print("\n[中断] 等待进行中的 worker 完成...")
        # ProcessPoolExecutor 的 __exit__ 会等待已提交的任务

    elapsed_total = time.time() - t_total_start
    print(f"[耗时] 总耗时 {elapsed_total:.1f}s")

    return completed_games, all_samples


def play_one_game(model, device, hyperparams: dict, game_idx: int, run_id: str) -> List[dict]:
    """执行一局自我对弈

    Args:
        model: CNNV7Model 实例
        device: torch device
        hyperparams: 超参数字典
        game_idx: 局编号（用于 log_key）
        run_id: 本次运行唯一标识（用于 log_key 前缀）

    Returns:
        sample 列表，每项 {"state": [...], "pi": [...], "z": int}
    """
    simulations = hyperparams["simulations"]
    c_puct = hyperparams["c_puct"]
    temp_steps = hyperparams["temp_steps"]
    temperature = hyperparams["temperature"]
    dirichlet_epsilon = hyperparams["dirichlet_epsilon"]
    dirichlet_alpha = hyperparams["dirichlet_alpha"]
    batch_size = hyperparams["batch_size"]
    virtual_loss = hyperparams["virtual_loss"]
    use_prior = hyperparams["use_prior"]

    # 通过 AIEngineBuilder 创建策略-价值引擎（内部自动创建 MCTSSearcher + PolicyValueAiEngine）
    log_key_prefix = f"run{run_id}_game{game_idx}"
    engine = AIEngineBuilder.build_policy_value(
        model=model,
        num_simulations=simulations,
        c_puct=c_puct,
        dirichlet_epsilon=dirichlet_epsilon,
        dirichlet_alpha=dirichlet_alpha,
        batch_size=batch_size,
        virtual_loss=virtual_loss,
        temperature=temperature,
        temp_steps=temp_steps,
        log_key_prefix=log_key_prefix,
        use_prior=use_prior,
    )

    # 初始化棋盘
    board = [0] * 225
    current_player = 1  # 黑先
    samples = []
    move_count = 0

    while True:
        state = board + [current_player]

        # 将 flat board 转为 15×15 二维矩阵（get_move 新接口要求）
        board_2d = [[board[i * 15 + j] for j in range(15)] for i in range(15)]

        # MCTS 搜索 + 决策（log_key 由引擎内部自动生成，strategy="sampling" 使用温度探索）
        md = engine.get_move(board_2d, current_player, strategy=STRATEGY_SAMPLING)
        x, y = md.get_position()
        action = x * 15 + y
        visit_counts = md.get_mcts_visit_counts()

        # 计算 π 分布（τ=1.0 归一化）
        total_visits = sum(visit_counts)
        if total_visits > 0:
            pi = [c / total_visits for c in visit_counts]
        else:
            pi = [1.0 / 225] * 225

        # 记录样本（z 暂填 0，终局后回填）
        samples.append({"state": state, "pi": pi, "z": 0})

        # 落子到棋盘
        board[action] = current_player

        # 终局判定
        winner = check_winner(board, action)
        is_terminal = (winner != 0) or is_board_full(board)

        # 子树复用（无条件调用 after_move，终局节点的 advance_root 同样成功）
        # 使用 (x, y) 坐标代替旧 action 索引
        engine.after_move(x, y)

        if is_terminal:
            if winner == 0:
                winner = 0  # 平局
            break

        # 切换玩家
        current_player = 2 if current_player == 1 else 1
        move_count += 1

    # 回填 z 值（z 值翻转算法）
    for sample in samples:
        P = sample["state"][225]  # 该步的落子方
        if winner == 0:
            sample["z"] = 0
        elif winner == P:
            sample["z"] = 1
        else:
            sample["z"] = -1

    return samples


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
