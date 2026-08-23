"""cnn_v4.3 自我对弈样本生成脚本（intention 54）

让 cnn_v4.3 模型自我对弈产生训练样本，输出到样本目录。
与 cnn_v11 样本缓冲区的核心区别：**不使用 MCTS 搜索**，
每步直接采用模型的输出概率分布作为样本的概率分布（pi）。

样本结构与 cnn_v11 完全一致：
    {"state": [226 维棋盘向量], "pi": [225 维概率分布], "z": 1.0 / -1.0 / 0.0}

对弈策略：
- 前 temp_steps 步按 temperature 做温度采样（p^(1/T) 加权）
- 之后切换为 argmax 贪心

用法：
    python src/ai/scripts/cnn_v4_self_play.py
    python src/ai/scripts/cnn_v4_self_play.py --games 200 --temp-steps 10
    python src/ai/scripts/cnn_v4_self_play.py --output samples/boots/2 --model cnn_v4.3
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import List, Optional

# 确保项目根与 src 在 sys.path 中（脚本路径: src/ai/scripts/，3 层 dirname 到项目根）
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_src_dir = os.path.join(_proj_root, "src")
for _p in (_proj_root, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from ai.constants import STRATEGY_ARGMAX, STRATEGY_SAMPLING  # noqa: E402
from ai.engine import SinglePolicyAiEngine  # noqa: E402
from ai.registry import get_model  # noqa: E402
from game.game import check_winner, is_board_full  # noqa: E402


# ---------- 常量 ----------

# 周期保存间隔（局数），不可通过 CLI 配置
_SAVE_INTERVAL = 50

# 中断标志位，由信号处理器设置，主循环检查
_interrupted = False


# ---------- 参数解析 ----------

def parse_args(argv: Optional[List[str]] = None):
    """解析 CLI 参数

    Args:
        argv: 参数列表，None 时使用 sys.argv

    Returns:
        解析后的 argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description="cnn_v4.3 自我对弈脚本 — 生成训练样本 sample.json（不使用 MCTS，"
                    "直接采用模型输出概率分布作为 pi）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="cnn_v4.3",
                   help="模型 registry spec（默认 cnn_v4.3）")
    p.add_argument("--games", type=int, default=100,
                   help="对弈局数（默认 100）")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="温度采样系数（默认 1.0）")
    p.add_argument("--temp-steps", type=int, default=12,
                   help="温度采样全局步数，前 N 步温度采样，之后 argmax（默认 12）")
    p.add_argument("--output", default="samples/boots/1",
                   help="样本保存目录（默认 samples/boots/1）")
    p.add_argument("--device", default="cuda",
                   help="推理设备（默认 cuda，不可用时回退 cpu）")
    return p.parse_args(argv)


# ---------- 超参数 ----------

def _build_hyperparams(args) -> dict:
    """从 CLI args 构建超参数字典

    Args:
        args: argparse.Namespace

    Returns:
        超参数字典
    """
    return {
        "temperature": args.temperature,
        "temp_steps": args.temp_steps,
    }


# ---------- 中断处理 ----------

def _signal_handler(signum, frame):
    """SIGINT 信号处理器 — 设置中断标志，主循环检测后触发保存"""
    global _interrupted
    _interrupted = True
    print("\n[中断] 收到 Ctrl+C，正在保存数据...")


# ---------- 单局对弈 ----------

def _play_one_game(
    model,
    temperature: float,
    temp_steps: int,
    rng=None,
) -> List[dict]:
    """执行一局自我对弈，返回本局样本

    双方轮流执方（同一模型），每步：
    1. 记录当前局面 state = board + [current_player]
    2. 前 temp_steps 步用温度采样、之后 argmax 选择落子
    3. 直接采用模型输出的 225 维概率分布作为样本 pi
    4. 终局后按胜负回填 z 值

    Args:
        model: cnn_v4.3 模型实例（继承 BaseModel）
        temperature: 温度采样系数，必须 > 0
        temp_steps: 温度采样全局步数，>= 0（0 表示全程 argmax）
        rng: 可选随机源，注入引擎用于温度采样；None 时使用全局随机源

    Returns:
        本局样本列表，每项 {"state": [...226], "pi": [...225], "z": 1.0/-1.0/0.0}
    """
    engine = SinglePolicyAiEngine(model, temperature=temperature, rng=rng)

    board: List[int] = [0] * 225
    current_player: int = 1
    move_count: int = 0
    samples: List[dict] = []
    winner: int = 0

    while True:
        # 记录当前局面（落子前快照）
        state = board + [current_player]
        board_2d = [[board[i * 15 + j] for j in range(15)] for i in range(15)]

        # 前 temp_steps 步温度采样，之后 argmax
        strategy = STRATEGY_SAMPLING if move_count < temp_steps else STRATEGY_ARGMAX
        md = engine.get_move(board_2d, current_player, strategy=strategy)

        # 采用模型输出概率分布作为样本 pi（与 cnn_v11 的 MCTS visit 分布不同）
        pi = md.get_probs()
        samples.append({"state": state, "pi": pi, "z": 0})

        # 落子并判定终局
        x, y = md.get_position()
        action = x * 15 + y
        board[action] = current_player
        move_count += 1

        winner = check_winner(board, action)
        if winner != 0 or is_board_full(board):
            break

        # 切换玩家
        current_player = 2 if current_player == 1 else 1

    # 终局回填 z 值：1=当前方胜，-1=当前方败，0=平局
    final_winner = winner if winner != 0 else 0
    for sample in samples:
        player = sample["state"][225]
        if final_winner == 0:
            sample["z"] = 0.0
        elif final_winner == player:
            sample["z"] = 1.0
        else:
            sample["z"] = -1.0

    return samples


# ---------- 保存 ----------

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
    # 超参数
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


# ---------- 结果描述 ----------

def _describe_result(samples: List[dict]) -> str:
    """用人类可读方式描述对局结果

    Args:
        samples: 本局样本列表

    Returns:
        结果描述字符串，如 "玩家1胜" / "平局"
    """
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


# ---------- 主流程 ----------

def run_self_play(args) -> None:
    """执行所有局自我对弈（单进程串行）

    Args:
        args: argparse.Namespace（由 parse_args 返回）
    """
    # 确定输出根目录并创建
    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    # 设备：cuda 不可用时回退 cpu
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"[设备] {device}")

    # 超参数
    hyperparams = _build_hyperparams(args)
    print(f"[超参数] {hyperparams}")
    print(f"[模型] {args.model}")

    # 加载模型
    print(f"[加载模型] {args.model}")
    model = get_model(args.model)
    model.move_to_device(device)

    # 加载已有样本（支持断点续跑）
    all_samples = _load_existing_samples(output_root)
    if all_samples:
        print(f"[加载] 已有 {len(all_samples)} 条样本")

    # 注册 SIGINT 信号处理器
    global _interrupted
    _interrupted = False
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    try:
        completed_games = 0
        for game_idx in range(args.games):
            if _interrupted:
                print(f"\n[中断] 已停止，共完成 {completed_games} 局")
                break

            print(f"\n{'=' * 50}")
            print(f"[局 {game_idx + 1}/{args.games}] 开始")
            t_start = time.time()

            samples = _play_one_game(
                model,
                temperature=hyperparams["temperature"],
                temp_steps=hyperparams["temp_steps"],
            )
            all_samples.extend(samples)
            completed_games += 1

            elapsed = time.time() - t_start
            print(f"[局 {game_idx + 1}/{args.games}] 完成 — {len(samples)} 步, "
                  f"结果: {_describe_result(samples)}, 耗时: {elapsed:.1f}s")

            # 周期保存：每 _SAVE_INTERVAL 局写入磁盘，防止中途崩溃丢失数据
            if completed_games % _SAVE_INTERVAL == 0:
                _save_progress(all_samples, output_root, args, completed_games)

        # 最终保存
        if completed_games > 0:
            _save_progress(all_samples, output_root, args, completed_games)

        print(f"\n[完成] 共 {completed_games} 局，{len(all_samples)} 条样本，"
              f"输出: {output_root}")

    finally:
        signal.signal(signal.SIGINT, original_handler)


# ---------- 入口 ----------

if __name__ == "__main__":
    run_self_play(parse_args())
