"""data_parser 模块 — record.json 解析与样本匹配

负责：
1. scan_round_dir: 扫描 round_dir，返回对局摘要列表
2. parse_game_steps: 解析单局步数据（record.json + 样本匹配）
3. match_samples: 用 step_id 匹配样本到 history 各步
4. infer_temp_side: 从 history 推断采样方

零耦合：不 import 任何 ai/ 包模块。
"""

import json
import os
from typing import Dict, List


# ------------------------------------------------------------------
# 显示映射常量
# ------------------------------------------------------------------

# winner 值 → 中文显示
WINNER_DISPLAY_MAP = {1: "黑胜", 2: "白胜", 3: "平局"}

# temp_side 值 → 中文显示
TEMP_SIDE_DISPLAY_MAP = {
    "both": "双方采样",
    "black": "黑方采样",
    "white": "白方采样",
    "none": "无(argmax)",
}


def scan_round_dir(round_dir: str) -> List[Dict]:
    """扫描 round_dir，返回对局摘要列表

    按目录名（数字）排序扫描，解析每局 record.json。
    损坏的 JSON 跳过不报错。

    Args:
        round_dir: 轮次目录路径

    Returns:
        对局摘要列表，每项含 game_id、winner、temp_side 等。

    Raises:
        FileNotFoundError: round_dir 不存在
    """
    if not os.path.exists(round_dir):
        raise FileNotFoundError(f"数据目录不存在: {round_dir}")

    games = []
    # 按目录名排序（仅数字目录名有效）
    entries = []
    for entry in os.listdir(round_dir):
        try:
            int(entry)  # 仅数字目录名
            entries.append(entry)
        except ValueError:
            continue

    for entry in sorted(entries, key=int):
        record_path = os.path.join(round_dir, entry, "record.json")
        if not os.path.exists(record_path):
            continue

        try:
            with open(record_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # 跳过损坏文件

        winner = record.get("winner", 0)
        history = record.get("history", [])
        temp_side = infer_temp_side(history)

        games.append({
            "game_id": int(entry),
            "winner": winner,
            "winner_display": WINNER_DISPLAY_MAP.get(winner, "未知"),
            "temp_side": temp_side,
            "temp_side_display": TEMP_SIDE_DISPLAY_MAP.get(temp_side, "未知"),
            "total_steps": len(history),
            "is_draw": winner == 3,
        })

    return games


def parse_game_steps(round_dir: str, game_id: int,
                     sample_path: str = None) -> Dict:
    """解析单局步数据

    读取 record.json，可选匹配样本文件，返回步数据列表。

    Args:
        round_dir: 轮次目录路径
        game_id: 对局编号（从 1 开始）
        sample_path: 样本输出文件路径，None 时不匹配样本

    Returns:
        {game_id, winner, winner_display, is_draw, steps: [StepData, ...]}

    Raises:
        FileNotFoundError: record.json 不存在
        json.JSONDecodeError: record.json 格式损坏
    """
    record_path = os.path.join(round_dir, str(game_id), "record.json")
    if not os.path.exists(record_path):
        raise FileNotFoundError(f"对局 {game_id} 不存在")

    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # 读取样本文件（如存在）
    samples = []
    if sample_path and os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

    # 匹配样本
    sample_map = match_samples(record.get("history", []), samples)

    # 构建步数据
    history = record.get("history", [])
    steps = []
    for i, step in enumerate(history):
        board = step["input"][:225]
        player = step["input"][225]
        move_pos = step["position"]

        steps.append({
            "step_index": i,
            "player": player,
            "move_position": move_pos,
            "board": board,
            "raw_output": step["output"],
            "sampling_probs": step.get("sampling_probs"),
            "strategy": step.get("strategy"),
            "sample_output": sample_map.get(i),
        })

    winner = record.get("winner", 0)
    return {
        "game_id": game_id,
        "winner": winner,
        "winner_display": WINNER_DISPLAY_MAP.get(winner, "未知"),
        "is_draw": winner == 3,
        "steps": steps,
    }


def match_samples(history: List[Dict], samples: List[Dict]) -> Dict[int, List[float]]:
    """用 step_id 将样本匹配到 history 各步

    匹配算法：构建 step_id → sample_output 映射，O(1) 查找。
    D4 增强产生的 8 个同 step_id 变体，仅首次写入（恒等变换 tid=0）。
    旧格式数据（无 step_id）自动跳过。

    Args:
        history: record.json 的 history 列表
        samples: 样本输出列表

    Returns:
        {step_index: sample_output} 映射。
        未匹配的步不在映射中，调用方视为 null。
    """
    # 构建 step_id → sample_output 映射（仅首次写入）
    sample_map = {}
    for sample in samples:
        sid = sample.get("step_id")
        if sid is not None and sid not in sample_map:
            sample_map[sid] = sample["output"]

    # 用 step_id 匹配
    result = {}
    for i, step in enumerate(history):
        sid = step.get("step_id")
        if sid is not None and sid in sample_map:
            result[i] = sample_map[sid]

    return result


def infer_temp_side(history: List[Dict]) -> str:
    """从 history 推断该局的采样方

    检查每步的 strategy 字段：strategy="sampling" 时，
    根据第 226 维（input[225]）判断采样方。

    Args:
        history: record.json 的 history 列表

    Returns:
        "both" / "black" / "white" / "none"
    """
    black_sampling = False
    white_sampling = False

    for step in history:
        if step.get("strategy") == "sampling":
            player = step["input"][225]
            if player == 1:
                black_sampling = True
            if player == 2:
                white_sampling = True

    if black_sampling and white_sampling:
        return "both"
    if black_sampling:
        return "black"
    if white_sampling:
        return "white"
    return "none"