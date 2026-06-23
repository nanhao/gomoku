"""棋型样本生成器（正式工具版）

按 design/sample_generator_design.md 实现。

核心职责
--------
1. 按指定棋型（chong4 / huo4 / huo3 / huo2 / defend_chong4 / defend_huo3）
   合成一个 15×15 棋盘，并标记其"进攻位 / 防守位"索引 ``attack_positions``。
2. 调用 :class:`ai.reward_calculator.RewardCalculator` 计算每个空位 reward，
   经 :func:`ai.probability_utils.masked_softmax` 归一化为训练标签 ``output``。
3. 对生成的样本进行 label 质量体检：Hit@N / Hit@N+1 / Hit@N+2 /
   进攻位概率和 / 区分度 gap，以及失败样例列表。
4. 提供 CLI 入口：``python -m ai.sample_generator --shape ... --num-samples ... --output ...``

设计约束（与设计文档严格对齐）
------------------------------
- 单次运行只生成一种棋型、写一个文件；棋型 / 样本数 / 输出路径参数化
- 6 种棋型共用统一生成核心（4 种进攻 + 2 种防守委托）
- 生成阶段记录 attack_positions，验证阶段直接消费，不再事后 detect
- 干扰子放置后必须通过 ``detect_shapes`` 二次校验（每个进攻位落子是否形成期望棋型），
  不合格则整盘丢弃重试
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from ai.probability_utils import masked_softmax
from ai.reward_calculator import RewardCalculator, RewardConfig
from ai.shape_detector import detect_shapes


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BOARD_SIZE: int = 15
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE

# 四个方向：横 / 纵 / 主对角 / 副对角
_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

# 支持的棋型
SUPPORTED_SHAPES: Tuple[str, ...] = (
    # 进攻（attacker 视角，current_player 持有该棋型）
    "chong4", "huo4", "huo3", "huo2",
    # 防守（defender 视角，current_player 面对对手威胁）
    "defend_chong4", "defend_huo3",
)

_ATTACK_SHAPES = {"chong4", "huo4", "huo3", "huo2"}

# 防守棋型 → 对应进攻棋型（用于委托生成 + 颜色翻转）
_DEFEND_TO_ATTACK: Dict[str, str] = {
    "defend_chong4": "chong4",
    "defend_huo3":   "huo3",
}

# 每种进攻棋型的参数
#   stones:      连珠数
#   open_ends:   两端开放数（1=一端挡一端开放, 2=两端都开放）
#   expect:      进攻位落子后期望形成的 detect_shapes 棋型关键词
#   require_outer_open:
#                进攻位"外侧"必须是空位（避免落子后 blocked=1 只形成次级棋型）
_ATTACK_SPECS: Dict[str, Dict] = {
    "chong4": {"stones": 4, "open_ends": 1, "expect": ("连五",),           "require_outer_open": False},
    "huo4":   {"stones": 4, "open_ends": 2, "expect": ("连五",),           "require_outer_open": False},
    "huo3":   {"stones": 3, "open_ends": 2, "expect": ("活四",),           "require_outer_open": True},
    "huo2":   {"stones": 2, "open_ends": 2, "expect": ("活三", "多活三"),   "require_outer_open": True},
}

# 单次生成尝试上限（防止无限循环）
_MAX_ATTEMPTS_PER_BOARD: int = 300
# 单个干扰子放置时的最大重试
_MAX_EXTRA_TRIES: int = 50


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _in_board(x: int, y: int) -> bool:
    """坐标是否在 15×15 棋盘内"""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _idx(x: int, y: int) -> int:
    """二维坐标 → 一维索引"""
    return x * BOARD_SIZE + y


# ---------------------------------------------------------------------------
# 进攻棋型生成核心
# ---------------------------------------------------------------------------

def _generate_attack(shape: str, rg: random.Random,
                     max_extra: int) -> Optional[Tuple[List[int], List[int], int]]:
    """生成一个进攻棋型棋盘

    Returns:
        (board, attack_positions(一维索引列表), player) 或 None（本次失败）
    """
    spec = _ATTACK_SPECS[shape]
    stones: int = spec["stones"]
    open_ends: int = spec["open_ends"]
    expect_shapes: Tuple[str, ...] = spec["expect"]
    require_outer_open: bool = spec["require_outer_open"]

    for _ in range(_MAX_ATTEMPTS_PER_BOARD):
        player = rg.choice((1, 2))
        opponent = 3 - player
        dx, dy = rg.choice(_DIRECTIONS)

        # 起点（连珠第 0 颗棋子）
        sx = rg.randint(0, BOARD_SIZE - 1)
        sy = rg.randint(0, BOARD_SIZE - 1)

        # 放置 stones 颗连珠
        stone_positions: List[Tuple[int, int]] = []
        for i in range(stones):
            x, y = sx + i * dx, sy + i * dy
            if not _in_board(x, y):
                break
            stone_positions.append((x, y))
        if len(stone_positions) != stones:
            continue

        # 两端坐标
        neg_end = (sx - dx, sy - dy)
        last_x, last_y = stone_positions[-1]
        pos_end = (last_x + dx, last_y + dy)
        neg_valid = _in_board(*neg_end)
        pos_valid = _in_board(*pos_end)

        attack_positions: List[Tuple[int, int]] = []
        blocked_positions: List[Tuple[int, int]] = []
        outers: List[Tuple[int, int]] = []  # 需要保持为空的外侧位（huo3/huo2）

        if open_ends == 2:
            # huo4 / huo3 / huo2：两端都必须在盘内 + 空位
            if not (neg_valid and pos_valid):
                continue
            attack_positions = [neg_end, pos_end]

            # huo3 / huo2：两端外侧必须是盘内空位（否则落子后变冲四/眠三）
            if require_outer_open:
                neg_outer = (neg_end[0] - dx, neg_end[1] - dy)
                pos_outer = (pos_end[0] + dx, pos_end[1] + dy)
                if not (_in_board(*neg_outer) and _in_board(*pos_outer)):
                    continue
                outers = [neg_outer, pos_outer]
        else:
            # chong4：恰好一端被阻挡（对手棋子或边界），另一端开放
            if neg_valid and pos_valid:
                # 两端都在盘内，随机选一端用对手棋子阻挡
                if rg.choice((True, False)):
                    blocked_positions = [neg_end]
                    attack_positions = [pos_end]
                else:
                    blocked_positions = [pos_end]
                    attack_positions = [neg_end]
            elif neg_valid and not pos_valid:
                # pos 端是边界（天然阻挡），开放端是 neg
                attack_positions = [neg_end]
            elif pos_valid and not neg_valid:
                attack_positions = [pos_end]
            else:
                # 两端都在盘内之外的情况已按 neg/pos 分支处理；
                # 剩下的极端情形：两端都越界（连珠紧贴两条边界，长度恰好 4）
                # → 不存在任何可放的 chong4 进攻位，整盘重试
                continue

        # ----- 构建棋盘 -----
        board: List[int] = [0] * NUM_CELLS
        for x, y in stone_positions:
            board[_idx(x, y)] = player
        for x, y in blocked_positions:
            board[_idx(x, y)] = opponent

        # ----- protected 集合：禁止干扰子覆盖 -----
        # 覆盖任何一个元素都会破坏棋型或进攻位判定，故全部锁死：
        # - stone_positions：连珠棋子本体
        # - blocked_positions：chong4 被阻挡端的对手棋子
        # - attack_positions：进攻位（必须保持空位才能判定 detect）
        # - outers：huo3/huo2 外侧位（保持空位避免落子后退化为冲四/眠三）
        protected = set(stone_positions)
        protected.update(blocked_positions)
        protected.update(attack_positions)
        protected.update(outers)  # huo3/huo2 外侧位必须保持为空

        # ----- 放置干扰子 -----
        num_extra = rg.randint(0, max_extra) if max_extra > 0 else 0
        for _ in range(num_extra):
            for _ in range(_MAX_EXTRA_TRIES):
                rx = rg.randint(0, BOARD_SIZE - 1)
                ry = rg.randint(0, BOARD_SIZE - 1)
                if board[_idx(rx, ry)] != 0:
                    continue
                if (rx, ry) in protected:
                    continue
                board[_idx(rx, ry)] = rg.choice((player, opponent))
                break

        # ----- 最终 detect 校验：每个 attack_position 落子必须形成期望棋型 -----
        if not _verify_attack_positions(board, attack_positions, player, expect_shapes):
            continue

        attack_idx = [_idx(x, y) for x, y in attack_positions]
        return board, attack_idx, player

    return None


def _verify_attack_positions(board: List[int], attack_positions: List[Tuple[int, int]],
                              player: int, expect_shapes: Tuple[str, ...]) -> bool:
    """对每个进攻位做落子 detect 校验，任一不符合即返回 False"""
    for x, y in attack_positions:
        if board[_idx(x, y)] != 0:
            return False
        shape_names = {s["shape"] for s in detect_shapes(board, (x, y), player)}
        if not any(exp in shape_names for exp in expect_shapes):
            return False
    return True


# ---------------------------------------------------------------------------
# 防守棋型生成（委托 + 颜色翻转）
# ---------------------------------------------------------------------------

def _generate_defend(shape: str, rg: random.Random,
                     max_extra: int) -> Optional[Tuple[List[int], List[int], int]]:
    """防守棋型：复用进攻生成器，仅切换视角（不翻转棋盘）

    语义：
    - 进攻生成器返回的棋盘上，attacker 色形成棋型；attack_positions 是进攻位
    - 防守样本即同一棋盘换视角：current_player = attacker 的对手（被威胁方）
    - attack_positions 索引不变：对防守方而言，**这些坐标正是"对手若落子即
      形成致命威胁"的位置，也就是我方必须抢先占据的防守位**。reward_calculator
      的 max(己方, 对手 × 0.95) 机制会让这些位置天然获得极高 reward。
    - 不做颜色翻转：随机种子已保证颜色分布自然均衡，翻转反而会让 current_player
      与棋型持有方重合导致语义倒置。
    """
    attack_shape = _DEFEND_TO_ATTACK[shape]
    result = _generate_attack(attack_shape, rg, max_extra)
    if result is None:
        return None
    board, attack_idx, attacker = result
    defender = 3 - attacker
    return board, attack_idx, defender


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def generate_shape_board(shape: str, random_gen: random.Random,
                         max_extra: int = 4) -> Optional[Tuple[List[int], List[int], int]]:
    """生成一个目标棋型的棋盘

    Args:
        shape: SUPPORTED_SHAPES 之一
        random_gen: 随机源（保证可复现）
        max_extra: 每样本最多放几颗干扰子

    Returns:
        (board, attack_positions, current_player) 或 None（本次生成失败）
        - board: 长度 225 的棋盘列表
        - attack_positions: 进攻位 / 防守位索引列表（长度 1 或 2）
        - current_player: 视角方（进攻棋型=持有者，防守棋型=被威胁方）

    Raises:
        ValueError: shape 不在 SUPPORTED_SHAPES 内
    """
    if shape not in SUPPORTED_SHAPES:
        raise ValueError(
            f"不支持的 shape: {shape!r}，可选值: {SUPPORTED_SHAPES}")
    if shape in _ATTACK_SHAPES:
        return _generate_attack(shape, random_gen, max_extra)
    return _generate_defend(shape, random_gen, max_extra)


def create_sample(board: List[int], attack_positions: List[int],
                  current_player: int, shape: str,
                  reward_calculator: RewardCalculator) -> Dict:
    """将棋盘状态封装为训练样本

    Args:
        board: 长度 225 的棋盘列表
        attack_positions: 进攻位 / 防守位索引列表（元数据）
        current_player: 视角方（1=黑，2=白）
        shape: SUPPORTED_SHAPES 之一（元数据）
        reward_calculator: 用于计算每个合法空位的 reward

    Returns:
        样本字典：{input, output, mask, _attack_positions, _shape}
        - input: 长度 226（225 板面 + 1 current_player）
        - output: 长度 225，masked_softmax 后的概率
        - mask: 长度 225，1=合法空位，0=已落子
        - 下划线字段为元数据（训练代码忽略，仅供验证/可视化）
    """
    input_vector: List[float] = [float(v) for v in board] + [float(current_player)]

    mask: List[int] = [0] * NUM_CELLS
    raw_rewards: List[float] = [0.0] * NUM_CELLS

    for i in range(NUM_CELLS):
        if board[i] != 0:
            continue  # 已落子：mask=0, reward=0
        mask[i] = 1
        x, y = i // BOARD_SIZE, i % BOARD_SIZE
        raw_rewards[i] = float(reward_calculator.compute_position_reward(
            board, x, y, current_player))

    output = masked_softmax(raw_rewards, mask)

    return {
        "input":  input_vector,
        "output": output,
        "mask":   mask,
        "_attack_positions": list(attack_positions),
        "_shape": shape,
    }


# ---------------------------------------------------------------------------
# 验证：Hit@N 主指标 + 辅助指标
# ---------------------------------------------------------------------------

def verify_samples(samples: List[Dict]) -> Dict:
    """对一批样本做 label 质量体检

    Args:
        samples: create_sample 返回的样本列表，必须含 _attack_positions
            传入空列表也是合法的：返回的报告各计数为 0，stats 字段的
            min/avg/max 均为 0.0，failed_indices 为空列表。

    Returns:
        报告字典，字段见 design/sample_generator_design.md
    """
    total = len(samples)
    hit_n = hit_n1 = hit_n2 = 0
    attack_prob_sums: List[float] = []
    prob_gaps: List[float] = []
    failed_indices: List[int] = []

    for idx, sample in enumerate(samples):
        output = sample["output"]
        attack = sample["_attack_positions"]
        n = len(attack)
        attack_set = set(attack)

        # 按 output 降序排位
        ranked = sorted(range(NUM_CELLS), key=lambda i: output[i], reverse=True)
        top_n = set(ranked[:n])
        top_n1 = set(ranked[:n + 1])
        top_n2 = set(ranked[:n + 2])

        if attack_set.issubset(top_n):
            hit_n += 1
        elif len(failed_indices) < 100:
            failed_indices.append(idx)

        if attack_set.issubset(top_n1):
            hit_n1 += 1
        if attack_set.issubset(top_n2):
            hit_n2 += 1

        # 辅助指标
        prob_sum = sum(output[i] for i in attack)
        attack_prob_sums.append(prob_sum)
        mean_attack_prob = prob_sum / n if n > 0 else 0.0
        max_non_attack = 0.0
        for i in range(NUM_CELLS):
            if i in attack_set:
                continue
            if output[i] > max_non_attack:
                max_non_attack = output[i]
        prob_gaps.append(mean_attack_prob - max_non_attack)

    return {
        "total":             total,
        "hit_at_N":          hit_n,
        "hit_at_N_plus_1":   hit_n1,
        "hit_at_N_plus_2":   hit_n2,
        "attack_prob_sum":   _stats(attack_prob_sums),
        "prob_gap":          _stats(prob_gaps),
        "failed_indices":    failed_indices,
    }


def _stats(xs: List[float]) -> Dict[str, float]:
    """返回 {min, avg, max}（空列表返回 0）"""
    if not xs:
        return {"min": 0.0, "avg": 0.0, "max": 0.0}
    return {"min": min(xs), "avg": sum(xs) / len(xs), "max": max(xs)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(report: Dict, samples: List[Dict], print_failures: int) -> None:
    """打印验证报告"""
    total = report["total"]
    if total == 0:
        print("No samples.")
        return

    def pct(n):
        return f"{n}/{total} ({100 * n / total:.2f}%)"

    print("\n===== 样本验证报告 =====")
    print(f"总样本数: {total}")
    print(f"Hit@N    : {pct(report['hit_at_N'])}")
    print(f"Hit@N+1  : {pct(report['hit_at_N_plus_1'])}")
    print(f"Hit@N+2  : {pct(report['hit_at_N_plus_2'])}")
    s1 = report["attack_prob_sum"]
    print(f"attack_prob_sum: min={s1['min']:.4f}  avg={s1['avg']:.4f}  max={s1['max']:.4f}")
    s2 = report["prob_gap"]
    print(f"prob_gap       : min={s2['min']:.4f}  avg={s2['avg']:.4f}  max={s2['max']:.4f}")

    if print_failures <= 0 or not report["failed_indices"]:
        return
    k = min(print_failures, len(report["failed_indices"]))
    print(f"\n----- Hit@N 未命中样例（前 {k} 个） -----")
    for i in report["failed_indices"][:k]:
        s = samples[i]
        attack = s["_attack_positions"]
        output = s["output"]
        top5 = sorted(range(NUM_CELLS), key=lambda j: output[j], reverse=True)[:5]
        print(f"[样本 #{i}] shape={s.get('_shape')} attack={attack}")
        print(f"   top5 idx={top5}")
        print(f"   top5 prob={[round(output[j], 4) for j in top5]}")
        print(f"   attack prob={[round(output[j], 4) for j in attack]}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ai.sample_generator",
        description="棋型样本生成器（chong4 / huo4 / huo3 / huo2 / defend_chong4 / defend_huo3）")
    parser.add_argument("--shape", required=True, choices=list(SUPPORTED_SHAPES),
                        help="目标棋型")
    parser.add_argument("--num-samples", type=int, required=True,
                        help="目标样本数")
    parser.add_argument("--output", required=True,
                        help="输出 JSON 文件路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）。相同 seed + 相同参数组合可复现"
                             "完全一致的样本批次，便于调试 label / 对比实验。")
    parser.add_argument("--max-extra", type=int, default=4,
                        help="每样本最多干扰子数（默认 4）")
    parser.add_argument("--print-failures", type=int, default=5,
                        help="验证未命中样本打印条数（默认 5，0=不打印）")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口"""
    args = _parse_args(argv)

    rg = random.Random(args.seed)
    rc = RewardCalculator(RewardConfig())

    print(f"开始生成 {args.num_samples} 个 [{args.shape}] 样本 "
          f"(seed={args.seed}, max_extra={args.max_extra}) ...")

    samples: List[Dict] = []
    # 防止极端参数无限循环：允许的"失败次数"上限 = 目标数 × 200
    fail_budget = max(args.num_samples * 200, 2000)
    fails = 0

    while len(samples) < args.num_samples:
        result = generate_shape_board(args.shape, rg, args.max_extra)
        if result is None:
            fails += 1
            if fails > fail_budget:
                print(f"生成失败次数超过上限 {fail_budget}，提前终止。")
                break
            continue
        board, attack, player = result
        samples.append(create_sample(board, attack, player, args.shape, rc))

        if len(samples) % max(args.num_samples // 10, 500) == 0:
            print(f"  已生成 {len(samples)}/{args.num_samples}")

    if len(samples) < args.num_samples:
        print(f"警告：只生成了 {len(samples)}/{args.num_samples} 个样本")

    # 验证
    report = verify_samples(samples)
    _print_report(report, samples, args.print_failures)

    # 保存
    out_abs = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

    size_mb = os.path.getsize(out_abs) / 1024 / 1024
    print(f"\n已保存 {len(samples)} 个样本 → {out_abs} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
