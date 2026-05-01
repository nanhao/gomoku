"""move_sampler 模块：自对弈开局温度采样。

对应设计文档：design/move_sampler_design.md。

职责：
1. 提供温度加权采样函数 :func:`temperature_sample`，按公式
   ``q_i ∝ p_i ** (1/T)`` 对模型输出概率做加温后加权抽样。
2. 提供 :class:`TempConfig` 封装自对弈温度策略的所有参数。
3. 提供 :func:`should_sample` / :func:`resolve_temp_side` 两个辅助函数，
   供 :mod:`ai.game_manager` 在单局对弈中编排"前 N 步温度采样 + 之后
   argmax""按方启用""按局抽签" 的行为。

本模块保持无状态（所有函数均为纯函数；数据类只做校验），温度切换的
编排由调用方在 :mod:`ai.game_manager` 中完成——engine 不感知 move_count。
"""

from __future__ import annotations

import random as _stdrandom
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 常量：temp_side 四种取值
# ---------------------------------------------------------------------------

#: 只对黑方启用温度采样（白方全程 argmax）
TEMP_SIDE_BLACK: str = "black"
#: 只对白方启用温度采样（黑方全程 argmax）
TEMP_SIDE_WHITE: str = "white"
#: 双方都启用温度采样（默认）
TEMP_SIDE_BOTH: str = "both"
#: 每局开始前抽签决定由黑或白单方启用
TEMP_SIDE_RANDOM: str = "random"

#: temp_side 合法取值白名单
SUPPORTED_TEMP_SIDES: Tuple[str, ...] = (
    TEMP_SIDE_BLACK,
    TEMP_SIDE_WHITE,
    TEMP_SIDE_BOTH,
    TEMP_SIDE_RANDOM,
)


# ---------------------------------------------------------------------------
# TempConfig 数据类
# ---------------------------------------------------------------------------

@dataclass
class TempConfig:
    """温度采样配置。

    本数据类封装"自对弈阶梯式温度采样"的所有可调参数。构造时通过
    :meth:`__post_init__` 做字段合法性校验——非法值立刻抛 ``ValueError``，
    不做静默兜底。

    Attributes:
        temperature: 温度值 T，必须 > 0。越大越随机；趋近 0 时等价 argmax。
        temp_moves: 阶梯切换的全局步数阈值。``move_count < temp_moves`` 的步
            用温度采样，其后切为 argmax。``temp_moves=0`` 表示"任何步都不用
            温度"（等同关闭）。必须 >= 0。
        temp_side: 启用采样的一方，取值见 :data:`SUPPORTED_TEMP_SIDES`。
        top_k: 可选截断阈值；采样前只保留概率前 K 个合法位置，其余置 0。
            ``None`` 表示不截断。必须为 None 或正整数。
        seed: 可选随机种子。``None`` 表示使用默认随机源（不可复现）；
            指定后 :mod:`ai.game_manager` 会据此构造 :class:`random.Random`
            用于按局抽签与采样。
    """

    temperature: float = 1.0
    temp_moves: int = 12
    temp_side: str = TEMP_SIDE_BOTH
    top_k: Optional[int] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        # 温度必须为正实数
        if not isinstance(self.temperature, (int, float)):
            raise ValueError(
                f"temperature 必须是数值类型，收到: {type(self.temperature).__name__}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature 必须 > 0，收到: {self.temperature}"
            )

        # temp_moves 允许 0（= 关闭），禁止负数
        if not isinstance(self.temp_moves, int) or isinstance(self.temp_moves, bool):
            raise ValueError(
                f"temp_moves 必须是整数，收到: {type(self.temp_moves).__name__}"
            )
        if self.temp_moves < 0:
            raise ValueError(f"temp_moves 必须 >= 0，收到: {self.temp_moves}")

        # temp_side 必须在白名单里
        if self.temp_side not in SUPPORTED_TEMP_SIDES:
            raise ValueError(
                f"temp_side 必须属于 {SUPPORTED_TEMP_SIDES}，收到: {self.temp_side!r}"
            )

        # top_k 要么 None，要么正整数
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
                raise ValueError(
                    f"top_k 必须是 None 或正整数，收到: "
                    f"{type(self.top_k).__name__}"
                )
            if self.top_k <= 0:
                raise ValueError(f"top_k 必须 > 0，收到: {self.top_k}")


# ---------------------------------------------------------------------------
# 核心函数：温度采样
# ---------------------------------------------------------------------------

def temperature_sample(
    valid_positions: List[Tuple[int, int]],
    valid_probs: List[float],
    temperature: float,
    top_k: Optional[int] = None,
    rng: Optional[_stdrandom.Random] = None,
) -> Tuple[int, int]:
    """对合法位置按温度加权采样，返回最终落子坐标。

    算法：
    1. 可选 top_k 截断：仅保留概率前 K 个位置，其余权重置 0。
    2. 温度变换：对每个概率取 ``p ** (1/T)``。
    3. 归一化；若所有权重都 <= 0（模型把概率压在已落子位或 top_k=0），
       退化为在全部合法位置上均匀抽样。
    4. 用 ``rng.choices`` 加权抽一个下标并返回对应位置。

    Args:
        valid_positions: 合法（空位）坐标列表，至少 1 个元素。
        valid_probs: 与 ``valid_positions`` 一一对应的模型概率（>= 0）。
        temperature: 温度 T，必须 > 0。
        top_k: 截断阈值；``None`` 或 ``>= len(valid_positions)`` 表示不截断。
        rng: 随机源；``None`` 时使用 :mod:`random` 模块全局源。

    Returns:
        Tuple[int, int]: 选中的 ``(x, y)`` 坐标。

    Raises:
        ValueError: ``temperature <= 0``、``valid_positions`` 为空、
            或与 ``valid_probs`` 长度不一致。
    """
    # 参数校验
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0，收到: {temperature}")
    n = len(valid_positions)
    if n == 0:
        raise ValueError("valid_positions 不能为空")
    if len(valid_probs) != n:
        raise ValueError(
            f"valid_positions 与 valid_probs 长度必须一致，"
            f"收到 {n} vs {len(valid_probs)}"
        )

    rng = rng if rng is not None else _stdrandom

    # ---- 1. top_k 截断 ----
    # 对原始概率复制一份（避免修改调用方数据），过滤概率后再做温度变换。
    filtered = [max(float(p), 0.0) for p in valid_probs]
    if top_k is not None and top_k < n:
        # 找到第 top_k 大的概率阈值，小于该阈值的位置权重归 0
        # 使用 sorted 副本的第 top_k 个（按降序），简洁但 O(n log n)；对
        # 本项目 n<=225 完全够用。
        threshold = sorted(filtered, reverse=True)[top_k - 1]
        # 严格小于阈值的位置归 0；等于阈值的位置保留（并列时由加权采样自行打破）
        filtered = [p if p >= threshold else 0.0 for p in filtered]

    # ---- 2. 温度变换 ----
    inv_T = 1.0 / temperature
    weights = [(p ** inv_T) if p > 0 else 0.0 for p in filtered]

    # ---- 3. 归一化 / 全零回退 ----
    total = sum(weights)
    if total <= 0:
        # 模型把全部概率压在已落子位置上，或 top_k 截断过后权重全零：
        # 退化为在合法位置上均匀抽样，避免抛异常。
        weights = [1.0] * n

    # ---- 4. 加权采样 ----
    idx = rng.choices(range(n), weights=weights, k=1)[0]
    return valid_positions[idx]


# ---------------------------------------------------------------------------
# 按步/按方判定
# ---------------------------------------------------------------------------

def should_sample(
    move_count: int,
    current_player: int,
    temp_config: TempConfig,
    temp_side_this_game: str,
) -> bool:
    """判定当前步是否应该用温度采样。

    Args:
        move_count: 该局已落子总数（黑白合计，从 0 起计；当前步尚未落子）。
        current_player: 当前落子方，1=黑 / 2=白。
        temp_config: 温度配置。
        temp_side_this_game: 本局已解析的 temp_side（只能取
            :data:`TEMP_SIDE_BLACK` / :data:`TEMP_SIDE_WHITE` /
            :data:`TEMP_SIDE_BOTH`；``random`` 应在入局时由
            :func:`resolve_temp_side` 预先解析）。

    Returns:
        ``True`` → 该步用温度采样；``False`` → 该步用 argmax。
    """
    # temp_moves=0 等同关闭
    if temp_config.temp_moves <= 0:
        return False
    # 超过阈值 → argmax
    if move_count >= temp_config.temp_moves:
        return False

    side = temp_side_this_game
    if side == TEMP_SIDE_BOTH:
        return True
    if side == TEMP_SIDE_BLACK and current_player == 1:
        return True
    if side == TEMP_SIDE_WHITE and current_player == 2:
        return True
    return False


def resolve_temp_side(
    temp_side: str,
    rng: Optional[_stdrandom.Random] = None,
) -> str:
    """把 temp_side 解析为当局生效值。

    - :data:`TEMP_SIDE_RANDOM` → 等概率返回 :data:`TEMP_SIDE_BLACK` 或
      :data:`TEMP_SIDE_WHITE`。
    - 其他合法值 → 原样返回。

    Args:
        temp_side: :attr:`TempConfig.temp_side` 字段值。
        rng: 随机源，仅在 ``temp_side == TEMP_SIDE_RANDOM`` 时生效；
            ``None`` 时使用 :mod:`random` 全局源。

    Returns:
        已解析的 side（不会再是 random）。

    Raises:
        ValueError: ``temp_side`` 不在 :data:`SUPPORTED_TEMP_SIDES`。
    """
    if temp_side not in SUPPORTED_TEMP_SIDES:
        raise ValueError(
            f"temp_side 必须属于 {SUPPORTED_TEMP_SIDES}，收到: {temp_side!r}"
        )
    if temp_side == TEMP_SIDE_RANDOM:
        r = rng if rng is not None else _stdrandom
        return r.choice([TEMP_SIDE_BLACK, TEMP_SIDE_WHITE])
    return temp_side
