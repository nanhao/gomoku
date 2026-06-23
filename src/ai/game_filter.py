"""对局级样本过滤判定模块

提供 :func:`should_keep_game` 函数，判定单局对弈是否应保留用于样本生成。

过滤规则：
- 非单方采样（temp_side_this_game 为 None 或 "both"）→ 不过滤，全部保留
- 单方采样且采样方获胜 → 保留
- 单方采样且贪婪方获胜或平局 → 丢弃

该模块供 :mod:`ai.create_samples` 在生成训练样本时调用，
通过 :class:`~ai.sample_creator.SampleCreator` 的 ``record_filter`` 参数注入。
"""

import logging
from typing import Optional


__all__ = ["should_keep_game"]

_logger = logging.getLogger(__name__)


def should_keep_game(record: dict) -> bool:
    """判定单局对弈是否应保留用于样本生成。

    规则：
    - temp_side_this_game 为 None / "both" → True（非单方采样，不过滤）
    - temp_side_this_game == "black" 且 winner == 1 → True（采样方黑胜）
    - temp_side_this_game == "white" 且 winner == 2 → True（采样方白胜）
    - 其余 → False（贪婪方胜或平局）

    Args:
        record: record.json 解析后的字典，需包含 ``winner`` 字段，
            可选 ``temp_side_this_game`` 字段。旧格式 record.json 不含
            ``temp_side_this_game`` 时，``record.get()`` 返回 None，
            等同 null 语义，行为与修改前一致。

    Returns:
        True → 保留该对局；False → 丢弃
    """
    # 使用 get() 兼容旧格式 record.json（无 temp_side_this_game 字段）
    temp_side: Optional[str] = record.get("temp_side_this_game")
    raw_winner = record.get("winner", 0)
    # 防御性类型转换：确保 winner 为 int（旧数据或异常写入可能为 str）
    try:
        winner: int = int(raw_winner)
    except (TypeError, ValueError):
        _logger.warning("Invalid winner value in record: %r, treating as 0", raw_winner)
        winner = 0

    # 非单方采样模式：不过滤
    if temp_side is None or temp_side == "both":
        return True

    # 单方采样：采样方获胜才保留
    if temp_side == "black" and winner == 1:
        return True
    if temp_side == "white" and winner == 2:
        return True

    # 非法 temp_side_this_game 值：记录 warning 以便排查
    if temp_side not in ("black", "white"):
        _logger.warning(
            "Unexpected temp_side_this_game value: %r, discarding game", temp_side
        )

    # 贪婪方胜、平局、未结束等
    return False
