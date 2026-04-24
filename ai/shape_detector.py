"""棋型检测引擎

纯棋型检测逻辑，无状态，无副作用。
从 evaluator.py 提取，负责模拟落子后检测四个方向的棋型。

支持 Numba JIT 加速：当 numba 可用时，核心计算逻辑编译为机器码，
预期 10~30 倍加速。Numba 不可用时自动回退到纯 Python 实现。
"""

from typing import List, Dict, Tuple, Optional

# Numba JIT 加速支持
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

import numpy as np


# 四个检测方向：横向、纵向、主对角线、副对角线
DIRECTIONS = [
    (0, 1),   # 横向
    (1, 0),   # 纵向
    (1, 1),   # 主对角线
    (1, -1),  # 副对角线
]

# 方向名称映射
DIRECTION_NAMES = {
    (0, 1): "横向",
    (1, 0): "纵向",
    (1, 1): "主对角线",
    (1, -1): "副对角线",
}

# ==================== 棋型编码映射（Numba JIT 用）====================
# Numba @njit 不支持返回字符串，因此核心计算返回整数编码，
# 外层通过 _SHAPE_CODE_MAP 转换为棋型名称。
# 编码 0 表示无棋型（两端都堵或无效），不在映射中。
_SHAPE_CODE_MAP = {
    1: "连五",
    2: "活四",
    3: "冲四",
    4: "活三",
    5: "眠三",
    6: "活二",
    7: "眠二",
    8: "独一",
}


# ==================== Numba JIT 核心计算函数 ====================

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _detect_single_direction_jit(board_array, x, y, player, dx, dy):
        """JIT 编译的单方向棋型检测，返回整数编码

        board_array: np.int8 一维数组（225 维）
        返回: 棋型编码（0=无, 1=连五, 2=活四, 3=冲四, 4=活三, 5=眠三, 6=活二, 7=眠二, 8=独一）
        """
        opponent = 3 - player
        count = 1

        # 正方向计数
        nx = x + dx
        ny = y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board_array[nx * 15 + ny] == player:
            count += 1
            nx += dx
            ny += dy
        # 检查正方向是否被阻挡
        blocked_positive = 0
        if not (0 <= nx < 15 and 0 <= ny < 15):
            blocked_positive = 1
        elif board_array[nx * 15 + ny] == opponent:
            blocked_positive = 1

        # 反方向计数
        nx = x - dx
        ny = y - dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board_array[nx * 15 + ny] == player:
            count += 1
            nx -= dx
            ny -= dy
        # 检查反方向是否被阻挡
        blocked_negative = 0
        if not (0 <= nx < 15 and 0 <= ny < 15):
            blocked_negative = 1
        elif board_array[nx * 15 + ny] == opponent:
            blocked_negative = 1

        blocked_count = blocked_positive + blocked_negative

        # 判断棋型，返回整数编码
        if count >= 5:
            return 1  # 连五
        if count == 4:
            if blocked_count == 0:
                return 2  # 活四
            elif blocked_count == 1:
                return 3  # 冲四
            else:
                return 0  # 两端堵
        if count == 3:
            if blocked_count == 0:
                return 4  # 活三
            elif blocked_count == 1:
                return 5  # 眠三
            else:
                return 0  # 两端堵
        if count == 2:
            if blocked_count == 0:
                return 6  # 活二
            elif blocked_count == 1:
                return 7  # 眠二
            else:
                return 0  # 两端堵
        if count == 1:
            return 8  # 独一
        return 0
else:
    def _detect_single_direction_jit(board_array, x, y, player, dx, dy):
        """Numba 不可用时的占位函数，不应被直接调用"""
        raise RuntimeError("Numba 不可用，请使用纯 Python 版本")


def _detect_single_direction_pure_python(board, x, y, player, dx, dy):
    """纯 Python 版单方向棋型检测，作为 JIT 不可用时的回退

    Args:
        board: 棋盘状态（list 或 ndarray）
        x, y: 当前位置
        player: 玩家
        dx, dy: 检测方向

    Returns:
        棋型类型字符串，或 None
    """
    opponent = 3 - player

    # 统计该方向上连续己方棋子数（包含当前位置）
    count = 1

    # 正方向计数
    nx, ny = x + dx, y + dy
    while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
        count += 1
        nx += dx
        ny += dy
    # 检查正方向是否被阻挡
    blocked_positive = (not (0 <= nx < 15 and 0 <= ny < 15)) or (board[nx * 15 + ny] == opponent)

    # 反方向计数
    nx, ny = x - dx, y - dy
    while 0 <= nx < 15 and 0 <= ny < 15 and board[nx * 15 + ny] == player:
        count += 1
        nx -= dx
        ny -= dy
    # 检查反方向是否被阻挡
    blocked_negative = (not (0 <= nx < 15 and 0 <= ny < 15)) or (board[nx * 15 + ny] == opponent)

    # 计算被阻挡数
    blocked_count = int(blocked_positive) + int(blocked_negative)

    # 判断棋型
    if count >= 5:
        return "连五"
    if count == 4:
        if blocked_count == 0:
            return "活四"
        elif blocked_count == 1:
            return "冲四"
        else:
            return None
    if count == 3:
        if blocked_count == 0:
            return "活三"
        elif blocked_count == 1:
            return "眠三"
        else:
            return None
    if count == 2:
        if blocked_count == 0:
            return "活二"
        elif blocked_count == 1:
            return "眠二"
        else:
            return None
    if count == 1:
        return "独一"
    return None


def detect_shapes(board: List[int], position: Tuple[int, int], player: int) -> List[Dict]:
    """检测落子后形成的所有棋型

    Args:
        board: 当前棋盘状态
        position: 落子位置 (x, y)
        player: 落子方

    Returns:
        棋型列表，每项包含 {"shape": 棋型类型, "directions": [方向列表]}

    棋型汇总规则：
    - 多冲四：两个或以上方向形成冲四 -> 返回 "多冲四"
    - 冲四：仅一个方向形成冲四 -> 返回 "冲四"
    - 多活三：两个或以上方向形成活三 -> 返回 "多活三"
    - 活三：仅一个方向形成活三 -> 返回 "活三"
    - 眠三/活二：记录所有方向的数量，用于 count_directions
    """
    x, y = position
    raw_shapes = {}  # shape_type -> list of directions

    # 性能优化：原地模拟落子，避免 list(board) 拷贝
    # 调用方保证该位置为空（board[idx] == 0），检测后恢复原值
    idx = x * 15 + y
    original_value = board[idx]
    board[idx] = player

    # 检测四个方向
    for dx, dy in DIRECTIONS:
        shape_type = detect_single_direction(board, x, y, player, dx, dy)
        if shape_type:
            dir_name = DIRECTION_NAMES[(dx, dy)]
            if shape_type not in raw_shapes:
                raw_shapes[shape_type] = []
            raw_shapes[shape_type].append(dir_name)

    # 恢复原值
    board[idx] = original_value

    # 合并棋型：处理多冲四、多活三等
    result = []

    # 优先级：连五 > 活四 > 多冲四 > 冲四 > 多活三 > 活三 > 眠三 > 活二 > 眠二 > 独一
    # 如果有连五，直接返回
    if "连五" in raw_shapes:
        return [{"shape": "连五", "directions": raw_shapes["连五"]}]

    # 活四
    if "活四" in raw_shapes:
        result.append({"shape": "活四", "directions": raw_shapes["活四"]})

    # 冲四 -> 多冲四 或 冲四
    if "冲四" in raw_shapes:
        if len(raw_shapes["冲四"]) >= 2:
            result.append({"shape": "多冲四", "directions": raw_shapes["冲四"]})
        else:
            result.append({"shape": "冲四", "directions": raw_shapes["冲四"]})

    # 活三 -> 多活三 或 活三
    if "活三" in raw_shapes:
        if len(raw_shapes["活三"]) >= 2:
            result.append({"shape": "多活三", "directions": raw_shapes["活三"]})
        else:
            result.append({"shape": "活三", "directions": raw_shapes["活三"]})

    # 眠三（记录所有方向）
    if "眠三" in raw_shapes:
        result.append({"shape": "眠三", "directions": raw_shapes["眠三"]})

    # 活二（记录所有方向）
    if "活二" in raw_shapes:
        result.append({"shape": "活二", "directions": raw_shapes["活二"]})

    # 眠二
    if "眠二" in raw_shapes:
        result.append({"shape": "眠二", "directions": raw_shapes["眠二"]})

    # 独一（保留所有方向，即使同时存在其他棋型）
    if "独一" in raw_shapes:
        result.append({"shape": "独一", "directions": raw_shapes["独一"]})

    return result


def detect_single_direction(board, x: int, y: int,
                            player: int, dx: int, dy: int) -> Optional[str]:
    """检测单个方向上的棋型

    优先使用 Numba JIT 加速版本。JIT 不可用时回退到纯 Python 实现。

    Args:
        board: 棋盘状态（已模拟落子），支持 list[int] 或 np.ndarray
        x, y: 当前位置
        player: 玩家
        dx, dy: 检测方向

    Returns:
        棋型类型字符串，或 None
    """
    # 性能优化：Numba JIT 路径
    if _NUMBA_AVAILABLE:
        try:
            # 将 board 转为 numpy 数组（JIT 函数需要）
            if not isinstance(board, np.ndarray):
                board_array = np.array(board, dtype=np.int8)
            else:
                board_array = board
            code = _detect_single_direction_jit(board_array, x, y, player, dx, dy)
            # 将整数编码转换为棋型名称，0 表示无棋型
            return _SHAPE_CODE_MAP.get(code)
        except Exception:
            # JIT 失败时回退到纯 Python
            pass

    # 纯 Python 回退路径
    return _detect_single_direction_pure_python(board, x, y, player, dx, dy)
