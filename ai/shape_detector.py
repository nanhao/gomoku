"""棋型检测引擎

纯棋型检测逻辑，无状态，无副作用。
从 evaluator.py 提取，负责模拟落子后检测四个方向的棋型。
"""

from typing import List, Dict, Tuple, Optional


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


def detect_single_direction(board: List[int], x: int, y: int,
                            player: int, dx: int, dy: int) -> Optional[str]:
    """检测单个方向上的棋型

    Args:
        board: 棋盘状态（已模拟落子）
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
        else:  # blocked_count == 2: 两端都被阻挡，不构成威胁
            return None

    if count == 3:
        if blocked_count == 0:
            return "活三"
        elif blocked_count == 1:
            return "眠三"
        else:  # 两端都被阻挡
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
