"""五子棋游戏核心逻辑包"""

__version__ = '1.0.0'


def get_winner_text(winner: int) -> str:
    """将获胜方数字转换为描述文本

    统一用词为"黑方获胜"、"白方获胜"、"平局"，与 GUI 风格一致。

    Args:
        winner: 获胜方编号
            0 = 未结束
            1 = 黑方获胜
            2 = 白方获胜
            3 = 平局

    Returns:
        获胜方描述文本；未结束或无效值返回空字符串
    """
    if winner == 1:
        return "黑方获胜"
    elif winner == 2:
        return "白方获胜"
    elif winner == 3:
        return "平局"
    return ""


def get_player_name(player: int) -> str:
    """将玩家数字转换为名称

    Args:
        player: 玩家编号
            1 = 黑方
            2 = 白方

    Returns:
        玩家名称；无效值返回空字符串
    """
    if player == 1:
        return "黑方"
    elif player == 2:
        return "白方"
    return ""


from .controller import GomokuController

__all__ = [
    'GomokuController',
    'get_winner_text',
    'get_player_name',
]
