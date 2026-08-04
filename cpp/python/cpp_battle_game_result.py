"""C++ MCTS 对战结果数据类

定义 CppBattleGameResult，表示单局 C++ MCTS 对战的结果。
供 CppMctsBattleAdapter 使用，字段从 C++ GameResult 转换而来。
"""

from dataclasses import dataclass


@dataclass
class CppBattleGameResult:
    """单局 C++ MCTS 对战结果

    由 CppMctsBattleAdapter.get_results() 返回，字段均从 C++ GameResult
    转换而来。model1_won / model2_won 由 adapter 层根据 winner 和执方计算。

    Attributes:
        game_index: 局编号 0..N-1
        winner: 获胜方（0=平局, 1=黑胜, 2=白胜）
        model1_color: model1 执方（1=黑, 2=白）
        model2_color: model2 执方（1=黑, 2=白）
        model1_won: model1 是否获胜
        model2_won: model2 是否获胜
        move_count: 本局总步数
    """
    game_index: int
    winner: int
    model1_color: int
    model2_color: int
    model1_won: bool
    model2_won: bool
    move_count: int
