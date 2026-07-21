"""MCTS 配置数据类模块

包含 MctsSelfPlayConfig，被 mcts_game_runner、mcts_self_play_worker、
mcts_battle_worker 等多个模块共用。
"""

from dataclasses import dataclass


@dataclass
class MctsSelfPlayConfig:
    """MCTS 自对弈/对战超参数配置

    替代旧的 hyperparams 字典，符合"结构化数据用 dataclass"规范。
    所有字段均有默认值，与当前 mcts_self_play.py 的 CLI 默认值一致。
    """
    simulations: int = 800
    c_puct: float = 2.0
    temp_steps: int = 8
    temperature: float = 1.0
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.03
    batch_size: int = 225
    virtual_loss: float = 3.0
    use_prior: bool = True
