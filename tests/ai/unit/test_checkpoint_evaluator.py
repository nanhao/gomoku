"""CheckpointEvaluator 单元测试

测试 checkpoint vs best 的 C++ MCTS 对战评估功能。
使用 mock 避免对 cpp_mcts 的实际依赖。
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from ai.scripts.mcts_config import MctsSelfPlayConfig


# ========== Fixtures ==========

@pytest.fixture
def default_config() -> MctsSelfPlayConfig:
    """创建默认 MCTS 配置"""
    return MctsSelfPlayConfig(
        simulations=800,
        c_puct=2.0,
        temp_steps=2,
        temperature=1.0,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.03,
        virtual_loss=3.0,
        use_prior=True,
    )


# ========== 辅助函数 ==========

def _make_cpp_result(winner: int, model1_color: int, model2_color: int) -> MagicMock:
    """创建模拟的 C++ GameResult 对象

    Args:
        winner: 获胜方 (0=平局, 1=黑, 2=白)
        model1_color: model1 执方
        model2_color: model2 执方

    Returns:
        带所需属性的 MagicMock
    """
    r = MagicMock()
    r.winner = winner
    r.model1_color = model1_color
    r.model2_color = model2_color
    r.move_count = 30
    return r


def _make_mock_cpp_mcts(mock_runner, mock_game_config_cls=None):
    """创建 mock 的 cpp_mcts 模块，注入 GameRunner 和 GameConfig

    Args:
        mock_runner: mock 的 GameRunner 实例（MagicMock）
        mock_game_config_cls: mock 的 GameConfig 类。若为 None，使用空 MagicMock

    Returns:
        mock 的 cpp_mcts MagicMock 模块
    """
    mock_module = MagicMock()
    mock_module.GameRunner = MagicMock(return_value=mock_runner)
    if mock_game_config_cls is not None:
        mock_module.GameConfig = mock_game_config_cls
    else:
        mock_module.GameConfig = MagicMock()
    return mock_module


# ========== TC-CE-01: 正常评估 ==========

def test_evaluate_returns_valid_win_rate(default_config):
    """使用 mock GameRunner 测试胜率计算是否正确"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    # checkpoint (model1) 胜 30 局，best (model2) 胜 20 局 → win_rate = 0.6
    results = []
    for i in range(50):
        if i < 30:
            r = _make_cpp_result(winner=1, model1_color=1, model2_color=2)
        else:
            # best 获胜：model1=黑(1), model2=白(2), winner=2(白)
            r = _make_cpp_result(winner=2, model1_color=1, model2_color=2)
        r.game_index = i
        results.append(r)

    mock_runner = MagicMock()
    mock_runner.get_results.return_value = results
    mock_runner.is_finished.return_value = True
    mock_runner.get_completed_count.return_value = 50
    mock_runner.get_total_count.return_value = 50

    mock_cpp_mcts = _make_mock_cpp_mcts(mock_runner)

    evaluator = CheckpointEvaluator(default_config)
    with patch("os.path.isfile", return_value=True), \
         patch.dict("sys.modules", {"cpp_mcts": mock_cpp_mcts}):
        win_rate = evaluator.evaluate(
            checkpoint_ts_path="/fake/checkpoint.pt",
            best_ts_path="/fake/best.pt",
            num_games=50,
            temp_steps=2,
        )

    assert 0.59 <= win_rate <= 0.61  # 约 30/50 = 0.6


# ========== TC-CE-02: 模型文件不存在 ==========

def test_evaluate_file_not_found(default_config):
    """TorchScript 文件不存在时应抛出 FileNotFoundError"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    evaluator = CheckpointEvaluator(default_config)
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate(
                checkpoint_ts_path="/nonexistent/checkpoint.pt",
                best_ts_path="/nonexistent/best.pt",
                num_games=50,
                temp_steps=2,
            )


def test_evaluate_file_not_found_best_only(default_config):
    """仅 best TorchScript 文件不存在时应抛出 FileNotFoundError"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    evaluator = CheckpointEvaluator(default_config)
    # checkpoint 存在，best 不存在
    def fake_isfile(path):
        if "checkpoint" in path:
            return True
        return False

    with patch("os.path.isfile", side_effect=fake_isfile):
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate(
                checkpoint_ts_path="/fake/checkpoint.pt",
                best_ts_path="/fake/best.pt",
                num_games=50,
                temp_steps=2,
            )


# ========== TC-CE-03: temp_steps 参数传递 ==========

def test_evaluate_passes_temp_steps_to_config(default_config):
    """验证 temp_steps 参数被正确传递给 GameConfig"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    results = [_make_cpp_result(winner=1, model1_color=1, model2_color=2) for _ in range(50)]
    for i, r in enumerate(results):
        r.game_index = i

    mock_runner = MagicMock()
    mock_runner.get_results.return_value = results
    mock_runner.is_finished.return_value = True
    mock_runner.get_completed_count.return_value = 50
    mock_runner.get_total_count.return_value = 50

    mock_game_config_cls = MagicMock()
    mock_config_instance = MagicMock()
    mock_game_config_cls.return_value = mock_config_instance

    evaluator = CheckpointEvaluator(default_config)

    # 测试 temp_steps=3
    mock_cpp_mcts = _make_mock_cpp_mcts(mock_runner, mock_game_config_cls)
    with patch("os.path.isfile", return_value=True), \
         patch.dict("sys.modules", {"cpp_mcts": mock_cpp_mcts}):
        evaluator.evaluate(
            checkpoint_ts_path="/fake/checkpoint.pt",
            best_ts_path="/fake/best.pt",
            num_games=50,
            temp_steps=3,
        )

    # 验证 temp_steps 被设置到 GameConfig
    assert mock_config_instance.temp_steps == 3

    # 测试 temp_steps=0
    mock_cpp_mcts2 = _make_mock_cpp_mcts(mock_runner, mock_game_config_cls)
    with patch("os.path.isfile", return_value=True), \
         patch.dict("sys.modules", {"cpp_mcts": mock_cpp_mcts2}):
        evaluator.evaluate(
            checkpoint_ts_path="/fake/checkpoint.pt",
            best_ts_path="/fake/best.pt",
            num_games=50,
            temp_steps=0,
        )

    assert mock_config_instance.temp_steps == 0


# ========== TC-CE-04: 双方执方均匀 ==========

def test_evaluate_alternating_sides(default_config):
    """验证 checkpoint 和 best 各执黑执白次数大致均等"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    # 50 局中，model1 各执黑/白各半，每局 model1 均获胜
    # 因此 winner == model1_color 对每局成立
    results = []
    for i in range(50):
        m1_color = 1 if i % 2 == 0 else 2
        m2_color = 3 - m1_color
        r = _make_cpp_result(winner=m1_color, model1_color=m1_color, model2_color=m2_color)
        r.game_index = i
        results.append(r)

    mock_runner = MagicMock()
    mock_runner.get_results.return_value = results
    mock_runner.is_finished.return_value = True
    mock_runner.get_completed_count.return_value = 50
    mock_runner.get_total_count.return_value = 50

    mock_cpp_mcts = _make_mock_cpp_mcts(mock_runner)

    evaluator = CheckpointEvaluator(default_config)
    with patch("os.path.isfile", return_value=True), \
         patch.dict("sys.modules", {"cpp_mcts": mock_cpp_mcts}):
        win_rate = evaluator.evaluate(
            checkpoint_ts_path="/fake/checkpoint.pt",
            best_ts_path="/fake/best.pt",
            num_games=50,
            temp_steps=2,
        )

    # model1 胜率应为 100%（所有局 model1 胜）
    assert win_rate == 1.0


# ========== 边界: 平局 ==========

def test_evaluate_with_draws(default_config):
    """平局不计入任何一方的胜场"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    results = []
    for i in range(50):
        if i < 20:
            r = _make_cpp_result(winner=1, model1_color=1, model2_color=2)
        elif i < 30:
            r = _make_cpp_result(winner=0, model1_color=1, model2_color=2)  # 平局
        else:
            # best 获胜：model1=黑(1), model2=白(2), winner=2(白)
            r = _make_cpp_result(winner=2, model1_color=1, model2_color=2)
        r.game_index = i
        results.append(r)

    mock_runner = MagicMock()
    mock_runner.get_results.return_value = results
    mock_runner.is_finished.return_value = True
    mock_runner.get_completed_count.return_value = 50
    mock_runner.get_total_count.return_value = 50

    mock_cpp_mcts = _make_mock_cpp_mcts(mock_runner)

    evaluator = CheckpointEvaluator(default_config)
    with patch("os.path.isfile", return_value=True), \
         patch.dict("sys.modules", {"cpp_mcts": mock_cpp_mcts}):
        win_rate = evaluator.evaluate(
            checkpoint_ts_path="/fake/checkpoint.pt",
            best_ts_path="/fake/best.pt",
            num_games=50,
            temp_steps=2,
        )

    # model1 胜 20 局，平 10 局，model2 胜 20 局
    # win_rate = 20 / 50 = 0.4
    assert 0.39 <= win_rate <= 0.41


# ========== 边界: 0 局 ==========

def test_evaluate_zero_games(default_config):
    """0 局对战时应返回 0.0"""
    from ai.checkpoint_evaluator import CheckpointEvaluator

    mock_runner = MagicMock()
    mock_runner.get_results.return_value = []
    mock_runner.is_finished.return_value = True
    mock_runner.get_completed_count.return_value = 0
    mock_runner.get_total_count.return_value = 0

    mock_cpp_mcts = _make_mock_cpp_mcts(mock_runner)

    evaluator = CheckpointEvaluator(default_config)
    with patch("os.path.isfile", return_value=True), \
         patch.dict("sys.modules", {"cpp_mcts": mock_cpp_mcts}):
        win_rate = evaluator.evaluate(
            checkpoint_ts_path="/fake/checkpoint.pt",
            best_ts_path="/fake/best.pt",
            num_games=0,
            temp_steps=2,
        )

    assert win_rate == 0.0
