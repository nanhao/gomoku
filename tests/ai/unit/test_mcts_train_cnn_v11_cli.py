"""mcts_train_cnn_v11 CLI 单元测试

测试 CLI 参数解析、默认值、边界校验。
"""
import sys
from unittest.mock import MagicMock, patch

import pytest


# ========== TC-CLI-01: 必填参数缺失 ==========

def test_missing_required_model_param():
    """不带 --model 时应报错并打印 usage"""
    test_args = ["mcts_train_cnn_v11.py"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            from ai.scripts.mcts_train_cnn_v11 import main
            main()
    assert exc_info.value.code != 0  # exit code ≠ 0


# ========== TC-CLI-02: 参数边界校验 ==========

def test_batch_size_zero_rejected():
    """--batch-size 0 应被拒绝"""
    test_args = [
        "mcts_train_cnn_v11.py",
        "--model", "cnn_v11.1",
        "--batch-size", "0",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            from ai.scripts.mcts_train_cnn_v11 import main
            main()
    assert exc_info.value.code != 0


def test_updates_per_checkpoint_zero_rejected():
    """--updates-per-checkpoint 0 应被拒绝"""
    test_args = [
        "mcts_train_cnn_v11.py",
        "--model", "cnn_v11.1",
        "--updates-per-checkpoint", "0",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            from ai.scripts.mcts_train_cnn_v11 import main
            main()
    assert exc_info.value.code != 0


# ========== TC-CLI-03: 参数类型校验 ==========

def test_eval_games_invalid_type():
    """--eval-games abc 应被 argparse 拒绝"""
    test_args = [
        "mcts_train_cnn_v11.py",
        "--model", "cnn_v11.1",
        "--eval-games", "abc",
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            from ai.scripts.mcts_train_cnn_v11 import main
            main()
    assert exc_info.value.code != 0


# ========== TC-CLI-04: 默认值生效 ==========

def test_defaults_applied():
    """仅传 --model 时其余参数使用默认值"""
    # 清除缓存模块，确保 mock 正确生效
    sys.modules.pop("ai.scripts.mcts_train_cnn_v11", None)

    test_args = [
        "mcts_train_cnn_v11.py",
        "--model", "cnn_v11.1",
    ]
    with patch.object(sys, "argv", test_args):
        with patch("ai.scripts.mcts_train_cnn_v11.CnnV11TrainLoop") as mock_loop:
            mock_result = MagicMock()
            mock_result.stopped_by = "promoted"
            mock_result.promoted_to = "cnn_v11.2"
            mock_result.total_updates = 50
            mock_result.checkpoints_evaluated = 1
            mock_result.best_win_rate = 0.62
            mock_result.loss_history = [1.0, 0.9]
            mock_loop.return_value.run.return_value = mock_result
            from ai.scripts.mcts_train_cnn_v11 import main
            main()

    # 验证 CnnV11TrainLoop 以默认参数构造
    call_kwargs = mock_loop.call_args.kwargs
    assert call_kwargs["buffer_dir"] == "sample_buffer"
    assert call_kwargs["batch_size"] == 512
    assert call_kwargs["updates_per_checkpoint"] == 50
    assert call_kwargs["eval_games"] == 50
    assert call_kwargs["max_consecutive_failures"] == 3
    assert call_kwargs["lr"] == 0.001
    assert call_kwargs["weight_decay"] == 1e-4


# ========== TC-CLI-05: 全部自定义参数 ==========

def test_all_params_custom():
    """所有参数自定义时应正确传递"""
    # 清除缓存模块，确保 mock 正确生效
    sys.modules.pop("ai.scripts.mcts_train_cnn_v11", None)

    test_args = [
        "mcts_train_cnn_v11.py",
        "--model", "cnn_v11.3",
        "--buffer-dir", "/custom/buffer",
        "--batch-size", "256",
        "--updates-per-checkpoint", "20",
        "--eval-games", "30",
        "--max-failures", "5",
        "--simulations", "400",
        "--c-puct", "1.5",
        "--lr", "0.0005",
        "--weight-decay", "0.01",
    ]
    with patch.object(sys, "argv", test_args):
        with patch("ai.scripts.mcts_train_cnn_v11.CnnV11TrainLoop") as mock_loop:
            mock_result = MagicMock()
            mock_result.stopped_by = "promoted"
            mock_result.promoted_to = "cnn_v11.4"
            mock_result.total_updates = 100
            mock_result.checkpoints_evaluated = 5
            mock_result.best_win_rate = 0.75
            mock_result.loss_history = [1.0]
            mock_loop.return_value.run.return_value = mock_result
            from ai.scripts.mcts_train_cnn_v11 import main
            main()

    call_kwargs = mock_loop.call_args.kwargs
    assert call_kwargs["model_spec"] == "cnn_v11.3"
    assert call_kwargs["buffer_dir"] == "/custom/buffer"
    assert call_kwargs["batch_size"] == 256
    assert call_kwargs["updates_per_checkpoint"] == 20
    assert call_kwargs["eval_games"] == 30
    assert call_kwargs["max_consecutive_failures"] == 5
    assert call_kwargs["lr"] == 0.0005
    assert call_kwargs["weight_decay"] == 0.01
    # 验证 MCTS 配置
    assert call_kwargs["mcts_config"].simulations == 400
    assert call_kwargs["mcts_config"].c_puct == 1.5
