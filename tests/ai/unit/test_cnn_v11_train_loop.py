"""CnnV11TrainLoop 单元测试

测试训练-评估循环的控制流逻辑（晋升/失败/停止条件）。
核心组件（trainer、buffer、evaluator）均使用 mock 隔离。
"""
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from ai.scripts.mcts_config import MctsSelfPlayConfig


# ========== 辅助函数 ==========

def _make_sample(state=None, pi=None, z=1.0):
    """创建一条模拟样本"""
    if state is None:
        state = [0] * 226
    if pi is None:
        pi = [1.0 / 225] * 225
    return {"state": state, "pi": pi, "z": z}


def _make_mock_trainer():
    """创建 mock trainer，支持 _train_step、save_model"""
    trainer = MagicMock()
    trainer._train_step.return_value = {
        "value_loss": 0.5,
        "policy_loss": 1.2,
        "total_loss": 1.7,
        "grad_norms": {"layer1": 0.1},
    }
    trainer.save_model.return_value = "/fake/checkpoint.pth"
    trainer.model = MagicMock()
    trainer.model.network = MagicMock()
    trainer.model_name = "cnn_v11"
    trainer.model.gen = 1
    trainer.gen = 1
    trainer.device = torch.device("cpu")
    return trainer


def _make_mock_reader(total_samples=1000):
    """创建 mock SampleBufferReader"""
    reader = MagicMock()
    reader.total_samples.return_value = total_samples
    reader.refresh.return_value = 0
    reader.random_sample.return_value = [_make_sample() for _ in range(512)]
    return reader


def _make_mock_evaluator(win_rate=0.6):
    """创建 mock CheckpointEvaluator"""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = win_rate
    return evaluator


def _make_mock_logger():
    """创建 mock LossLogger"""
    logger = MagicMock()
    return logger


TEST_MCTS_CONFIG = MctsSelfPlayConfig(
    simulations=800,
    c_puct=2.0,
    temp_steps=2,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
    virtual_loss=3.0,
    use_prior=True,
)


# ========== TC-TL-01: 正常晋升 ==========

def test_promotion_on_high_win_rate():
    """checkpoint 胜率 >= 55% 时应触发晋升并停止"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_reader = _make_mock_reader()
    mock_evaluator = _make_mock_evaluator(win_rate=0.58)
    mock_logger = _make_mock_logger()

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=5,  # 用较小值加速测试
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    assert isinstance(result, TrainLoopResult)
    assert result.stopped_by == "promoted"
    assert result.promoted_to == "cnn_v11.2"
    assert result.checkpoints_evaluated == 1
    assert result.total_updates == 5
    assert result.best_win_rate >= 0.55
    assert len(result.loss_history) == 5


# ========== TC-TL-02: 连续失败停止 ==========

def test_max_failures_stop():
    """连续 3 次 checkpoint 胜率 < 55% 应停止训练"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_reader = _make_mock_reader()
    mock_evaluator = _make_mock_evaluator(win_rate=0.45)  # 始终低于 55%
    mock_logger = _make_mock_logger()

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=3,
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    assert result.stopped_by == "max_failures"
    assert result.promoted_to is None
    assert result.checkpoints_evaluated == 3
    assert result.total_updates == 9  # 3 checkpoints × 3 updates
    assert result.best_win_rate < 0.55


# ========== TC-TL-03: 先失败后晋升 ==========

def test_fail_then_promote():
    """第 1 次失败，第 2 次晋升"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_reader = _make_mock_reader()
    mock_logger = _make_mock_logger()

    # evaluator 先返回低胜率，再返回高胜率
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.side_effect = [0.45, 0.56]

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=2,
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    assert result.stopped_by == "promoted"
    assert result.promoted_to == "cnn_v11.2"
    assert result.checkpoints_evaluated == 2
    assert result.total_updates == 4  # 2 checkpoints × 2 updates


# ========== TC-TL-04: 样本不足时使用全部 ==========

def test_insufficient_samples_uses_all():
    """缓冲区仅 200 条，batch_size=512 时使用全部样本训练"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    # 模拟返回 200 条样本
    samples_200 = [_make_sample() for _ in range(200)]
    mock_reader = _make_mock_reader(total_samples=200)
    mock_reader.random_sample.return_value = samples_200
    mock_evaluator = _make_mock_evaluator(win_rate=0.45)
    mock_logger = _make_mock_logger()

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=1,
            eval_games=50,
            max_consecutive_failures=1,  # 只评估 1 次即停止
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    # 应该正常完成（使用全部 200 条训练）
    assert result.total_updates == 1
    assert result.checkpoints_evaluated == 1
    # 验证 _convert_inputs 收到 200 条 states
    convert_call = mock_trainer._convert_inputs.call_args
    assert len(convert_call[0][0]) == 200  # states list length


# ========== TC-TL-05: 初始模型不存在 ==========

def test_model_not_found_raises():
    """model_spec 对应的 gen 目录不存在时应报错"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               side_effect=ValueError("多代模型必须指定代数")), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=_make_mock_reader()), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=_make_mock_evaluator()), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=_make_mock_logger()), \
         patch("os.path.isdir", return_value=False), \
         patch("os.makedirs", return_value=None):
        with pytest.raises(ValueError, match="多代模型必须指定代数"):
            CnnV11TrainLoop(
                model_spec="cnn_v11.99",
                buffer_dir="/fake/buffer",
                batch_size=512,
                updates_per_checkpoint=50,
                eval_games=50,
                max_consecutive_failures=3,
                mcts_config=TEST_MCTS_CONFIG,
                lr=0.001,
                weight_decay=1e-4,
            )


# ========== TC-TL-06: loss 历史记录 ==========

def test_loss_history_collected():
    """每个 mini-batch 的 total_loss 应被记录到 loss_history"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_trainer._train_step.return_value = {
        "value_loss": 0.3,
        "policy_loss": 0.8,
        "total_loss": 1.1,
        "grad_norms": {"layer1": 0.05},
    }
    mock_reader = _make_mock_reader()
    mock_evaluator = _make_mock_evaluator(win_rate=0.45)
    mock_logger = _make_mock_logger()

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=2,
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    # 6 个 mini-batch（2 per checkpoint × 3 checkpoints 直到 max_failures）
    assert result.total_updates == 6
    assert len(result.loss_history) == 6
    assert result.loss_history[0] == 1.1
    assert result.loss_history[1] == 1.1


# ========== TC-TL-07: best TorchScript 缓存 ==========

def test_best_torchscript_cached():
    """best TorchScript 只在初始化时导出一次，后续评估复用"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_reader = _make_mock_reader()
    mock_evaluator = _make_mock_evaluator(win_rate=0.45)
    mock_logger = _make_mock_logger()

    mock_scripted = MagicMock()

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=mock_scripted) as mock_script, \
         patch("torch.jit.save", return_value=None) as mock_jit_save, \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=1,
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    # torch.jit.script 调用次数 = 1(best) + 3(checkpoint) = 4
    assert mock_script.call_count == 4
    # torch.jit.save 调用次数 = 1(best) + 3(checkpoint) = 4
    assert mock_jit_save.call_count == 4


# ========== TC-TL-08: 晋升后 gen 目录正确 ==========

def test_promotion_gen_directory_correct():
    """初始 gen=5，晋升后应为 cnn_v11.6"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_trainer.gen = 5
    mock_trainer.model.gen = 5
    mock_reader = _make_mock_reader()
    mock_evaluator = _make_mock_evaluator(win_rate=0.58)
    mock_logger = _make_mock_logger()

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.5",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=1,
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    assert result.stopped_by == "promoted"
    assert result.promoted_to == "cnn_v11.6"


# ========== TC-TL-09: best_win_rate 追踪 ==========

def test_best_win_rate_tracked():
    """多次评估时 best_win_rate 应记录所有 checkpoint 中的最高胜率"""
    from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult

    mock_trainer = _make_mock_trainer()
    mock_reader = _make_mock_reader()
    mock_logger = _make_mock_logger()

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.side_effect = [0.42, 0.48, 0.44]

    with patch("ai.cnn_v11_train_loop.registry.get_trainer",
               return_value=mock_trainer), \
         patch("ai.cnn_v11_train_loop.SampleBufferReader",
               return_value=mock_reader), \
         patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
               return_value=mock_evaluator), \
         patch("ai.cnn_v11_train_loop.LossLogger",
               return_value=mock_logger), \
         patch("torch.jit.script", return_value=MagicMock()), \
         patch("torch.jit.save", return_value=None), \
         patch("torch.save", return_value=None), \
         patch("os.makedirs", return_value=None), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=[]), \
         patch("shutil.move", return_value=None):
        loop = CnnV11TrainLoop(
            model_spec="cnn_v11.1",
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=1,
            eval_games=50,
            max_consecutive_failures=3,
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
        )
        result = loop.run()

    assert result.stopped_by == "max_failures"
    assert result.best_win_rate == pytest.approx(0.48)
    assert result.checkpoints_evaluated == 3
