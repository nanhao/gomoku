"""CnnV11TrainLoop 双代评估 单元测试

测试双代历史最强评估逻辑（intention 56 + intention 60 优化）：
- checkpoint 依次与当前代 best、上一代 best 对弈（intention 60：
  vs best 胜率 >= 55% 才继续 vs prev，否则跳过直接失败）
- 两场胜率都 >= 55% 才晋升
- 无上一代时仅单场评估
- best_win_rate 取两场较低者

复用 test_cnn_v11_train_loop.py 的辅助函数（mock trainer/reader/evaluator/logger）。
"""
import os
from unittest.mock import MagicMock, patch

from tests.ai.unit.test_cnn_v11_train_loop import (
    TEST_MCTS_CONFIG,
    _make_mock_evaluator,
    _make_mock_logger,
    _make_mock_reader,
    _make_mock_trainer,
)

from ai.cnn_v11_train_loop import CnnV11TrainLoop, TrainLoopResult


# ========== 通用 patch 上下文 ==========

def _common_patches():
    """返回双代评估测试所需的公共 patch 列表

    覆盖：registry、buffer reader、evaluator、logger、torch 导出、
    目录与文件操作（含 prev_best 文件存在性）。
    """
    return [
        patch("ai.cnn_v11_train_loop.registry.get_trainer",
              return_value=_make_mock_trainer()),
        patch("ai.cnn_v11_train_loop.SampleBufferReader",
              return_value=_make_mock_reader()),
        patch("ai.cnn_v11_train_loop.CheckpointEvaluator",
              return_value=_make_mock_evaluator()),
        patch("ai.cnn_v11_train_loop.LossLogger",
              return_value=_make_mock_logger()),
        patch("torch.jit.script", return_value=MagicMock()),
        patch("torch.jit.save", return_value=None),
        patch("torch.save", return_value=None),
        patch("os.makedirs", return_value=None),
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=[]),
        patch("shutil.move", return_value=None),
    ]


def _run_loop(model_spec, evaluator, prev_exists, **kwargs):
    """构造并运行训练循环（双代评估专用）

    Args:
        model_spec: 初始模型 spec
        evaluator: mock 评估器（自定义 side_effect/return_value）
        prev_exists: 上一代 best 文件是否存在（patch os.path.isfile）
        **kwargs: 覆盖训练循环参数

    Returns:
        (result, mock_evaluator)
    """
    # 捕获真实 isfile 引用，避免 fake_isfile 内部递归调用被 patch 的版本
    real_isfile = os.path.isfile

    patchers = _common_patches()
    for p in patchers:
        p.start()

    isfile_patch = None
    if prev_exists:
        # 仅对上一代路径返回存在；其余 isfile 调用保持真实检查
        def fake_isfile(path):
            if "gen" in str(path) and "torchscript" in str(path):
                return True
            return real_isfile(path)

        isfile_patch = patch("ai.cnn_v11_train_loop.os.path.isfile",
                             side_effect=fake_isfile)
        isfile_patch.start()

    evaluator_patch = None
    try:
        loop = CnnV11TrainLoop(
            model_spec=model_spec,
            buffer_dir="/fake/buffer",
            batch_size=512,
            updates_per_checkpoint=1,
            eval_games=50,
            max_consecutive_failures=kwargs.pop("max_consecutive_failures", 3),
            mcts_config=TEST_MCTS_CONFIG,
            lr=0.001,
            weight_decay=1e-4,
            **kwargs,
        )
        # 替换训练循环持有的 evaluator 为自定义 mock，精确控制两场胜率
        evaluator_patch = patch.object(loop, "_evaluator", evaluator)
        evaluator_patch.start()
        result = loop.run()
    finally:
        if evaluator_patch is not None:
            evaluator_patch.stop()
        if isfile_patch is not None:
            isfile_patch.stop()
        for p in patchers:
            p.stop()
    return result, evaluator


# ========== TC-TL-10: 双代都通过晋升 ==========

def test_promotion_requires_both_generations_pass():
    """两场评估（vs best、vs prev）都 >= 55% 时应晋升"""
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = [0.58, 0.57]  # vs best, vs prev

    result, evaluator = _run_loop(
        model_spec="cnn_v11.5",
        evaluator=evaluator,
        prev_exists=True,
        max_consecutive_failures=3,
    )

    assert isinstance(result, TrainLoopResult)
    assert result.stopped_by == "promoted"
    assert result.promoted_to == "cnn_v11.6"
    assert result.checkpoints_evaluated == 1
    assert result.win_rate_vs_best == 0.58
    assert result.win_rate_vs_prev == 0.57
    # 有效胜率取两场较低者
    assert result.best_win_rate == 0.57
    # evaluate 被调用 2 次（best、prev 各一次）
    assert evaluator.evaluate.call_count == 2


# ========== TC-TL-11: vs prev 失败不晋升 ==========

def test_fail_when_prev_generation_below_threshold():
    """vs best 通过但 vs prev 低于阈值时不应晋升，计一次失败"""
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = [0.60, 0.45]  # vs best 通过, vs prev 失败

    result, evaluator = _run_loop(
        model_spec="cnn_v11.5",
        evaluator=evaluator,
        prev_exists=True,
        max_consecutive_failures=1,  # 一次失败即停止，便于断言
    )

    assert result.stopped_by == "max_failures"
    assert result.promoted_to is None
    assert result.checkpoints_evaluated == 1
    # best_win_rate 取两场较低者
    assert result.best_win_rate == 0.45
    assert result.win_rate_vs_best == 0.60
    assert result.win_rate_vs_prev == 0.45


# ========== TC-TL-12: 无上一代单场评估 ==========

def test_single_generation_evaluation_when_no_prev():
    """gen1 无上一代时仅单场评估，晋升判定只看 vs best"""
    evaluator = _make_mock_evaluator(win_rate=0.58)

    result, evaluator = _run_loop(
        model_spec="cnn_v11.1",
        evaluator=evaluator,
        prev_exists=False,
        max_consecutive_failures=3,
    )

    assert result.stopped_by == "promoted"
    assert result.promoted_to == "cnn_v11.2"
    assert result.win_rate_vs_best == 0.58
    assert result.win_rate_vs_prev is None
    assert result.best_win_rate == 0.58
    # 仅调用 1 次 evaluate
    assert evaluator.evaluate.call_count == 1


# ========== TC-TL-13: 双代判定失败后重试晋升 ==========

def test_promote_after_prev_failure_retry():
    """第 1 轮 vs prev 失败，第 2 轮双代都通过时晋升"""
    evaluator = MagicMock()
    # 第 1 轮: vs best=0.60, vs prev=0.45；第 2 轮: vs best=0.58, vs prev=0.57
    evaluator.evaluate.side_effect = [0.60, 0.45, 0.58, 0.57]

    result, evaluator = _run_loop(
        model_spec="cnn_v11.5",
        evaluator=evaluator,
        prev_exists=True,
        max_consecutive_failures=3,
    )

    assert result.stopped_by == "promoted"
    assert result.promoted_to == "cnn_v11.6"
    assert result.checkpoints_evaluated == 2
    # 有效胜率：max(min(0.60,0.45), min(0.58,0.57)) = max(0.45, 0.57) = 0.57
    assert result.best_win_rate == 0.57
    assert result.win_rate_vs_best == 0.58
    assert result.win_rate_vs_prev == 0.57
    assert evaluator.evaluate.call_count == 4


# ========== TC-TL-14: vs best 失败时跳过 vs prev 评估 ==========

def test_skip_prev_evaluation_when_best_below_threshold():
    """vs best 胜率未达 55% 时直接失败，不再评估 vs prev（短路省时）"""
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = [0.45]  # 仅 vs best（失败）

    result, evaluator = _run_loop(
        model_spec="cnn_v11.5",
        evaluator=evaluator,
        prev_exists=True,  # 上一代存在，但不应触发其评估
        max_consecutive_failures=1,  # 一次失败即停止，便于断言
    )

    assert result.stopped_by == "max_failures"
    assert result.promoted_to is None
    assert result.checkpoints_evaluated == 1
    assert result.win_rate_vs_best == 0.45
    # 跳过 vs prev：未评估，字段为 None
    assert result.win_rate_vs_prev is None
    # 有效胜率即 vs best 胜率
    assert result.best_win_rate == 0.45
    # 仅调用 1 次 evaluate（vs best 失败即短路）
    assert evaluator.evaluate.call_count == 1
