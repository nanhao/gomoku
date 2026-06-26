"""测试 create_engine 不再回退到 RandomModel

S3 改造后，create_engine 移除了 except ValueError 回退逻辑，
未知模型名称时直接上抛 ValueError，不再静默回退到 RandomModel。
"""

import sys
import os
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.engine import create_engine


class TestCreateEngineNoFallback:
    """验证 create_engine 不再回退到 RandomModel"""

    def test_unknown_model_raises_value_error(self):
        """未知模型名称应直接抛出 ValueError，不回退"""
        with pytest.raises(ValueError, match="未知的模型"):
            create_engine("nonexistent_model")

    def test_multi_gen_without_gen_raises_value_error(self):
        """多代模型未指定代数应抛出 ValueError，不回退"""
        with pytest.raises(ValueError, match="多代模型必须指定代数"):
            create_engine("cnn_v3")

    def test_valid_model_still_works(self):
        """合法模型名称应正常创建引擎，并可完成端到端预测+落子。

        原版只断言 engine/model 非 None，无法发现"模型成功 import 但
        predict/get_move 损坏"的情况。补上端到端 smoke test。
        """
        engine = create_engine("random")
        assert engine is not None
        assert engine.model is not None

        # 端到端：predict 返回 225 维概率向量
        empty_board = [[0] * 15 for _ in range(15)]
        vector = engine.board_to_vector(empty_board, current_player=1)
        assert len(vector) == 226

        probs = engine.model.predict(vector)
        assert len(probs.policy) == 225, f"predict 应返回 225 维，实际 {len(probs.policy)}"
        assert all(p >= 0 for p in probs.policy), "概率分布不得为负"
        assert sum(probs.policy) > 0, "概率分布总和必须为正，否则无法采样"

        # 端到端：get_move 返回合法落子位置
        decision = engine.get_move(empty_board, current_player=1)
        x, y = decision.get_position()
        assert 0 <= x < 15 and 0 <= y < 15, f"落子位置越界: ({x}, {y})"
        assert empty_board[x][y] == 0, f"落子位置 ({x}, {y}) 必须为空"
