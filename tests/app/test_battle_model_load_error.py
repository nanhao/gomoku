"""测试 BattleSession 模型加载异常转换

验证 BattleSession.run() 在 create_engine 抛出 ValueError 时，
将其转换为 ModelLoadError 并向上抛出。

测试场景（对应设计文档 §9.5）：
- BattleSession 黑方模型失败 → raise ModelLoadError
- BattleSession 白方模型失败 → raise ModelLoadError
"""

import sys
import os
import tempfile
import shutil
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import BattleSession
from app.exceptions import ModelLoadError


class TestBattleSessionModelLoadError:
    """BattleSession 模型加载异常转换测试"""

    def test_black_model_unknown_raises_model_load_error(self):
        """黑方 AI 名称未知时，run() 应抛出 ModelLoadError"""
        # 使用临时目录避免写入保护
        temp_dir = tempfile.mkdtemp(prefix="test_battle_err_")
        ai_data_dir = os.path.join(temp_dir, "ai_data")
        train_data_dir = os.path.join(temp_dir, "train_data")
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

        try:
            session = BattleSession(
                ai1_name="unknown",
                ai2_name="random",
                num_games=1,
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
            )
            with pytest.raises(ModelLoadError, match="模型加载失败"):
                session.run()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_white_model_unknown_raises_model_load_error(self):
        """白方 AI 名称未知时，run() 应抛出 ModelLoadError"""
        temp_dir = tempfile.mkdtemp(prefix="test_battle_err_")
        ai_data_dir = os.path.join(temp_dir, "ai_data")
        train_data_dir = os.path.join(temp_dir, "train_data")
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

        try:
            session = BattleSession(
                ai1_name="random",
                ai2_name="unknown",
                num_games=1,
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
            )
            with pytest.raises(ModelLoadError, match="模型加载失败"):
                session.run()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_black_model_error_preserves_cause(self):
        """黑方模型失败时，ModelLoadError 应保留原始 ValueError 作为 __cause__"""
        temp_dir = tempfile.mkdtemp(prefix="test_battle_err_")
        ai_data_dir = os.path.join(temp_dir, "ai_data")
        train_data_dir = os.path.join(temp_dir, "train_data")
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

        try:
            session = BattleSession(
                ai1_name="unknown",
                ai2_name="random",
                num_games=1,
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
            )
            with pytest.raises(ModelLoadError) as exc_info:
                session.run()
            assert isinstance(exc_info.value.__cause__, ValueError)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_white_model_error_preserves_cause(self):
        """白方模型失败时，ModelLoadError 应保留原始 ValueError 作为 __cause__"""
        temp_dir = tempfile.mkdtemp(prefix="test_battle_err_")
        ai_data_dir = os.path.join(temp_dir, "ai_data")
        train_data_dir = os.path.join(temp_dir, "train_data")
        os.makedirs(ai_data_dir, exist_ok=True)
        os.makedirs(train_data_dir, exist_ok=True)

        try:
            session = BattleSession(
                ai1_name="random",
                ai2_name="unknown",
                num_games=1,
                ai_data_dir=ai_data_dir,
                train_data_dir=train_data_dir,
            )
            with pytest.raises(ModelLoadError) as exc_info:
                session.run()
            assert isinstance(exc_info.value.__cause__, ValueError)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
