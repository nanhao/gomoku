"""测试 GameSession 模型加载异常转换

验证 GameSession.__init__ 在 create_engine 抛出 ValueError 时，
将其转换为 ModelLoadError 并向上抛出，包含详细的错误信息。

测试场景（对应设计文档 §9.5）：
- GameSession 模型不存在 → raise ModelLoadError
- GameSession 多代未指定代数 → raise ModelLoadError
- GameSession 正常加载 → 无异常
"""

import sys
import os
import tempfile
import shutil
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game import GomokuController
from game.save import GomokuSave
from app import GameSession
from app.exceptions import ModelLoadError


def _make_controller():
    """创建使用临时目录的 controller"""
    temp_dir = tempfile.mkdtemp(prefix="test_session_err_")
    save_dir = os.path.join(temp_dir, "saves")
    c = GomokuController()
    c.save = GomokuSave(save_dir)
    c._temp_dir = temp_dir  # 保存引用以便清理
    return c


def _cleanup_controller(controller):
    """清理 controller 的临时目录"""
    if hasattr(controller, '_temp_dir') and os.path.exists(controller._temp_dir):
        shutil.rmtree(controller._temp_dir, ignore_errors=True)


class TestGameSessionModelLoadError:
    """GameSession 模型加载异常转换测试"""

    def test_unknown_model_raises_model_load_error(self):
        """ai_name='unknown' 时应抛出 ModelLoadError"""
        controller = _make_controller()
        try:
            with pytest.raises(ModelLoadError, match="模型加载失败"):
                GameSession(mode="pve", controller=controller, ai_name="unknown", ai_player=2)
        finally:
            _cleanup_controller(controller)

    def test_unknown_model_preserves_cause(self):
        """ModelLoadError 应保留原始 ValueError 作为 __cause__"""
        controller = _make_controller()
        try:
            with pytest.raises(ModelLoadError) as exc_info:
                GameSession(mode="pve", controller=controller, ai_name="unknown", ai_player=2)
            assert isinstance(exc_info.value.__cause__, ValueError)
        finally:
            _cleanup_controller(controller)

    def test_multi_gen_without_gen_raises_model_load_error(self):
        """ai_name='cnn_v3'（多代模型未指定代数）应抛出 ModelLoadError"""
        controller = _make_controller()
        try:
            with pytest.raises(ModelLoadError, match="模型加载失败"):
                GameSession(mode="pve", controller=controller, ai_name="cnn_v3", ai_player=2)
        finally:
            _cleanup_controller(controller)

    def test_multi_gen_error_message_contains_hint(self):
        """多代模型未指定代数时，错误信息应包含提示"""
        controller = _make_controller()
        try:
            with pytest.raises(ModelLoadError, match="多代模型必须指定代数"):
                GameSession(mode="pve", controller=controller, ai_name="cnn_v3", ai_player=2)
        finally:
            _cleanup_controller(controller)

    def test_valid_model_no_error(self):
        """ai_name='random' 时应正常创建 session，无异常"""
        controller = _make_controller()
        try:
            session = GameSession(mode="pve", controller=controller, ai_name="random", ai_player=2)
            assert session is not None
            assert session.ai_player == 2
        finally:
            _cleanup_controller(controller)

    def test_pvp_mode_no_model_load(self):
        """pvp 模式不涉及 AI 模型加载，不会抛出 ModelLoadError"""
        controller = _make_controller()
        try:
            session = GameSession(mode="pvp", controller=controller)
            assert session is not None
            assert session.mode == "pvp"
        finally:
            _cleanup_controller(controller)
