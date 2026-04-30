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
        """合法模型名称应正常创建引擎"""
        engine = create_engine("random")
        assert engine is not None
        assert engine.model is not None
