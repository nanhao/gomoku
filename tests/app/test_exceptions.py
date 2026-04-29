"""测试 ModelLoadError 异常定义

验证 app/exceptions.py 中 ModelLoadError 的定义和行为：
- ModelLoadError 继承自 Exception
- 可携带错误信息
- 可作为 ValueError 的异常链使用（from e）
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.exceptions import ModelLoadError


class TestModelLoadError:
    """验证 ModelLoadError 异常类定义"""

    def test_is_exception_subclass(self):
        """ModelLoadError 应继承自 Exception"""
        assert issubclass(ModelLoadError, Exception)

    def test_can_carry_message(self):
        """ModelLoadError 应能携带错误信息"""
        err = ModelLoadError("模型加载失败：未知的模型: 'unknown'")
        assert "模型加载失败" in str(err)
        assert "unknown" in str(err)

    def test_can_chain_from_value_error(self):
        """ModelLoadError 应支持异常链（raise ... from e）"""
        original = ValueError("未知的模型: 'unknown'")
        try:
            raise ModelLoadError(f"模型加载失败：{original}") from original
        except ModelLoadError as err:
            assert err.__cause__ is original
            assert isinstance(err.__cause__, ValueError)

    def test_importable_from_app_package(self):
        """ModelLoadError 应能从 app 包直接导入"""
        from app import ModelLoadError as AppModelLoadError
        assert AppModelLoadError is ModelLoadError
