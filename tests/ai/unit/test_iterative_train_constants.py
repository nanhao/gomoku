"""IterativeTrainer 参数化单元测试（意图 13）— 常量与硬编码检查

验证重构后模块级常量正确保留/删除，以及日志输出中无硬编码模型 spec。
对应设计文档 test_design.md UT-5、UT-6。
"""

import inspect

import pytest


class TestLogOutput:
    """UT-5：验证日志输出中的硬编码字符串已替换为变量。

    通过检查模块源码验证不存在硬编码 "cnn_v4.99" / "cnn_v4.100"
    的打印语句（除注释和文档字符串外）。
    """

    def test_no_hardcoded_model_specs_in_print(self):
        """验证模块源码中无硬编码模型 spec 字符串。

        排除注释行和文档字符串内容行后检查。
        """
        import ai.iterative_train as it_mod

        source = inspect.getsource(it_mod.IterativeTrainer)
        lines = source.split('\n')

        # 排除注释行和文档字符串行：追踪是否在 docstring 内部
        # 注意：此实现假设源代码中不会在非 docstring 代码行中出现 """，
        # 且多行 docstring 使用三引号格式开头和结尾。对 iterative_train.py 已验证有效。
        code_lines = []
        in_docstring = False
        for l in lines:
            stripped = l.strip()
            # 检测 docstring 边界（注意：单行 docstring 如 """...""" 会被 toggle 两次，
            # 但因 continue 跳过该行，实际效果正确）
            if stripped.startswith('"""') or stripped.endswith('"""'):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith('#'):
                continue
            code_lines.append(l)

        code_text = '\n'.join(code_lines)

        # 不应包含硬编码的 cnn_v4.99 / cnn_v4.100
        assert '"cnn_v4.99"' not in code_text
        assert "'cnn_v4.99'" not in code_text
        assert '"cnn_v4.100"' not in code_text
        assert "'cnn_v4.100'" not in code_text


class TestRetainedConstants:
    """验证保留的常量存在且值正确，删除的常量已移除。

    对应 main_design.md §2.2（删除的常量）、§2.3（保留的常量）。
    """

    def test_retained_constants_exist(self):
        """验证保留的常量（§2.3）未被删除且值正确。"""
        import ai.iterative_train as it_mod

        retained = {
            'SAMPLE_GAMES': 500,
            'BATTLE_GAMES': 100,
            'TRAIN_EPOCHS': 300,
            'TRAIN_BATCH_SIZE': 4096,
            'TEMP_MOVES': 2,
            'TEMPERATURE': 2,
        }
        for const_name, expected_value in retained.items():
            assert hasattr(it_mod, const_name), (
                f"保留常量 {const_name} 应存在但未找到"
            )
            assert getattr(it_mod, const_name) == expected_value, (
                f"常量 {const_name} 值应为 {expected_value}，"
                f"实际为 {getattr(it_mod, const_name)}"
            )

        # PYTHON_EXECUTABLE 存在即可（值与 sys.executable 相关）
        assert hasattr(it_mod, 'PYTHON_EXECUTABLE')

    def test_deleted_constants_removed(self):
        """验证删除的常量（§2.2）已被移除。"""
        import ai.iterative_train as it_mod

        deleted = [
            'BASE_MODEL_NAME', 'PARENT_GEN', 'CHILD_GEN',
            'MODELS_DIR', 'SAMPLE_OUTPUT_DIR',
        ]
        for const_name in deleted:
            assert not hasattr(it_mod, const_name), (
                f"删除常量 {const_name} 应不存在但仍存在"
            )
