"""IterativeTrainer 参数化单元测试（意图 13/22）— 常量与硬编码检查

验证重构后模块级常量正确保留/删除，以及日志输出中无硬编码模型 spec。
对应设计文档 test_design.md UT-5、UT-6。
意图 22：SAMPLE_GAMES 和 TRAIN_EPOCHS 已删除（下放为构造参数），新增 PROMOTION_THRESHOLD。
"""

import inspect

import pytest


class TestRetainedConstants:
    """验证保留的常量存在且值正确。

    对应 main_design.md §2.3（保留的常量）。
    """

    def test_retained_constants_exist(self):
        """验证保留的常量（§2.3）未被删除且值正确。

        意图 22：SAMPLE_GAMES/TRAIN_EPOCHS 已删除，新增 PROMOTION_THRESHOLD。
        """
        import ai.iterative_train as it_mod

        retained = {
            'PROMOTION_THRESHOLD': 55,
            'BATTLE_GAMES': 100,
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
        """意图 22：SAMPLE_GAMES 和 TRAIN_EPOCHS 已删除。"""
        import ai.iterative_train as it_mod

        for deleted in ['SAMPLE_GAMES', 'TRAIN_EPOCHS']:
            assert not hasattr(it_mod, deleted), (
                f"已删除常量 {deleted} 不应存在"
            )
