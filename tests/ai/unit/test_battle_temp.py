"""ai/battle.py 温度采样参数支持 — 单元测试

测试 run_battle() 签名变更、main() argparse 参数解析、
TempConfig 构造逻辑及错误处理。

用例编号与 test_design.md §2 对齐（UT-1.1 ~ UT-4.2）。
"""

import sys
import os
import unittest
import argparse
import io
import warnings
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径（支持从任何目录运行）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 需要在 battle 模块导入前处理 sys.path（与 ai/battle.py 一致）----
_ai_dir = os.path.join(_project_root, "ai")
for _p in list(sys.path):
    if os.path.normcase(os.path.abspath(_p)) == os.path.normcase(_ai_dir):
        sys.path.remove(_p)

from ai.move_sampler import TempConfig, SUPPORTED_TEMP_SIDES, TEMP_SIDE_BOTH
import ai.battle as battle_module


# ============================================================
# UT-1.x: run_battle() — temp_config 参数测试
# ============================================================

class TestRunBattleTempConfig(unittest.TestCase):
    """UT-1.1 ~ UT-1.4：验证 run_battle() 正确透传 temp_config"""

    def setUp(self):
        """为每个测试创建 mock 依赖链"""
        self.mock_get_model_patcher = patch('ai.battle.get_model')
        self.mock_ai_engine_patcher = patch('ai.battle.AIEngine')
        self.mock_data_recorder_patcher = patch('ai.battle.DataRecorder')
        self.mock_start_patcher = patch(
            'ai.battle.AIGameManager.start_ai_vs_ai_random')

        self.mock_get_model = self.mock_get_model_patcher.start()
        self.mock_ai_engine = self.mock_ai_engine_patcher.start()
        self.mock_data_recorder = self.mock_data_recorder_patcher.start()
        self.mock_start = self.mock_start_patcher.start()

        # 设置 mock 返回值链
        self.mock_model = MagicMock()
        self.mock_get_model.return_value = self.mock_model
        self.mock_engine = MagicMock()
        self.mock_ai_engine.return_value = self.mock_engine
        self.mock_recorder = MagicMock()
        self.mock_data_recorder.return_value = self.mock_recorder
        self.mock_start.return_value = []

    def tearDown(self):
        """停止所有 patcher"""
        self.mock_get_model_patcher.stop()
        self.mock_ai_engine_patcher.stop()
        self.mock_data_recorder_patcher.stop()
        self.mock_start_patcher.stop()

    def test_ut1_1_no_temp_config_passes_none(self):
        """UT-1.1: run_battle() 不传 temp_config → 透传 None"""
        battle_module.run_battle("random", "random", 1)

        self.mock_start.assert_called_once()
        _, kwargs = self.mock_start.call_args
        self.assertIsNone(kwargs.get('temp_config'),
                          "不传 temp_config 时应透传 None")

    def test_ut1_2_explicit_temp_config_passed_through(self):
        """UT-1.2: 显式传入 TempConfig → 透传该对象"""
        cfg = TempConfig(temperature=2.0, temp_moves=6, temp_side="black")

        battle_module.run_battle("random", "random", 1, temp_config=cfg)

        self.mock_start.assert_called_once()
        _, kwargs = self.mock_start.call_args
        self.assertIs(kwargs.get('temp_config'), cfg,
                      "应透传同一个 TempConfig 对象")

    def test_ut1_3_explicit_none_passed_through(self):
        """UT-1.3: 显式传 temp_config=None → 透传 None"""
        battle_module.run_battle("random", "random", 1, temp_config=None)

        self.mock_start.assert_called_once()
        _, kwargs = self.mock_start.call_args
        self.assertIsNone(kwargs.get('temp_config'),
                          "显式传 None 应透传 None")

    def test_ut1_4_temp_moves_zero_passed_through(self):
        """UT-1.4: temp_moves=0 的 TempConfig → 透传该对象（不转换）"""
        cfg = TempConfig(temperature=1.0, temp_moves=0, temp_side="both")

        battle_module.run_battle("random", "random", 1, temp_config=cfg)

        self.mock_start.assert_called_once()
        _, kwargs = self.mock_start.call_args
        self.assertIs(kwargs.get('temp_config'), cfg,
                      "temp_moves=0 的 TempConfig 应透传，不做语义转换")


# ============================================================
# UT-2.x: main() — argparse 参数解析测试
# ============================================================

class TestMainArgparse(unittest.TestCase):
    """UT-2.1 ~ UT-2.13：验证 argparse 参数解析正确性"""

    def setUp(self):
        """每个测试前构建新的解析器"""
        self.parser = battle_module.build_parser()

    def _parse(self, args_list):
        """辅助方法：解析参数列表，返回 Namespace"""
        return self.parser.parse_args(args_list)

    # ---- 默认值测试 ----

    def test_ut2_1_default_values(self):
        """UT-2.1: 不传任何温度参数 → 所有默认值正确"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1"])
        self.assertFalse(args.no_temp_sampling)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.temp_moves, 12)
        self.assertEqual(args.temp_side, TEMP_SIDE_BOTH)
        self.assertIsNone(args.temp_seed)

    # ---- 单个参数测试 ----

    def test_ut2_2_temperature_custom(self):
        """UT-2.2: --temperature 2.0 → 正确解析"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1", "--temperature", "2.0"])
        self.assertEqual(args.temperature, 2.0)

    def test_ut2_3_temp_moves_custom(self):
        """UT-2.3: --temp-moves 6 → 正确解析"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1", "--temp-moves", "6"])
        self.assertEqual(args.temp_moves, 6)

    def test_ut2_4_temp_side_black(self):
        """UT-2.4: --temp-side black → 正确解析"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1", "--temp-side", "black"])
        self.assertEqual(args.temp_side, "black")

    def test_ut2_5_temp_side_random(self):
        """UT-2.5: --temp-side random → 正确解析"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1", "--temp-side", "random"])
        self.assertEqual(args.temp_side, "random")

    def test_ut2_6_temp_seed_custom(self):
        """UT-2.6: --temp-seed 42 → 正确解析"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1", "--temp-seed", "42"])
        self.assertEqual(args.temp_seed, 42)

    def test_ut2_7_no_temp_sampling_flag(self):
        """UT-2.7: --no-temp-sampling → flag 为 True"""
        args = self._parse(["--ai1", "random", "--ai2", "random",
                            "--games", "1", "--no-temp-sampling"])
        self.assertTrue(args.no_temp_sampling)

    # ---- 组合参数测试 ----

    def test_ut2_8_combined_params(self):
        """UT-2.8: 组合所有温度参数 → 全部正确解析"""
        args = self._parse([
            "--ai1", "random", "--ai2", "random", "--games", "1",
            "--temperature", "0.5", "--temp-moves", "8",
            "--temp-side", "white", "--temp-seed", "99",
        ])
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.temp_moves, 8)
        self.assertEqual(args.temp_side, "white")
        self.assertEqual(args.temp_seed, 99)

    # ---- 错误校验测试 ----

    def test_ut2_9_temp_side_invalid(self):
        """UT-2.9: --temp-side abc → SystemExit(2)"""
        with self.assertRaises(SystemExit) as ctx:
            self._parse(["--ai1", "random", "--ai2", "random",
                         "--games", "1", "--temp-side", "abc"])
        self.assertEqual(ctx.exception.code, 2)

    def test_ut2_10_temp_seed_invalid(self):
        """UT-2.10: --temp-seed abc → SystemExit(2)"""
        with self.assertRaises(SystemExit) as ctx:
            self._parse(["--ai1", "random", "--ai2", "random",
                         "--games", "1", "--temp-seed", "abc"])
        self.assertEqual(ctx.exception.code, 2)

    def test_ut2_11_temperature_inf_rejected(self):
        """UT-2.11: --temperature inf → SystemExit(2)，finite_float 拒绝"""
        with self.assertRaises(SystemExit) as ctx:
            self._parse(["--ai1", "random", "--ai2", "random",
                         "--games", "1", "--temperature", "inf"])
        self.assertEqual(ctx.exception.code, 2)

    def test_ut2_12_temperature_nan_rejected(self):
        """UT-2.12: --temperature nan → SystemExit(2)，finite_float 拒绝"""
        with self.assertRaises(SystemExit) as ctx:
            self._parse(["--ai1", "random", "--ai2", "random",
                         "--games", "1", "--temperature", "nan"])
        self.assertEqual(ctx.exception.code, 2)

    # ---- help 输出测试 ----

    def test_ut2_13_help_contains_new_params(self):
        """UT-2.13: --help 输出包含所有 5 个新参数说明"""
        # 使用 argparse 的 print_help 捕获输出
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            try:
                self.parser.print_help()
            except SystemExit:
                pass
            output = mock_stdout.getvalue()

        expected_params = [
            "--no-temp-sampling",
            "--temperature",
            "--temp-moves",
            "--temp-side",
            "--temp-seed",
        ]
        for param in expected_params:
            self.assertIn(param, output,
                          f"--help 输出应包含参数: {param}")


# ============================================================
# UT-3.x: main() — TempConfig 构造逻辑测试
# ============================================================

class TestMainTempConfigConstruction(unittest.TestCase):
    """UT-3.1 ~ UT-3.5：验证从 argparse Namespace 构造 TempConfig"""

    def _build_args(self, **kwargs):
        """构造模拟的 argparse Namespace"""
        defaults = {
            "no_temp_sampling": False,
            "temperature": 1.0,
            "temp_moves": 12,
            "temp_side": TEMP_SIDE_BOTH,
            "temp_seed": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_ut3_1_default_temp_config(self):
        """UT-3.1: 默认构造 TempConfig（T=1.0, moves=12, side=both, seed=None）"""
        args = self._build_args()
        result = battle_module._build_temp_config(args)

        self.assertIsNotNone(result, "默认应构造 TempConfig，不应为 None")
        self.assertEqual(result.temperature, 1.0)
        self.assertEqual(result.temp_moves, 12)
        self.assertEqual(result.temp_side, TEMP_SIDE_BOTH)
        self.assertIsNone(result.seed)

    def test_ut3_2_no_temp_sampling_returns_none(self):
        """UT-3.2: --no-temp-sampling → 返回 None，不调用构造器"""
        args = self._build_args(no_temp_sampling=True)
        result = battle_module._build_temp_config(args)

        self.assertIsNone(result,
                          "--no-temp-sampling 时 temp_config 应为 None")

    def test_ut3_3_temperature_zero_raises(self):
        """UT-3.3: temperature=0 → TempConfig.__post_init__ 抛 ValueError"""
        args = self._build_args(temperature=0)

        with self.assertRaises(ValueError) as ctx:
            battle_module._build_temp_config(args)
        self.assertIn("temperature 必须 > 0", str(ctx.exception))

    def test_ut3_4_temperature_negative_raises(self):
        """UT-3.4: temperature=-1 → TempConfig.__post_init__ 抛 ValueError"""
        args = self._build_args(temperature=-1)

        with self.assertRaises(ValueError) as ctx:
            battle_module._build_temp_config(args)
        self.assertIn("temperature 必须 > 0", str(ctx.exception))

    def test_ut3_5_temp_moves_negative_raises(self):
        """UT-3.5: temp_moves=-5 → TempConfig.__post_init__ 抛 ValueError"""
        args = self._build_args(temp_moves=-5)

        with self.assertRaises(ValueError) as ctx:
            battle_module._build_temp_config(args)
        self.assertIn("temp_moves 必须 >= 0", str(ctx.exception))


# ============================================================
# UT-4.x: main() — 错误处理测试
# ============================================================

class TestMainErrorHandling(unittest.TestCase):
    """UT-4.1 ~ UT-4.2：验证 main() 错误处理路径"""

    def test_ut4_1_temp_config_value_error_caught(self):
        """UT-4.1: 非法温度参数 → ValueError 被捕获，退出码 1"""
        with patch('sys.argv', [
            'battle.py', '--ai1', 'random', '--ai2', 'random',
            '--games', '1', '--temperature', '-1',
        ]), patch('sys.stdout', new_callable=io.StringIO), \
           patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                battle_module.main()
            self.assertEqual(ctx.exception.code, 1,
                             "ValueError 应导致退出码 1")

    def test_ut4_2_model_load_error_still_caught(self):
        """UT-4.2: 模型加载失败 → ValueError 被捕获，退出码 1"""
        with patch('sys.argv', [
            'battle.py', '--ai1', 'nonexistent_ai', '--ai2', 'random',
            '--games', '1',
        ]), patch('sys.stdout', new_callable=io.StringIO), \
           patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                battle_module.main()
            self.assertEqual(ctx.exception.code, 1,
                             "模型加载失败应导致退出码 1")


# ============================================================
# 附加：warnings 提示测试（--no-temp-sampling + 其他参数）
# ============================================================

class TestNoTempSamplingWarning(unittest.TestCase):
    """验证 --no-temp-sampling 时对其他温度参数的警告行为"""

    def test_warning_when_no_temp_sampling_with_custom_temp(self):
        """--no-temp-sampling + --temperature 2.0 → 发出 warning"""
        args = argparse.Namespace(
            no_temp_sampling=True, temperature=2.0,
            temp_moves=12, temp_side=TEMP_SIDE_BOTH, temp_seed=None,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = battle_module._build_temp_config(args)

        self.assertIsNone(result)
        self.assertTrue(any("--no-temp-sampling" in str(warning.message)
                           for warning in w),
                        "应发出 --no-temp-sampling 相关警告")

    def test_no_warning_when_no_temp_sampling_with_defaults(self):
        """--no-temp-sampling 但其他参数全是默认值 → 不发出 warning"""
        args = argparse.Namespace(
            no_temp_sampling=True, temperature=1.0,
            temp_moves=12, temp_side=TEMP_SIDE_BOTH, temp_seed=None,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = battle_module._build_temp_config(args)

        self.assertIsNone(result)
        # 应无 warning（所有温度参数都是默认值）
        temp_warnings = [x for x in w if "--no-temp-sampling" in str(x.message)]
        self.assertEqual(len(temp_warnings), 0,
                         "其他参数均为默认值时不应发出警告")

    def test_warning_when_no_temp_sampling_with_custom_seed(self):
        """--no-temp-sampling + --temp-seed 42 → 发出 warning"""
        args = argparse.Namespace(
            no_temp_sampling=True, temperature=1.0,
            temp_moves=12, temp_side=TEMP_SIDE_BOTH, temp_seed=42,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = battle_module._build_temp_config(args)

        self.assertIsNone(result)
        self.assertTrue(any("--no-temp-sampling" in str(warning.message)
                           for warning in w),
                        "应发出 --no-temp-sampling 相关警告")


if __name__ == "__main__":
    unittest.main()
