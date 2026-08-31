"""mcts_train_cnn_v11_offpolicy CLI 单元测试（设计 58 TC-CLI-01~08）

验证 CLI 入口：默认值生效与自动识别、参数边界校验、argparse 类型校验、
数据目录不存在、模型不存在等主流程与错误分支。

所有测试通过 mock 隔离 OffPolicyTrainLoop 与 get_highest_gen，
不触碰真实模型与样本文件；临时目录随测试结束自动清理。
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class OffPolicyCliTestBase(unittest.TestCase):
    """CLI 测试公共基类：模块导入失败时跳过（TDD RED 阶段）"""

    def setUp(self):
        try:
            from ai.scripts import mcts_train_cnn_v11_offpolicy as cli_mod
            self.cli = cli_mod
        except ImportError:
            self.cli = None
        if self.cli is None:
            self.skipTest("mcts_train_cnn_v11_offpolicy 尚未实现")


class TestOffPolicyCliDefaults(OffPolicyCliTestBase):
    """TC-CLI-01：默认值生效与自动识别最高代数"""

    def test_defaults_and_auto_detect(self):
        """仅传 --data-dir 时自动识别 cnn_v11 最高代数，其余参数用文档默认值"""
        with tempfile.TemporaryDirectory(prefix="cli_data_") as data_dir:
            test_args = ["mcts_train_cnn_v11_offpolicy.py", "--data-dir", data_dir]
            with patch.object(sys, "argv", test_args):
                with patch("ai.registry.get_highest_gen", return_value=5) as mock_hg:
                    with patch(
                        "ai.scripts.mcts_train_cnn_v11_offpolicy.OffPolicyTrainLoop"
                    ) as mock_loop:
                        mock_result = SimpleNamespace(
                            saved_gen=6,
                            saved_spec="cnn_v11.6",
                            requested_epochs=100,
                            total_epochs=100,
                            interrupted=False,
                            final_total_loss=0.5,
                            final_val_loss=0.6,
                        )
                        mock_loop.return_value.run.return_value = mock_result
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            self.cli.main()

        # 自动识别调用
        mock_hg.assert_called_once_with("cnn_v11")
        # 默认参数全部生效
        call_kwargs = mock_loop.call_args.kwargs
        self.assertEqual(call_kwargs["model_spec"], "cnn_v11.5")
        self.assertEqual(call_kwargs["data_dir"], data_dir)
        self.assertEqual(call_kwargs["batch_size"], 2048)
        self.assertEqual(call_kwargs["epochs"], 100)
        self.assertEqual(call_kwargs["val_split"], 0.1)
        self.assertEqual(call_kwargs["lr"], 0.001)
        self.assertEqual(call_kwargs["weight_decay"], 1e-4)
        self.assertEqual(call_kwargs["device"], "cuda")
        # 自动识别打印
        output = buf.getvalue()
        self.assertIn("自动识别", output)
        self.assertIn("cnn_v11.5", output)


class TestOffPolicyCliValidate(OffPolicyCliTestBase):
    """TC-CLI-02~06：参数边界与类型校验"""

    def test_batch_size_zero_rejected(self):
        """TC-CLI-02：--batch-size 0 打印"必须 > 0"并退出"""
        args = self.cli.build_parser().parse_args(["--batch-size", "0"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                self.cli.validate_args(args)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("必须 > 0", err.getvalue())

    def test_epochs_negative_rejected(self):
        """TC-CLI-03：--epochs -1 打印错误并退出"""
        args = self.cli.build_parser().parse_args(["--epochs", "-1"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                self.cli.validate_args(args)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("--epochs", err.getvalue())

    def test_validate_set_one_rejected(self):
        """TC-CLI-04：--validate-set 1.0 打印"必须 < 1"并退出"""
        args = self.cli.build_parser().parse_args(["--validate-set", "1.0"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                self.cli.validate_args(args)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("必须 < 1", err.getvalue())

    def test_validate_set_negative_rejected(self):
        """TC-CLI-05：--validate-set -0.1 打印"必须 >= 0"并退出"""
        args = self.cli.build_parser().parse_args(["--validate-set", "-0.1"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                self.cli.validate_args(args)
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("必须 >= 0", err.getvalue())

    def test_epochs_non_int_rejected(self):
        """TC-CLI-06：--epochs abc 由 argparse 拒绝（exit code 2）"""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                self.cli.build_parser().parse_args(["--epochs", "abc"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertNotEqual(err.getvalue().strip(), "")


class TestOffPolicyCliMainErrors(OffPolicyCliTestBase):
    """TC-CLI-07~08：main() 数据目录与模型错误路径"""

    def test_data_dir_not_exist_exits(self):
        """TC-CLI-07：--data-dir 不存在时打印错误并退出，不构造训练循环"""
        missing_dir = os.path.join(tempfile.gettempdir(), "no_such_dir_xyz_12345")
        test_args = [
            "mcts_train_cnn_v11_offpolicy.py",
            "--model", "cnn_v11.1",
            "--data-dir", missing_dir,
        ]
        with patch.object(sys, "argv", test_args):
            with patch(
                "ai.scripts.mcts_train_cnn_v11_offpolicy.OffPolicyTrainLoop"
            ) as mock_loop:
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    with self.assertRaises(SystemExit) as ctx:
                        self.cli.main()
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("目录不存在", err.getvalue())
        mock_loop.assert_not_called()

    def test_model_not_exist_exits_without_gen_dir(self):
        """TC-CLI-08：--model cnn_v11.99 模型不存在时退出，不产生 gen 目录"""
        with tempfile.TemporaryDirectory(prefix="cli_models_") as tmp_root:
            model_dir = os.path.join(tmp_root, "models")
            from ai.models.cnn_v11.model import CNNV11Model
            with patch.object(
                CNNV11Model, "DEFAULT_MODEL_PATH",
                os.path.join(model_dir, "cnn_v11.pth"),
            ):
                data_dir = os.path.join(tmp_root, "data")
                os.makedirs(data_dir, exist_ok=True)
                test_args = [
                    "mcts_train_cnn_v11_offpolicy.py",
                    "--model", "cnn_v11.99",
                    "--data-dir", data_dir,
                ]
                with patch.object(sys, "argv", test_args):
                    with patch(
                        "ai.scripts.mcts_train_cnn_v11_offpolicy.OffPolicyTrainLoop",
                        side_effect=ValueError("模型不存在: cnn_v11.99"),
                    ) as mock_loop:
                        err = io.StringIO()
                        with contextlib.redirect_stderr(err):
                            with self.assertRaises(SystemExit) as ctx:
                                self.cli.main()
                self.assertNotEqual(ctx.exception.code, 0)
                self.assertIn("模型不存在", err.getvalue())
                mock_loop.assert_called_once()
                # 不产生任何 gen 目录
                self.assertFalse(os.path.isdir(model_dir))


if __name__ == "__main__":
    unittest.main()
