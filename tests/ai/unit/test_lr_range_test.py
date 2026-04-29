"""lr_range_test.py 测试

测试模型加载、数据加载、LR Range Test 执行流程和绘图保存功能。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

from ai.lr_range_test import (
    load_model,
    _get_input_converter,
    load_data,
    run_lr_range_test,
    plot_lr_loss_curve,
    parse_args,
    main,
)
from ai.registry import _parse_model_spec, _import_class, MODEL_REGISTRY
from tests.ai.helpers import create_sample_dir


class TestLoadModel(unittest.TestCase):
    """测试 load_model 函数"""

    @patch('ai.lr_range_test._import_class')
    @patch('ai.lr_range_test._parse_model_spec')
    def test_load_supported_model(self, mock_parse, mock_import):
        """应成功加载支持的模型并移动到目标设备"""
        mock_model_cls = MagicMock()
        mock_network = MagicMock()
        mock_model = MagicMock()
        mock_model.network = mock_network
        mock_model_cls.return_value = mock_model

        mock_parse.return_value = _parse_model_spec("cnn_v2")
        mock_import.return_value = mock_model_cls

        device = torch.device('cpu')
        model = load_model("cnn_v2", device)

        mock_model_cls.assert_called_once_with(fresh=True)
        mock_network.to.assert_called_once_with(device)
        self.assertEqual(model, mock_model)

    @patch('ai.lr_range_test._import_class')
    @patch('ai.lr_range_test._parse_model_spec')
    def test_load_unsupported_model_raises(self, mock_parse, mock_import):
        """不支持的模型名称应抛出 ValueError"""
        mock_parse.return_value = _parse_model_spec("unknown_model")

        with self.assertRaises(ValueError) as ctx:
            load_model("unknown_model", torch.device('cpu'))

        self.assertIn("不支持的模型", str(ctx.exception))

    @patch('ai.lr_range_test._import_class')
    @patch('ai.lr_range_test._parse_model_spec')
    def test_load_model_none_network(self, mock_parse, mock_import):
        """network 为 None 时不应调用 .to()"""
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.network = None
        mock_model_cls.return_value = mock_model

        mock_parse.return_value = _parse_model_spec("cnn_v2")
        mock_import.return_value = mock_model_cls

        device = torch.device('cpu')
        model = load_model("cnn_v2", device)
        self.assertEqual(model, mock_model)

    @patch('ai.lr_range_test._import_class')
    @patch('ai.lr_range_test._parse_model_spec')
    def test_load_multi_gen_model(self, mock_parse, mock_import):
        """多代模型 spec 应传递 gen 参数"""
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.network = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_parse.return_value = _parse_model_spec("cnn_v3.2")
        mock_import.return_value = mock_model_cls

        device = torch.device('cpu')
        model = load_model("cnn_v3.2", device)

        mock_model_cls.assert_called_once_with(fresh=True, gen=2)
        self.assertEqual(model, mock_model)


class TestGetInputConverter(unittest.TestCase):
    """测试 _get_input_converter 函数"""

    def test_returns_converter_for_neural_model(self):
        """有 _prepare_input 方法时返回 converter 函数"""
        mock_model = MagicMock()
        mock_model._prepare_input = MagicMock(
            side_effect=lambda v, d: torch.tensor([[v]])
        )

        device = torch.device('cpu')
        converter = _get_input_converter(mock_model, device)
        self.assertIsNotNone(converter)

        # 验证 converter 工作正常
        result = converter([1.0, 2.0])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)

    def test_returns_none_for_non_neural_model(self):
        """没有 _prepare_input 方法时返回 None（如 RandomModel）"""
        mock_model = MagicMock(spec=[])  # 空 spec，无任何属性

        device = torch.device('cpu')
        converter = _get_input_converter(mock_model, device)
        self.assertIsNone(converter)


class TestLoadData(unittest.TestCase):
    """测试 load_data 函数"""

    def setUp(self):
        self.sample_loader = MagicMock()
        self.sample_loader.load_merge_file = MagicMock(return_value=[{"input": []}])
        self.sample_loader.discover_and_load = MagicMock(return_value=[{"input": []}])
        self.sample_loader.prepare_tensors = MagicMock(return_value=(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
        ))

    def test_path_not_exists_raises(self):
        """路径不存在时应抛出 FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            load_data("/nonexistent/path", self.sample_loader)

    def test_load_single_file(self):
        """传入文件路径时应调用 load_merge_file"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'[{"input": []}]')
            tmp_path = f.name

        try:
            inputs, targets, masks = load_data(tmp_path, self.sample_loader)
            self.sample_loader.load_merge_file.assert_called_once_with(tmp_path)
            self.sample_loader.prepare_tensors.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_load_directory(self):
        """传入目录时应调用 discover_and_load"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_dir = create_sample_dir(tmp_dir, "test_data", n_items=2)

            loader = MagicMock()
            loader.discover_and_load = MagicMock(return_value=[{"input": []}])
            loader.prepare_tensors = MagicMock(return_value=(
                torch.tensor([1.0]),
                torch.tensor([2.0]),
                torch.tensor([3.0]),
            ))

            inputs, targets, masks = load_data(sample_dir, loader)
            loader.discover_and_load.assert_called_once_with(sample_dir)
            loader.prepare_tensors.assert_called_once()

    def test_empty_data_raises(self):
        """空数据应抛出 ValueError"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'[]')
            tmp_path = f.name

        try:
            loader = MagicMock()
            loader.load_merge_file = MagicMock(return_value=[])

            with self.assertRaises(ValueError) as ctx:
                load_data(tmp_path, loader)
            self.assertIn("数据为空", str(ctx.exception))
        finally:
            os.unlink(tmp_path)


class TestRunLRRangeTest(unittest.TestCase):
    """测试 run_lr_range_test 函数"""

    @patch('ai.lr_range_test._get_input_converter')
    @patch('ai.lr_range_test.load_data')
    @patch('ai.lr_range_test.load_model')
    @patch('ai.lr_range_test.get_device')
    def test_returns_correct_length(self, mock_get_device, mock_load_model, mock_load_data, mock_get_converter):
        """返回值长度应等于 rounds"""
        device = torch.device('cpu')
        mock_get_device.return_value = device

        # Mock model with a simple linear network
        param = torch.tensor([1.0], requires_grad=True)
        mock_network = MagicMock()
        mock_network.state_dict = MagicMock(return_value={"weight": param})
        mock_network.parameters = MagicMock(return_value=[param])
        mock_network.return_value = torch.tensor([[0.5] * 225], requires_grad=True)

        mock_model = MagicMock()
        mock_model.network = mock_network
        mock_load_model.return_value = mock_model
        mock_get_converter.return_value = None  # FC 模型无需转换

        # Mock data
        mock_load_data.return_value = (
            torch.tensor([[1.0] * 226]),
            torch.tensor([[0.5] * 225]),
            torch.tensor([[1.0] * 225]),
        )

        lrs, losses = run_lr_range_test(
            model_name="cnn_v2",
            data_path="fake/path",
            rounds=5,
            epochs=2,
            min_lr=1e-5,
            max_lr=1e-3,
            device=device
        )

        self.assertEqual(len(lrs), 5)
        self.assertEqual(len(losses), 5)
        self.assertEqual(mock_load_model.call_count, 1)

    @patch('ai.lr_range_test._get_input_converter')
    @patch('ai.lr_range_test.load_data')
    @patch('ai.lr_range_test.load_model')
    @patch('ai.lr_range_test.get_device')
    def test_lr_increases_linearly(self, mock_get_device, mock_load_model, mock_load_data, mock_get_converter):
        """lr 应线性递增"""
        device = torch.device('cpu')
        mock_get_device.return_value = device

        param = torch.tensor([1.0], requires_grad=True)
        mock_network = MagicMock()
        mock_network.state_dict = MagicMock(return_value={"param": param})
        mock_network.parameters = MagicMock(return_value=[param])
        mock_network.return_value = torch.tensor([[0.5] * 225], requires_grad=True)

        mock_model = MagicMock()
        mock_model.network = mock_network
        mock_load_model.return_value = mock_model
        mock_get_converter.return_value = None  # FC 模型无需转换

        mock_load_data.return_value = (
            torch.tensor([[1.0] * 226]),
            torch.tensor([[0.5] * 225]),
            torch.tensor([[1.0] * 225]),
        )

        lrs, _ = run_lr_range_test(
            model_name="cnn_v2",
            data_path="fake/path",
            rounds=5,
            epochs=1,
            min_lr=1e-5,
            max_lr=1e-3,
            device=device
        )

        self.assertAlmostEqual(lrs[0], 1e-5, places=10)
        self.assertAlmostEqual(lrs[4], 1e-3, places=10)
        # 检查递增趋势
        for i in range(1, len(lrs)):
            self.assertGreater(lrs[i], lrs[i - 1])

    @patch('ai.lr_range_test._get_input_converter')
    @patch('ai.lr_range_test.load_data')
    @patch('ai.lr_range_test.load_model')
    @patch('ai.lr_range_test.get_device')
    def test_model_reset_each_round(self, mock_get_device, mock_load_model, mock_load_data, mock_get_converter):
        """每个 round 开始前应重置模型状态"""
        device = torch.device('cpu')
        mock_get_device.return_value = device

        param = torch.tensor([1.0], requires_grad=True)
        mock_network = MagicMock()
        mock_network.state_dict = MagicMock(return_value={"param": param})
        mock_network.parameters = MagicMock(return_value=[param])
        mock_network.return_value = torch.tensor([[0.5] * 225], requires_grad=True)

        mock_model = MagicMock()
        mock_model.network = mock_network
        mock_load_model.return_value = mock_model
        mock_get_converter.return_value = None  # FC 模型无需转换

        mock_load_data.return_value = (
            torch.tensor([[1.0] * 226]),
            torch.tensor([[0.5] * 225]),
            torch.tensor([[1.0] * 225]),
        )

        run_lr_range_test(
            model_name="cnn_v2",
            data_path="fake/path",
            rounds=3,
            epochs=1,
            device=device
        )

        # load_state_dict 应在每个 round 开始时调用（rounds=3 次）
        self.assertEqual(mock_network.load_state_dict.call_count, 3)

    @patch('ai.lr_range_test._get_input_converter')
    @patch('ai.lr_range_test.load_data')
    @patch('ai.lr_range_test.load_model')
    @patch('ai.lr_range_test.get_device')
    def test_uses_provided_device(self, mock_get_device, mock_load_model, mock_load_data, mock_get_converter):
        """显式传入 device 时不应调用 get_device"""
        param = torch.tensor([1.0], requires_grad=True)
        mock_network = MagicMock()
        mock_network.state_dict = MagicMock(return_value={"param": param})
        mock_network.parameters = MagicMock(return_value=[param])
        mock_network.return_value = torch.tensor([[0.5] * 225], requires_grad=True)

        mock_model = MagicMock()
        mock_model.network = mock_network
        mock_load_model.return_value = mock_model
        mock_get_converter.return_value = None  # FC 模型无需转换

        mock_load_data.return_value = (
            torch.tensor([[1.0] * 226]),
            torch.tensor([[0.5] * 225]),
            torch.tensor([[1.0] * 225]),
        )

        device = torch.device('cpu')
        run_lr_range_test(
            model_name="cnn_v2",
            data_path="fake/path",
            rounds=1,
            epochs=1,
            device=device
        )

        mock_get_device.assert_not_called()


class TestPlotLRLossCurve(unittest.TestCase):
    """测试 plot_lr_loss_curve 函数"""

    @patch('ai.lr_range_test.os.makedirs')
    @patch('ai.lr_range_test.plt')
    @patch('ai.lr_range_test.matplotlib')
    def test_saves_figure(self, mock_matplotlib, mock_plt, mock_makedirs):
        """应调用 savefig 保存图片"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        lrs = [1e-5, 1e-4, 1e-3]
        losses = [0.5, 0.3, 0.1]
        output_path = "test_output/lr_curve.png"

        plot_lr_loss_curve(lrs, losses, output_path)

        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called_once_with(lrs, losses, linewidth=1)
        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches='tight')
        mock_plt.close.assert_called_once_with(mock_fig)
        mock_makedirs.assert_called_once_with("test_output", exist_ok=True)

    @patch('ai.lr_range_test.os.makedirs')
    @patch('ai.lr_range_test.plt')
    @patch('ai.lr_range_test.matplotlib')
    def test_no_makedirs_for_no_dir(self, mock_matplotlib, mock_plt, mock_makedirs):
        """输出路径无目录时不调用 makedirs"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_lr_loss_curve([1e-5], [0.5], "lr_curve.png")

        mock_makedirs.assert_not_called()


class TestParseArgs(unittest.TestCase):
    """测试 parse_args 函数"""

    def test_default_values(self):
        """默认参数值应正确"""
        with patch('sys.argv', ['lr_range_test.py', '--model', 'cnn_v2', '--data', 'data.json']):
            args = parse_args()
            self.assertEqual(args.model, 'cnn_v2')
            self.assertEqual(args.data, 'data.json')
            self.assertEqual(args.rounds, 50)
            self.assertEqual(args.epochs, 100)
            self.assertEqual(args.min_lr, 1e-7)
            self.assertEqual(args.max_lr, 1e-1)
            self.assertEqual(args.output, 'temp/lr_range_test.png')

    def test_custom_values(self):
        """自定义参数值应正确解析"""
        with patch('sys.argv', [
            'lr_range_test.py',
            '--model', 'cnn_v2',
            '--data', 'custom.json',
            '--rounds', '10',
            '--epochs', '20',
            '--min-lr', '1e-6',
            '--max-lr', '1e-2',
            '--output', 'out.png'
        ]):
            args = parse_args()
            self.assertEqual(args.rounds, 10)
            self.assertEqual(args.epochs, 20)
            self.assertEqual(args.min_lr, 1e-6)
            self.assertEqual(args.max_lr, 1e-2)
            self.assertEqual(args.output, 'out.png')

    def test_missing_required_args(self):
        """缺少必需参数应报错退出"""
        with patch('sys.argv', ['lr_range_test.py']):
            with self.assertRaises(SystemExit):
                parse_args()


class TestMain(unittest.TestCase):
    """测试 main 函数"""

    @patch('ai.lr_range_test.plot_lr_loss_curve')
    @patch('ai.lr_range_test.run_lr_range_test')
    @patch('ai.lr_range_test.parse_args')
    def test_main_runs_end_to_end(self, mock_parse_args, mock_run, mock_plot):
        """main 应调用 run_lr_range_test 和 plot_lr_loss_curve"""
        mock_args = MagicMock()
        mock_args.model = 'cnn_v2'
        mock_args.data = 'data.json'
        mock_args.rounds = 2
        mock_args.epochs = 3
        mock_args.min_lr = 1e-5
        mock_args.max_lr = 1e-3
        mock_args.output = 'out.png'
        mock_parse_args.return_value = mock_args

        mock_run.return_value = (
            [1e-5, 1e-4],
            [0.5, 0.4]
        )

        main()

        mock_run.assert_called_once_with(
            model_name='cnn_v2',
            data_path='data.json',
            rounds=2,
            epochs=3,
            min_lr=1e-5,
            max_lr=1e-3
        )
        mock_plot.assert_called_once_with(
            [1e-5, 1e-4],
            [0.5, 0.4],
            'out.png'
        )


if __name__ == '__main__':
    unittest.main()
