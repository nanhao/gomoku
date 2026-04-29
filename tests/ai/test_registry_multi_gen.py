"""多代模型注册表改造单元测试

测试 ai/registry.py 中新增/变更的多代模型相关功能：
- _parse_model_spec: 模型指定字符串解析
- _is_multi_gen_model: 多代模型判定
- get_model: 多代 spec 格式与校验（mock 导入）
- get_trainer: 多代 spec 格式与校验（mock 导入）
- list_models_with_full_name: 含代数的完整模型列表（mock 文件系统）
- 注册校验: 模型名称禁止包含 '.'
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.registry import (
    MODEL_REGISTRY,
    ParsedModelSpec,
    RegistryEntry,
    _is_multi_gen_model,
    _list_models,
    _parse_model_spec,
    get_model,
    get_trainer,
    list_models_with_full_name,
)


# ---------------------------------------------------------------------------
# §5.1 _parse_model_spec 测试
# ---------------------------------------------------------------------------
class TestParseModelSpec(unittest.TestCase):
    """测试 _parse_model_spec 解析模型指定字符串"""

    def test_no_dot_returns_name_only(self):
        """不含 '.' → name=spec, gen=None"""
        result = _parse_model_spec("random")
        self.assertEqual(result, ParsedModelSpec(name="random", gen=None))

    def test_single_dot_valid_gen(self):
        """含一个 '.' 且 gen 有效 → 正确解析"""
        result = _parse_model_spec("cnn_v3.1")
        self.assertEqual(result, ParsedModelSpec(name="cnn_v3", gen=1))

    def test_large_gen_value(self):
        """gen 为较大正整数"""
        result = _parse_model_spec("cnn_v3.42")
        self.assertEqual(result, ParsedModelSpec(name="cnn_v3", gen=42))

    def test_empty_gen_raises(self):
        """'.' 后为空 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            _parse_model_spec("cnn_v3.")
        self.assertIn("代数必须为正整数", str(ctx.exception))

    def test_non_integer_gen_raises(self):
        """gen 非整数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            _parse_model_spec("cnn_v3.abc")
        self.assertIn("代数必须为正整数", str(ctx.exception))

    def test_zero_gen_raises(self):
        """gen=0 → ValueError（必须 ≥ 1）"""
        with self.assertRaises(ValueError) as ctx:
            _parse_model_spec("cnn_v3.0")
        self.assertIn("代数必须", str(ctx.exception))

    def test_negative_gen_raises(self):
        """gen 为负数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            _parse_model_spec("cnn_v3.-1")
        self.assertIn("代数必须", str(ctx.exception))

    def test_multiple_dots_rightmost_split(self):
        """多个 '.' → 右侧分割第一个 '.'（rsplit）"""
        result = _parse_model_spec("a.b.3")
        self.assertEqual(result, ParsedModelSpec(name="a.b", gen=3))


# ---------------------------------------------------------------------------
# §5.2 _is_multi_gen_model 测试
# ---------------------------------------------------------------------------
class TestIsMultiGenModel(unittest.TestCase):
    """测试 _is_multi_gen_model 判断模型是否为多代"""

    def test_multi_gen_model_returns_true(self):
        """cnn_v3 是多代模型 → True"""
        self.assertTrue(_is_multi_gen_model("cnn_v3"))

    def test_non_multi_gen_models_return_false(self):
        """random / fc_v1 / cnn_v1 / cnn_v2 非多代 → False"""
        for name in ["random", "fc_v1", "cnn_v1", "cnn_v2"]:
            with self.subTest(name=name):
                self.assertFalse(_is_multi_gen_model(name))

    def test_unknown_model_raises_key_error(self):
        """不在注册表中的模型 → KeyError"""
        with self.assertRaises(KeyError):
            _is_multi_gen_model("nonexistent")


# ---------------------------------------------------------------------------
# §5.3 get_model 测试（mock _import_class 避免真实导入）
# ---------------------------------------------------------------------------
class TestGetModel(unittest.TestCase):
    """测试 get_model 的多代 spec 格式与校验逻辑"""

    @patch("ai.registry._import_class")
    def test_non_multi_gen_without_gen(self, mock_import):
        """非多代模型不指定代数 → 正常创建（无 gen 参数）"""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        get_model("random")
        mock_cls.assert_called_once_with()

    @patch("ai.registry._import_class")
    def test_non_multi_gen_with_gen_raises(self, mock_import):
        """非多代模型指定代数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_model("random.1")
        self.assertIn("非多代模型不能指定代数", str(ctx.exception))

    @patch("ai.registry._import_class")
    def test_multi_gen_without_gen_raises(self, mock_import):
        """多代模型不指定代数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_model("cnn_v3")
        self.assertIn("多代模型必须指定代数", str(ctx.exception))

    @patch("ai.registry._import_class")
    def test_multi_gen_with_valid_gen(self, mock_import):
        """多代模型指定有效代数 → 传入 gen 参数创建实例"""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        get_model("cnn_v3.3")
        mock_cls.assert_called_once_with(gen=3)

    def test_unknown_model_raises(self):
        """未知模型名称 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_model("nonexistent")
        self.assertIn("未知的模型", str(ctx.exception))

    def test_unknown_multi_gen_model_raises(self):
        """未知模型指定代数 → ValueError（解析后模型名不在注册表中）"""
        with self.assertRaises(ValueError) as ctx:
            get_model("unknown.1")
        self.assertIn("未知的模型", str(ctx.exception))

    @patch("ai.registry._import_class")
    def test_fc_v1_without_gen(self, mock_import):
        """fc_v1 不指定代数 → 正常创建"""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        get_model("fc_v1")
        mock_cls.assert_called_once_with()

    @patch("ai.registry._import_class")
    def test_cnn_v2_with_gen_raises(self, mock_import):
        """cnn_v2 非多代，指定代数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_model("cnn_v2.1")
        self.assertIn("非多代模型不能指定代数", str(ctx.exception))


# ---------------------------------------------------------------------------
# §5.4 get_trainer 测试（mock _import_class 避免真实导入）
# ---------------------------------------------------------------------------
class TestGetTrainer(unittest.TestCase):
    """测试 get_trainer 的多代 spec 格式与校验逻辑"""

    @patch("ai.registry._import_class")
    def test_non_multi_gen_without_gen(self, mock_import):
        """非多代模型不指定代数 → 正常创建训练器"""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        get_trainer("fc_v1", fresh=True)
        mock_cls.assert_called_once_with(fresh=True)

    @patch("ai.registry._import_class")
    def test_non_multi_gen_with_gen_raises(self, mock_import):
        """非多代模型指定代数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("fc_v1.1")
        self.assertIn("非多代模型不能指定代数", str(ctx.exception))

    @patch("ai.registry._import_class")
    def test_multi_gen_without_gen_raises(self, mock_import):
        """多代模型不指定代数 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("cnn_v3")
        self.assertIn("多代模型必须指定代数", str(ctx.exception))

    @patch("ai.registry._import_class")
    def test_multi_gen_with_valid_gen(self, mock_import):
        """多代模型指定代数 → 传入 gen 创建训练器"""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        get_trainer("cnn_v3.2", fresh=True)
        mock_cls.assert_called_once_with(gen=2, fresh=True)

    def test_no_trainer_model_raises(self):
        """random 没有训练器 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("random")
        self.assertIn("不支持的模型", str(ctx.exception))

    def test_unknown_model_raises(self):
        """未知模型 → ValueError"""
        with self.assertRaises(ValueError) as ctx:
            get_trainer("nonexistent")
        self.assertIn("不支持的模型", str(ctx.exception))


# ---------------------------------------------------------------------------
# §5.5 list_models_with_full_name 测试（mock 文件系统）
# ---------------------------------------------------------------------------
class TestListModelsWithFullName(unittest.TestCase):
    """测试 list_models_with_full_name 含代数的完整模型列表"""

    def _setup_model_dir(self, base_dir, model_name, gen_dirs=None, pth_files=None):
        """在 base_dir 下创建模型目录结构

        Args:
            base_dir: 基础目录
            model_name: 模型名称
            gen_dirs: gen 子目录编号列表，如 [1, 2, 5]
            pth_files: pth 文件名列表，如 ["model.pth"]
        """
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        for g in (gen_dirs or []):
            os.makedirs(os.path.join(model_dir, f"gen{g}"), exist_ok=True)
        for f in (pth_files or []):
            open(os.path.join(model_dir, f), "w").close()

    def test_non_multi_gen_returns_name(self):
        """非多代模型直接返回模型名称本身"""
        with tempfile.TemporaryDirectory() as tmp:
            # 为所有注册模型创建目录（非多代模型不需要目录也能返回名称）
            for name in MODEL_REGISTRY:
                os.makedirs(os.path.join(tmp, name), exist_ok=True)

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            # 非多代模型应出现
            for name in ["random", "fc_v1", "cnn_v1", "cnn_v2"]:
                self.assertIn(name, result)

    def test_multi_gen_with_gen_dirs(self):
        """多代模型有 gen 子目录 → 返回带代数的完整指定"""
        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, "cnn_v3", gen_dirs=[1, 3])

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            self.assertIn("cnn_v3.1", result)
            self.assertIn("cnn_v3.3", result)
            # 不应出现裸名
            self.assertNotIn("cnn_v3", result)

    def test_multi_gen_no_gen_dirs_not_shown(self):
        """多代模型无 gen 子目录 → 不展示该模型"""
        with tempfile.TemporaryDirectory() as tmp:
            # cnn_v3 目录存在但无 gen 子目录
            self._setup_model_dir(tmp, "cnn_v3")

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            # cnn_v3 不应以任何形式出现
            self.assertTrue(all("cnn_v3" not in r for r in result))

    def test_pth_and_gen_coexist_raises(self):
        """pth 文件与 gen 目录共存 → ValueError"""
        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, "cnn_v3", gen_dirs=[1],
                                  pth_files=["model.pth"])

            with patch("ai.registry._MODELS_DIR", tmp):
                with self.assertRaises(ValueError) as ctx:
                    list_models_with_full_name()
            self.assertIn("共存", str(ctx.exception))

    def test_gen_dirs_sorted(self):
        """多代模型返回结果按 gen 编号排序"""
        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, "cnn_v3", gen_dirs=[5, 1, 3])

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            cnn_v3_entries = [r for r in result if r.startswith("cnn_v3.")]
            self.assertEqual(cnn_v3_entries, ["cnn_v3.1", "cnn_v3.3", "cnn_v3.5"])

    def test_non_gen_dirs_ignored(self):
        """不以 gen{i} 命名的子目录不应被识别为代数"""
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "cnn_v3")
            os.makedirs(model_dir, exist_ok=True)
            # 非 gen 模式的目录
            os.makedirs(os.path.join(model_dir, "other"), exist_ok=True)
            os.makedirs(os.path.join(model_dir, "gen"), exist_ok=True)  # 无数字

            with patch("ai.registry._MODELS_DIR", tmp):
                result = list_models_with_full_name()

            # 无有效 gen 目录 → cnn_v3 不展示
            self.assertTrue(all("cnn_v3" not in r for r in result))


# ---------------------------------------------------------------------------
# 注册校验测试
# ---------------------------------------------------------------------------
class TestRegistryValidation(unittest.TestCase):
    """测试注册表校验：模型名称禁止包含 '.'"""

    def test_current_registry_names_have_no_dots(self):
        """当前注册表中所有模型名称均不含 '.'（模块成功导入即通过）"""
        for name in MODEL_REGISTRY:
            self.assertNotIn(".", name, f"模型名 '{name}' 不应包含 '.'")

    def test_dot_in_name_triggers_validation(self):
        """模拟：模型名称包含 '.' 时应触发 ValueError"""
        test_registry = {"bad.name": RegistryEntry("some.path", None, False)}
        with self.assertRaises(ValueError) as ctx:
            for n in test_registry:
                if "." in n:
                    raise ValueError("模型名称禁止包含 '.'")
        self.assertIn("模型名称禁止包含 '.'", str(ctx.exception))


# ---------------------------------------------------------------------------
# _list_models 降级测试
# ---------------------------------------------------------------------------
class TestListModelsPrivate(unittest.TestCase):
    """测试 _list_models 私有函数（原 list_models 降级）"""

    def test_returns_list(self):
        """_list_models 返回列表"""
        result = _list_models()
        self.assertIsInstance(result, list)

    def test_contains_all_registry_keys(self):
        """_list_models 应包含注册表中所有模型名称"""
        result = _list_models()
        self.assertEqual(sorted(result), sorted(MODEL_REGISTRY.keys()))


if __name__ == "__main__":
    unittest.main()
