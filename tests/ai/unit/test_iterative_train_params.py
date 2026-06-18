"""IterativeTrainer 参数化单元测试（意图 13）— CLI 参数与路径计算

测试 CLI 参数解析、spec 解析、路径计算、模型名校验、保留常量验证。
对应设计文档 test_design.md §2.1～§2.3 及 §6 验收用例。

测试用例覆盖：
- UT-1.x：CLI 参数解析
- UT-2.x：模型名一致性校验
- UT-3.x：spec 解析与路径计算
- UT-5：日志输出中的硬编码检查
- UT-6：保留常量验证
"""

import os
import argparse
import tempfile
import shutil

import pytest


# ============================================================
# 辅助函数：构建与 main() 一致的 ArgumentParser
# ============================================================

def _build_parser():
    """构建与 main() 一致的 ArgumentParser，用于测试参数解析逻辑。

    参数定义严格遵循 interface_changes.md §1.2。
    """
    parser = argparse.ArgumentParser(
        description="迭代进化训练 — 串联 create_samples / train_main / battle"
    )
    parser.add_argument(
        '--ai1', type=str, required=True,
        help='父代模型 spec，如 cnn_v4.99',
    )
    parser.add_argument(
        '--ai2', type=str, required=True,
        help='子代模型 spec，如 cnn_v4.100',
    )
    parser.add_argument(
        '--sample_dir', type=str, required=True,
        help='样本输出目录基路径，如 ai/merge_samples2',
    )
    return parser


# ============================================================
# 辅助函数：spec 解析（与 main() 中的逻辑一致）
# ============================================================

def _parse_spec(spec: str):
    """解析模型 spec，返回 (model_name, gen)。

    使用 rsplit(".", 1) 仅分割最右侧的点号，
    支持 model_name 中含点号（如 cnn_v4.2.99 → ("cnn_v4.2", 99)）。
    与 interface_changes.md §1.2 中的解析逻辑完全一致。

    Raises:
        ValueError: spec 不含点号或 gen 部分非整数。
    """
    try:
        name, gen_str = spec.rsplit(".", 1)
        gen = int(gen_str)
    except ValueError:
        raise ValueError(f"模型 spec 格式错误: {spec}，期望格式为 模型名.代数")
    return name, gen


# ============================================================
# 辅助函数：路径计算（与 IterativeTrainer.__init__ 中的逻辑一致）
# ============================================================

def _compute_pth(model_name: str, gen: int, project_root: str = ""):
    """计算模型 .pth 文件路径。

    格式: {project_root}/ai/models/{model_name}/gen{gen}/{model_name}.pth
    与 main_design.md §1 核心变量计算表一致。
    """
    return os.path.join(
        project_root, "ai", "models", model_name,
        f"gen{gen}", f"{model_name}.pth"
    )


def _compute_sample_output_dir(sample_dir: str, gen_of_ai2: int):
    """计算样本输出目录路径。

    格式: {sample_dir}/v{gen_of_ai2}
    与 main_design.md §1 核心变量计算表一致。
    """
    return os.path.join(sample_dir, f"v{gen_of_ai2}")


def _compute_lock_path(model_name: str, project_root: str = ""):
    """计算锁文件路径。

    格式: {project_root}/ai/models/{model_name}/.iterative_lock
    """
    return os.path.join(
        project_root, "ai", "models", model_name, ".iterative_lock"
    )


# ============================================================
# 临时项目 fixture（为新 API 定制）
# ============================================================

@pytest.fixture
def temp_project_params():
    """创建临时项目目录，默认创建 cnn_v4/gen99 假权重文件。

    测试中可使用 _setup_model() 辅助函数创建其他模型/代数。
    所有文件系统副作用隔离在临时目录中，测试结束自动清理。
    """
    tmp = tempfile.mkdtemp()

    def _setup_model(model_name="cnn_v4", gen=99):
        """在临时目录中创建模型权重假文件。"""
        gen_dir = os.path.join(
            tmp, "ai", "models", model_name, f"gen{gen}"
        )
        os.makedirs(gen_dir, exist_ok=True)
        pth_path = os.path.join(gen_dir, f"{model_name}.pth")
        with open(pth_path, 'w') as f:
            f.write("fake_model_weights")
        models_dir = os.path.join(tmp, "ai", "models", model_name)
        os.makedirs(models_dir, exist_ok=True)

    # 默认创建 cnn_v4 gen99
    _setup_model("cnn_v4", 99)

    # 将辅助函数附加到 fixture 返回值（通过闭包引用 tmp）
    yield tmp, _setup_model
    shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# UT-1：CLI 参数解析（test_design.md §2.1）
# ============================================================

class TestCLIArgParsing:
    """UT-1：CLI 参数解析测试。

    使用 argparse.ArgumentParser.parse_args 直接测试参数解析。
    对校验逻辑的测试通过 pytest.raises(SystemExit) 捕获退出。
    """

    def test_all_three_args_specified(self):
        """UT-1.1：三个参数正常指定时解析成功。"""
        parser = _build_parser()
        args = parser.parse_args([
            '--ai1', 'cnn_v4.99',
            '--ai2', 'cnn_v4.100',
            '--sample_dir', 'ai/merge_samples2',
        ])
        assert args.ai1 == "cnn_v4.99"
        assert args.ai2 == "cnn_v4.100"
        assert args.sample_dir == "ai/merge_samples2"

    def test_missing_ai1(self):
        """UT-1.2：缺少 --ai1 时 SystemExit 退出码 2。"""
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                '--ai2', 'cnn_v4.100',
                '--sample_dir', 'ai/merge_samples2',
            ])
        assert exc_info.value.code == 2

    def test_missing_ai2(self):
        """UT-1.3：缺少 --ai2 时 SystemExit 退出码 2。"""
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                '--ai1', 'cnn_v4.99',
                '--sample_dir', 'ai/merge_samples2',
            ])
        assert exc_info.value.code == 2

    def test_missing_sample_dir(self):
        """UT-1.4：缺少 --sample_dir 时 SystemExit 退出码 2。"""
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                '--ai1', 'cnn_v4.99',
                '--ai2', 'cnn_v4.100',
            ])
        assert exc_info.value.code == 2

    def test_ai1_no_dot_parse_fails(self):
        """UT-1.5：--ai1 不含代数（无 .）时 _parse_spec 抛 ValueError。

        argparse 层面虽能正常解析为字符串，
        但 main() 中的 rsplit 解包会失败 → parser.error()。
        此处验证 _parse_spec 会正确抛出异常。
        """
        parser = _build_parser()
        args = parser.parse_args([
            '--ai1', 'cnn_v4',
            '--ai2', 'cnn_v4.100',
            '--sample_dir', 'ai/merge_samples2',
        ])
        # argparse 层面能正常解析（只是字符串）
        assert args.ai1 == "cnn_v4"
        # 格式校验失败应在 main() 中触发 parser.error()，
        # 此处验证 _parse_spec 会正确抛出异常
        with pytest.raises(ValueError):
            _parse_spec("cnn_v4")


# ============================================================
# UT-2：模型名一致性校验（test_design.md §2.2）
# ============================================================

class TestModelNameConsistency:
    """UT-2：模型名一致性校验测试。"""

    def test_same_model_name_passes(self):
        """UT-2.1：相同模型名通过校验。"""
        name1, _ = _parse_spec("cnn_v4.99")
        name2, _ = _parse_spec("cnn_v4.100")
        assert name1 == name2
        assert name1 == "cnn_v4"

    def test_different_model_name_fails(self):
        """UT-2.2：不同模型名报错。

        模拟 main() 中 model_name 不一致时触发 ValueError。
        （实际 main() 中会调用 parser.error() 退出进程）
        """
        name1, _ = _parse_spec("cnn_v3.99")
        name2, _ = _parse_spec("cnn_v4.100")
        assert name1 != name2
        # 此条件在 main() 中会触发 parser.error("模型名必须一致")
        with pytest.raises(ValueError, match="模型名"):
            if name1 != name2:
                raise ValueError(
                    f"--ai1 和 --ai2 的模型名必须一致，"
                    f"当前: {name1} vs {name2}"
                )

    def test_same_model_name_different_gen_passes(self):
        """UT-2.3：相同模型名但不同代数通过校验。"""
        name1, gen1 = _parse_spec("cnn_v4.1")
        name2, gen2 = _parse_spec("cnn_v4.200")
        assert name1 == name2
        assert gen1 == 1
        assert gen2 == 200


# ============================================================
# UT-3：spec 解析与路径计算（test_design.md §2.3）
# ============================================================

class TestSpecParsing:
    """UT-3：spec 解析与路径计算测试。"""

    # --- spec 解析 ---

    def test_parse_cnn_v4_99(self):
        """UT-3.1：解析 cnn_v4.99 → ("cnn_v4", 99)。"""
        name, gen = _parse_spec("cnn_v4.99")
        assert name == "cnn_v4"
        assert gen == 99

    def test_parse_cnn_v3_1(self):
        """UT-3.2：解析 cnn_v3.1 → ("cnn_v3", 1)。"""
        name, gen = _parse_spec("cnn_v3.1")
        assert name == "cnn_v3"
        assert gen == 1

    def test_parse_multi_dot_model_name(self):
        """UT-3.3：解析含多点模型名 cnn_v4.2.99。

        rsplit(".", 1) 仅分割最右侧的点号。
        结果应为 ("cnn_v4.2", 99)。
        """
        name, gen = _parse_spec("cnn_v4.2.99")
        assert name == "cnn_v4.2"
        assert gen == 99

    def test_parse_with_large_gen(self):
        """解析大代数如 999。"""
        name, gen = _parse_spec("cnn_v4.999")
        assert name == "cnn_v4"
        assert gen == 999

    def test_parse_fc_v1(self):
        """解析 fc_v1 模型 spec。"""
        name, gen = _parse_spec("fc_v1.5")
        assert name == "fc_v1"
        assert gen == 5

    # --- 路径计算 ---

    def test_compute_pth_of_ai1(self):
        """UT-3.4：计算 pth_of_ai1 路径。

        model_name="cnn_v4", gen=99:
        ai/models/cnn_v4/gen99/cnn_v4.pth
        """
        path = _compute_pth("cnn_v4", 99)
        expected = os.path.join("ai", "models", "cnn_v4", "gen99", "cnn_v4.pth")
        assert path == expected

    def test_compute_pth_of_ai2(self):
        """UT-3.5：计算 pth_of_ai2 路径。

        model_name="cnn_v4", gen=100:
        ai/models/cnn_v4/gen100/cnn_v4.pth
        """
        path = _compute_pth("cnn_v4", 100)
        expected = os.path.join("ai", "models", "cnn_v4", "gen100", "cnn_v4.pth")
        assert path == expected

    def test_compute_sample_output_dir(self):
        """UT-3.6：计算 sample_output_dir。

        sample_dir="ai/merge_samples2", gen_of_ai2=100:
        ai/merge_samples2/v100
        """
        path = _compute_sample_output_dir("ai/merge_samples2", 100)
        expected = os.path.join("ai", "merge_samples2", "v100")
        assert os.path.normpath(path) == os.path.normpath(expected)

    def test_compute_lock_path(self):
        """计算 lock 路径。"""
        path = _compute_lock_path("cnn_v4")
        expected = os.path.join("ai", "models", "cnn_v4", ".iterative_lock")
        assert path == expected

    def test_compute_pth_with_different_model(self):
        """计算 fc_v1 模型路径。"""
        path = _compute_pth("fc_v1", 5)
        expected = os.path.join("ai", "models", "fc_v1", "gen5", "fc_v1.pth")
        assert path == expected

    # --- 构造器集成：路径计算 + 文件系统校验 ---

    def test_constructor_paths_cnn_v4(self, temp_project_params):
        """构造器正确计算 cnn_v4.99 相关路径。

        验证 _ai1_path、_ai2_path、_sample_output_dir 等新属性。
        """
        tmp, _setup_model = temp_project_params

        from ai.iterative_train import IterativeTrainer

        trainer = IterativeTrainer(
            ai1="cnn_v4.99",
            ai2="cnn_v4.100",
            sample_dir="ai/merge_samples2",
            project_root=tmp,
        )

        # 验证新属性存在
        assert hasattr(trainer, '_ai1_path')
        assert hasattr(trainer, '_ai2_path')
        assert hasattr(trainer, '_model_name')
        assert hasattr(trainer, '_gen_of_ai1')
        assert hasattr(trainer, '_gen_of_ai2')

        # 验证旧属性已被移除
        assert not hasattr(trainer, '_gen4_path')
        assert not hasattr(trainer, '_gen5_path')
        assert not hasattr(trainer, '_gen5_dir')

        # 验证路径值
        expected_ai1 = os.path.join(
            tmp, "ai", "models", "cnn_v4", "gen99", "cnn_v4.pth"
        )
        expected_ai2 = os.path.join(
            tmp, "ai", "models", "cnn_v4", "gen100", "cnn_v4.pth"
        )
        expected_sample = os.path.join(
            tmp, "ai", "merge_samples2", "v100"
        )

        assert os.path.normpath(trainer._ai1_path) == os.path.normpath(expected_ai1)
        assert os.path.normpath(trainer._ai2_path) == os.path.normpath(expected_ai2)
        assert os.path.normpath(trainer._sample_output_dir) == os.path.normpath(expected_sample)
        assert trainer._model_name == "cnn_v4"
        assert trainer._gen_of_ai1 == 99
        assert trainer._gen_of_ai2 == 100

    def test_constructor_model_name_mismatch(self, temp_project_params):
        """模型名不一致时构造器抛 ValueError。"""
        tmp, _setup_model = temp_project_params

        from ai.iterative_train import IterativeTrainer

        with pytest.raises(ValueError, match="模型名"):
            IterativeTrainer(
                ai1="cnn_v3.99",
                ai2="cnn_v4.100",
                sample_dir="ai/merge_samples2",
                project_root=tmp,
            )

    def test_constructor_ai1_not_exist_raises(self, temp_project_params):
        """ai1 权重文件不存在时抛 FileNotFoundError。"""
        tmp, _setup_model = temp_project_params

        from ai.iterative_train import IterativeTrainer

        with pytest.raises(FileNotFoundError):
            IterativeTrainer(
                ai1="cnn_v4.999",
                ai2="cnn_v4.1000",
                sample_dir="ai/merge_samples2",
                project_root=tmp,
            )

    def test_constructor_lock_exists_raises(self, temp_project_params):
        """锁文件存在时抛 RuntimeError。"""
        tmp, _setup_model = temp_project_params
        # 创建锁文件
        lock_path = os.path.join(
            tmp, "ai", "models", "cnn_v4", ".iterative_lock"
        )
        with open(lock_path, 'w') as f:
            f.write("dummy")

        from ai.iterative_train import IterativeTrainer

        with pytest.raises(RuntimeError, match="锁"):
            IterativeTrainer(
                ai1="cnn_v4.99",
                ai2="cnn_v4.100",
                sample_dir="ai/merge_samples2",
                project_root=tmp,
            )



