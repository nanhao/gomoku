"""tests/ai/ 共享 fixtures。

被 tests/ai/ 及其子目录下所有测试文件自动发现（pytest conftest 机制）。

意图 13 更新：不再从 ai.iterative_train 导入已删除的常量
（MODELS_DIR、PARENT_GEN 等），改为硬编码路径构造。
"""

import os
import shutil
import tempfile

import pytest

# 硬编码默认路径常量（意图 13：这些常量已从 ai/iterative_train.py 中移除）
_DEFAULT_MODEL_NAME = "cnn_v4"
_DEFAULT_PARENT_GEN = 99
_DEFAULT_MODELS_DIR = "ai/models/cnn_v4"


@pytest.fixture
def temp_project():
    """创建临时项目目录，包含 gen99/cnn_v4.pth 假文件和必要子目录。

    所有文件系统副作用隔离在临时目录中，测试结束自动清理。
    意图 13：不再从 ai.iterative_train 导入常量。
    """
    tmp = tempfile.mkdtemp()
    # 创建模型目录和假权重文件
    gen4_dir = os.path.join(
        tmp, _DEFAULT_MODELS_DIR, f"gen{_DEFAULT_PARENT_GEN}"
    )
    os.makedirs(gen4_dir, exist_ok=True)
    gen4_path = os.path.join(gen4_dir, f"{_DEFAULT_MODEL_NAME}.pth")
    with open(gen4_path, 'w') as f:
        f.write("fake_model_weights")

    # 创建 models 目录
    models_dir = os.path.join(tmp, _DEFAULT_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    yield tmp

    # 清理
    shutil.rmtree(tmp, ignore_errors=True)
