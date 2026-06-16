"""tests/ai/ 共享 fixtures。

被 tests/ai/ 及其子目录下所有测试文件自动发现（pytest conftest 机制）。
"""

import os
import shutil
import tempfile

import pytest

from ai.iterative_train import MODELS_DIR, PARENT_GEN


@pytest.fixture
def temp_project():
    """创建临时项目目录，包含 gen4/cnn_v4.pth 假文件和必要子目录。

    所有文件系统副作用隔离在临时目录中，测试结束自动清理。
    """
    tmp = tempfile.mkdtemp()
    # 创建模型目录和假权重文件
    gen4_dir = os.path.join(tmp, MODELS_DIR, f"gen{PARENT_GEN}")
    os.makedirs(gen4_dir, exist_ok=True)
    gen4_path = os.path.join(gen4_dir, "cnn_v4.pth")
    with open(gen4_path, 'w') as f:
        f.write("fake_model_weights")

    # 创建 models 目录
    models_dir = os.path.join(tmp, MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    yield tmp

    # 清理
    shutil.rmtree(tmp, ignore_errors=True)
