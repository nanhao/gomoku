"""AI 测试共享辅助工具

提取自 test_fc_v1_trainer.py、test_cnn_trainer.py、
test_mini_batch_training.py 等文件的重复辅助方法，
消除跨文件代码重复。
"""

import json
import os
import subprocess
from unittest.mock import Mock


def create_sample_dir(temp_dir: str, name: str, n_items: int = 2,
                      use_softmax_output: bool = True) -> str:
    """创建一个包含 sample.json 的测试目录

    Args:
        temp_dir: 临时根目录
        name: 子目录名
        n_items: 样本数量
        use_softmax_output: True 使用 Softmax 概率分布作为 output，
                           False 使用原始 reward 值

    Returns:
        sample 目录的绝对路径
    """
    sample_dir = os.path.join(temp_dir, name)
    os.makedirs(sample_dir, exist_ok=True)

    sample_data = []
    for i in range(n_items):
        if use_softmax_output:
            # Softmax 概率分布
            output = [0.0] * 225
            output[0] = 0.7
            output[1] = 0.2
            output[2] = 0.1
            mask = [1, 1, 1] + [0] * 222
        else:
            # 原始 reward 值（兼容旧测试）
            output = [0] * 225
            idx = i % 225
            output[idx] = 1000000 - i * 10000
            mask = [1 if j == idx else 0 for j in range(225)]

        sample_data.append({
            "input": [float(i + 1)] * 226,
            "output": output,
            "mask": mask
        })

    sample_path = os.path.join(sample_dir, "sample.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f)

    return sample_dir


def create_output(x: int, y: int, reward: int) -> list:
    """创建 225 维输出向量，索引 i 对应位置 (i//15, i%15)

    Args:
        x: 行坐标 (0-14)
        y: 列坐标 (0-14)
        reward: 该位置的 reward 值

    Returns:
        225 维列表
    """
    output = [0] * 225
    idx = x * 15 + y
    output[idx] = reward
    return output


def create_minimal_sample_file(tmp_dir: str) -> str:
    """在临时目录创建最小样本文件，返回 JSON 文件路径

    用于 lr 参数测试等仅需最简 sample 数据的场景。
    创建单个样本：input 为 226 维全 1.0，output 为 225 维全 0.0，
    mask 为 225 维全 1。

    Args:
        tmp_dir: 临时目录路径

    Returns:
        JSON 文件的绝对路径
    """
    json_path = os.path.join(tmp_dir, "data.json")
    sample = [{"input": [1.0] * 226, "output": [0.0] * 225, "mask": [1] * 225}]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample, f)
    return json_path


def _make_completed_process(returncode=0, stdout="", stderr=""):
    """构造 mock subprocess.CompletedProcess 对象。

    用于在迭代训练测试中模拟子进程返回值，
    避免真实启动 subprocess。
    """
    result = Mock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result
