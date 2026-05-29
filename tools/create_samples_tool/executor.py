"""executor 模块 — 子进程同步执行与 round_dir 发现

负责：
1. build_cli_command: 将前端参数构造为 CLI 命令字符串
2. execute: 同步阻塞执行子进程，收集输出，发现 round_dir
3. discover_round_dir: 通过差集 S_after - S_before 发现本轮数据目录

零耦合：不 import 任何 ai/ 包模块，仅使用标准库 subprocess。
"""

import os
import subprocess
from typing import Dict, Optional


# ------------------------------------------------------------------
# CLI 命令构造
# ------------------------------------------------------------------

def build_cli_command(params: Dict) -> str:
    """将前端参数构造为 CLI 命令字符串

    规则：
    - 必填参数（model、output）始终追加
    - 有默认值的参数追加（mode、games、temperature 等）
    - 布尔开关仅 true 时追加 flag（--no-augment、--no-temp-sampling）
    - temp_seed 仅非 null 时追加

    Args:
        params: 前端传入的参数字典

    Returns:
        完整 CLI 命令字符串
    """
    parts = ["python", "ai/create_samples.py"]

    # 必填参数
    parts.append(f"--model {params['model']}")
    parts.append(f"--output {params['output']}")

    # 有默认值的参数
    if "mode" in params:
        parts.append(f"--mode {params['mode']}")
    if "games" in params:
        parts.append(f"--games {params['games']}")

    # 布尔开关：仅 true 时追加
    if params.get("no_augment"):
        parts.append("--no-augment")
    if params.get("no_temp_sampling"):
        parts.append("--no-temp-sampling")

    # 温度参数
    if "temperature" in params:
        parts.append(f"--temperature {params['temperature']}")
    if "temp_moves" in params:
        parts.append(f"--temp-moves {params['temp_moves']}")
    if "temp_side" in params:
        parts.append(f"--temp-side {params['temp_side']}")

    # 可选数值：仅非 null 时追加
    if params.get("temp_seed") is not None:
        parts.append(f"--temp-seed {params['temp_seed']}")

    return " ".join(parts)


# ------------------------------------------------------------------
# 子进程执行
# ------------------------------------------------------------------

def execute(params: Dict, project_root: str = None,
            ai_data_dir: str = None,
            command_override: str = None) -> Dict:
    """同步阻塞执行子进程，收集输出，发现 round_dir

    Args:
        params: 前端传入的参数字典
        project_root: 项目根目录（cwd），默认当前工作目录
        ai_data_dir: ai_data 目录路径，默认 'ai/ai_data'
        command_override: 覆盖命令字符串（测试用，替代 build_cli_command 产出）

    Returns:
        ExecuteResult 字典，含 status、command、output、returncode、
        round_dir、output_path、games（成功时）
    """
    if project_root is None:
        project_root = os.getcwd()
    if ai_data_dir is None:
        ai_data_dir = os.path.join(project_root, "ai", "ai_data")

    # 记录执行前的目录集合
    s_before = _scan_dir_names(ai_data_dir)

    # 构造命令
    command = command_override or build_cli_command(params)

    # 启动子进程
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=project_root,
        shell=True,
    )

    # 阻塞等待并收集输出
    output_bytes, _ = process.communicate()
    output_text = output_bytes.decode("utf-8", errors="replace")

    if process.returncode == 0:
        # 成功路径：发现 round_dir
        round_dir_rel = discover_round_dir(ai_data_dir, s_before)

        # 扫描 round_dir 获取对局列表
        games = []
        if round_dir_rel is not None:
            round_dir_abs = os.path.join(project_root, round_dir_rel)
            try:
                from tools.create_samples_tool.data_parser import scan_round_dir
                games = scan_round_dir(round_dir_abs)
            except FileNotFoundError:
                pass

        return {
            "status": "completed",
            "command": command,
            "output": output_text,
            "returncode": 0,
            "round_dir": round_dir_rel,
            "output_path": params.get("output"),
            "games": games,
        }
    else:
        # 失败路径
        return {
            "status": "error",
            "command": command,
            "output": output_text,
            "returncode": process.returncode,
            "round_dir": None,
            "output_path": None,
            "games": None,
        }


# ------------------------------------------------------------------
# 差集发现 round_dir
# ------------------------------------------------------------------

def _scan_dir_names(ai_data_dir: str) -> set:
    """扫描 ai_data_dir 下的目录名集合

    Args:
        ai_data_dir: ai_data 目录路径

    Returns:
        目录名集合（仅子目录名，不含路径前缀）
    """
    if not os.path.exists(ai_data_dir):
        return set()
    return {
        entry for entry in os.listdir(ai_data_dir)
        if os.path.isdir(os.path.join(ai_data_dir, entry))
    }


def discover_round_dir(ai_data_dir: str, s_before: set) -> Optional[str]:
    """通过差集 S_after - S_before 发现本轮数据目录

    Args:
        ai_data_dir: ai_data 目录路径
        s_before: 执行前的子目录名集合

    Returns:
        发现的 round_dir 相对路径（相对于 project_root），
        差集为空时返回 None。
    """
    s_after = _scan_dir_names(ai_data_dir)
    diff = s_after - s_before

    if not diff:
        return None

    # 取排序后最后一个（最新创建的目录）
    newest = sorted(diff)[-1]
    # 返回相对于 project_root 的路径
    # ai_data_dir 是 project_root/ai/ai_data
    # round_dir 相对路径 = ai/ai_data/{newest}
    # 通过 ai_data_dir 路径计算相对部分
    parent = os.path.basename(ai_data_dir)  # "ai_data"
    grandparent = os.path.basename(os.path.dirname(ai_data_dir))  # "ai"
    return os.path.join(grandparent, parent, newest)