"""executor 模块 — 子进程同步执行与 round_dir 发现

负责：
1. build_cli_command: 将前端参数构造为 CLI 命令参数列表
2. execute: 同步阻塞执行子进程，收集输出，发现 round_dir
3. discover_round_dir: 通过差集 S_after - S_before 发现本轮数据目录
"""

import os
import subprocess
from typing import Dict, List, Optional

from tools.create_samples_tool.data_parser import scan_round_dir


# ------------------------------------------------------------------
# CLI 命令构造
# ------------------------------------------------------------------

def build_cli_command(params: Dict) -> List[str]:
    """将前端参数构造为 CLI 命令参数列表（shell=False 安全调用）

    规则：
    - 必填参数（model、output）始终追加
    - 有默认值的参数追加（mode、games、temperature 等）
    - 布尔开关仅 true 时追加 flag（--no-augment、--no-temp-sampling）
    - temp_seed 仅非 null 时追加

    Args:
        params: 前端传入的参数字典

    Returns:
        CLI 命令参数列表（可直接传给 subprocess.Popen）
    """
    parts = ["python", "ai/create_samples.py"]

    # 必填参数
    parts.extend(["--model", str(params["model"])])
    parts.extend(["--output", str(params["output"])])

    # 有默认值的参数
    if "mode" in params:
        parts.extend(["--mode", str(params["mode"])])
    if "games" in params:
        parts.extend(["--games", str(params["games"])])

    # 布尔开关：仅 true 时追加
    if params.get("no_augment"):
        parts.append("--no-augment")
    if params.get("no_temp_sampling"):
        parts.append("--no-temp-sampling")

    # 温度参数
    if "temperature" in params:
        parts.extend(["--temperature", str(params["temperature"])])
    if "temp_moves" in params:
        parts.extend(["--temp-moves", str(params["temp_moves"])])
    if "temp_side" in params:
        parts.extend(["--temp-side", str(params["temp_side"])])

    # 可选数值：仅非 null 时追加
    if params.get("temp_seed") is not None:
        parts.extend(["--temp-seed", str(params["temp_seed"])])

    return parts


# ------------------------------------------------------------------
# 子进程执行
# ------------------------------------------------------------------

def execute(params: Dict, project_root: str = None,
            ai_data_dir: str = None) -> Dict:
    """同步阻塞执行子进程，收集输出，发现 round_dir

    Args:
        params: 前端传入的参数字典
        project_root: 项目根目录（cwd），默认当前工作目录
        ai_data_dir: ai_data 目录路径，默认 'ai/ai_data'

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

    # 构造命令参数列表
    cmd_parts = build_cli_command(params)

    # 启动子进程（shell=False 避免注入；强制 UTF-8 输出，避免 Windows 中文乱码）
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        cmd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=project_root,
        shell=False,
        env=env,
    )

    # 阻塞等待并收集输出
    output_bytes, _ = process.communicate()
    output_text = output_bytes.decode("utf-8", errors="replace")

    # 用于展示的命令字符串
    command_display = " ".join(cmd_parts)

    if process.returncode == 0:
        # 成功路径：发现 round_dir
        round_dir_rel = discover_round_dir(ai_data_dir, s_before, project_root)

        # 扫描 round_dir 获取对局列表
        games = []
        if round_dir_rel is not None:
            round_dir_abs = os.path.join(project_root, round_dir_rel)
            try:
                games = scan_round_dir(round_dir_abs)
            except FileNotFoundError:
                pass

        return {
            "status": "completed",
            "command": command_display,
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
            "command": command_display,
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


def discover_round_dir(ai_data_dir: str, s_before: set,
                       project_root: str = None) -> Optional[str]:
    """通过差集 S_after - S_before 发现本轮数据目录

    Args:
        ai_data_dir: ai_data 目录路径
        s_before: 执行前的子目录名集合
        project_root: 项目根目录，用于计算相对路径

    Returns:
        发现的 round_dir 相对路径（相对于 project_root），
        差集为空时返回 None。
    """
    s_after = _scan_dir_names(ai_data_dir)
    diff = s_after - s_before

    if not diff:
        return None

    # 取差集中任意一个目录名，拼接完整路径后计算相对路径
    newest = sorted(diff)[-1]
    full_path = os.path.join(ai_data_dir, newest)
    if project_root:
        return os.path.relpath(full_path, project_root)
    return newest
