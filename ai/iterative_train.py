"""cnn_v3 迭代进化训练脚本 — 串联 create_samples / train_main / battle 为自动化迭代闭环。

用法: python ai/iterative_train.py
前置: ai/models/cnn_v3/gen4/cnn_v3.pth 必须存在；不可多实例并发（.iterative_lock 保护）。
v1.0 所有参数硬编码，不做命令行参数化。
"""

import sys
import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# --- 硬编码常量（v1.0 不可配置）---
BASE_MODEL_NAME = "cnn_v3"
PARENT_GEN = 4          # 父代编号（基线模型）
CHILD_GEN = 5           # 子代编号（待训练模型）
MODELS_DIR = "ai/models/cnn_v3"          # 权重目录（相对于 project_root）
SAMPLE_OUTPUT_DIR = "ai/merge_samples/v4"  # 样本输出目录
SAMPLE_GAMES = 500      # 目标样本局数
BATTLE_GAMES = 100      # 对战局数
TRAIN_EPOCHS = 1        # 训练 epoch 数
TRAIN_BATCH_SIZE = 4096 # 训练 batch size
WIN_RATE_THRESHOLD = 50.0  # 晋升胜率阈值（%，与 battle.py 输出精度 .1f 对齐）
PYTHON_EXECUTABLE = sys.executable  # 子进程 Python 解释器


class IterativeTrainer:
    """迭代进化训练编排器。

    管理 cnn_v3 的迭代进化流程：复制权重→生成样本→训练→对战→判定晋升/停止。
    公共接口：__init__(project_root=None) 校验前置条件；run()->dict 执行循环。
    """

    def __init__(self, project_root: Optional[str] = None):
        """校验前置条件：gen4 权重存在、锁文件不存在。

        Args:
            project_root: 项目根目录。None 时自动推导为 Path(__file__).parent.parent。
        Raises:
            FileNotFoundError: gen4/cnn_v3.pth 不存在。
            RuntimeError: 锁文件存在（多实例并发）。
        """
        # 推导项目根目录
        if project_root is None:
            project_root = str(Path(__file__).resolve().parent.parent)
        self.project_root = project_root

        # 构建关键路径
        self._gen4_path = os.path.join(
            self.project_root, MODELS_DIR, f"gen{PARENT_GEN}", "cnn_v3.pth"
        )
        self._gen5_dir = os.path.join(
            self.project_root, MODELS_DIR, f"gen{CHILD_GEN}"
        )
        self._gen5_path = os.path.join(self._gen5_dir, "cnn_v3.pth")
        self._lock_path = os.path.join(
            self.project_root, MODELS_DIR, ".iterative_lock"
        )
        self._sample_output_dir = os.path.join(
            self.project_root, SAMPLE_OUTPUT_DIR
        )

        # 前置校验：gen4 权重存在
        if not os.path.isfile(self._gen4_path):
            raise FileNotFoundError(
                f"gen{PARENT_GEN} 权重文件不存在: {self._gen4_path}"
            )

        # 前置校验：锁文件不存在（防止多实例并发）
        if os.path.isfile(self._lock_path):
            raise RuntimeError(
                f"检测到锁文件 {self._lock_path}，"
                f"可能有另一个迭代训练实例正在运行"
            )

        # 内部状态
        self._round_num = 1
        self._rounds: list = []

    # ============================================================
    # 公共方法
    # ============================================================

    def run(self) -> dict:
        """执行迭代循环：初始化→生成样本→校验→训练→对战→判定晋升/停止。

        Returns: 统计字典（total_iterations, final_round, rounds, stop_reason, final_model_path）。
        KeyboardInterrupt 时返回已完成轮次的统计（正常返回，不抛异常）。
        """
        # 创建锁文件
        self._create_lock()

        try:
            while True:
                # 步骤 a: 打印轮次头
                self._print_round_header()

                # 步骤 b: 模型初始化（copy gen4 → gen5）
                self._init_model()

                # 步骤 c: 样本生成
                sample_path = self._generate_samples()

                # 步骤 d: 样本校验（非空）
                self._validate_samples(sample_path)

                # 步骤 e: 模型训练
                self._train_model()

                # 步骤 f: 对战评估
                battle_stdout = self._run_battle()

                # 步骤 g: 解析胜率和统计数据
                stats = self._parse_battle_output(battle_stdout)
                win_rate = stats["gen5_win_rate"]
                gen4_wins = stats["gen4_wins"]
                gen5_wins = stats["gen5_wins"]
                draws = stats["draws"]

                # 打印对战结果（cnn_v3.4 胜场、cnn_v3.5 胜场、平局数、cnn_v3.5 胜率）
                print(
                    f"cnn_v3.4 胜 {gen4_wins} | "
                    f"cnn_v3.5 胜 {gen5_wins} | "
                    f"平局 {draws} | "
                    f"胜率 {win_rate:.1f}%"
                )

                # 步骤 h: 记录本轮统计
                round_record = {
                    "round_num": self._round_num,
                    "gen4_wins": gen4_wins,
                    "gen5_wins": gen5_wins,
                    "draws": draws,
                    "gen5_win_rate": win_rate,
                    "promoted": False,
                }

                # 步骤 i: 胜率判定
                if win_rate > WIN_RATE_THRESHOLD:
                    # 晋升
                    self._promote()
                    round_record["promoted"] = True
                    self._rounds.append(round_record)
                    self._print_round_result(win_rate, promoted=True)
                    self._round_num += 1
                else:
                    # 停止
                    self._rounds.append(round_record)
                    self._print_round_result(win_rate, promoted=False)
                    self._print_final_report()
                    break

        except KeyboardInterrupt:
            # 捕获 Ctrl+C，输出已完成轮次的统计
            print("\n用户中断")
            if self._rounds:
                self._print_final_report()
        finally:
            # 确保锁文件被清理
            self._remove_lock()

        return self._build_result_dict()

    # ============================================================
    # 内部方法：文件操作
    # ============================================================

    def _create_lock(self):
        """创建锁文件（写入当前 PID），防多实例并发运行。"""
        with open(self._lock_path, 'w', encoding='utf-8') as f:
            f.write(f"pid={os.getpid()}\n")

    def _remove_lock(self):
        """删除锁文件（幂等操作）。"""
        if os.path.isfile(self._lock_path):
            os.remove(self._lock_path)

    def _init_model(self):
        """复制 gen4 权重 → gen5。若 gen5/ 不存在则自动创建。"""
        os.makedirs(self._gen5_dir, exist_ok=True)
        shutil.copy2(self._gen4_path, self._gen5_path)

    def _ensure_sample_output_dir(self):
        """确保样本输出目录存在（不存在则自动创建）。"""
        os.makedirs(self._sample_output_dir, exist_ok=True)

    def _promote(self):
        """晋升：copy2 gen5 → gen4，成功后删除 gen5。

        copy2 失败时 gen4/gen5 均保留；copy2 成功后 remove 失败时 gen4 已更新、gen5 残留。
        """
        shutil.copy2(self._gen5_path, self._gen4_path)
        os.remove(self._gen5_path)

    # ============================================================
    # 内部方法：子进程调用
    # ============================================================

    def _generate_samples(self) -> str:
        """调用 create_samples.py（cnn_v3.4 自对弈，winner_path 模式）生成训练样本。

        Returns: 样本文件路径。Raises: RuntimeError（子进程非零退出码）。
        """
        self._ensure_sample_output_dir()

        sample_path = os.path.join(
            self._sample_output_dir, f"path{self._round_num}.json"
        )

        cmd = [
            PYTHON_EXECUTABLE,
            "ai/create_samples.py",
            "--model", "cnn_v3.4",
            "--mode", "winner_path",
            "--sample_games", str(SAMPLE_GAMES),
            "--output", sample_path,
            "--temp-side", "random",
        ]

        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"create_samples.py 失败，退出码: {result.returncode}"
            )

        print(f"样本生成完成: {sample_path}")
        return sample_path

    def _validate_samples(self, sample_path: str):
        """校验样本文件：存在性、JSON 解析、数组类型、非空。

        Raises: RuntimeError（文件不存在/JSON 解析失败/非数组/空数组）。
        """
        if not os.path.isfile(sample_path):
            raise RuntimeError(f"样本文件不存在: {sample_path}")

        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"样本文件 JSON 解析失败: {sample_path}, 错误: {e}"
            )

        if not isinstance(data, list):
            raise RuntimeError(
                f"样本文件格式异常：期望 JSON 数组，实际为 {type(data).__name__}"
            )

        if len(data) == 0:
            raise RuntimeError(f"空样本: {sample_path} 包含 0 条样本")

    def _train_model(self):
        """调用 train_main.py 训练 cnn_v3.5（1 epoch, batch 4096, shuffle）。

        Raises: RuntimeError（子进程非零退出码）。
        """
        sample_path = os.path.join(
            self._sample_output_dir, f"path{self._round_num}.json"
        )

        cmd = [
            PYTHON_EXECUTABLE,
            "ai/train_main.py",
            "--model", "cnn_v3.5",
            "--train-from-dir", sample_path,
            "--epochs", str(TRAIN_EPOCHS),
            "--batch-size", str(TRAIN_BATCH_SIZE),
            "--shuffle",
        ]

        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"train_main.py 失败，退出码: {result.returncode}"
            )

        print("训练完成")

    def _run_battle(self) -> str:
        """调用 battle.py 执行 cnn_v3.4 vs cnn_v3.5 对战。

        Returns: battle.py 标准输出（含逐局结果和最终统计）。
        Raises: RuntimeError（子进程非零退出码）。
        """
        cmd = [
            PYTHON_EXECUTABLE,
            "ai/battle.py",
            "--ai1", "cnn_v3.4",
            "--ai2", "cnn_v3.5",
            "--games", str(BATTLE_GAMES),
        ]

        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"battle.py 失败，退出码: {result.returncode}"
            )

        return result.stdout

    # ============================================================
    # 内部方法：胜率解析
    # ============================================================

    def _parse_battle_output(self, stdout: str) -> dict:
        """从 battle.py 输出解析对战统计。

        匹配 cnn_v3.5 胜场/胜率、cnn_v3.4 胜场、平局数（精确正则，避免误匹配 cnn_v3.4 行）。

        Returns: {"gen4_wins", "gen5_wins", "draws", "gen5_win_rate"}。
        Raises: ValueError（无法解析 cnn_v3.5 胜率）。
        """
        # 解析 cnn_v3.5 胜场和胜率（优先匹配，此行为必需的解析目标）
        gen5_match = re.search(
            r'cnn_v3\.5 胜:\s*(\d+)\s*局\s*\(([\d.]+)%\)', stdout
        )
        if not gen5_match:
            raise ValueError(
                f"无法解析 cnn_v3.5 胜率，原始输出:\n{stdout}"
            )

        gen5_wins = int(gen5_match.group(1))
        gen5_win_rate = float(gen5_match.group(2))

        # 解析 cnn_v3.4 胜场（可为 0）
        gen4_match = re.search(
            r'cnn_v3\.4 胜:\s*(\d+)\s*局', stdout
        )
        gen4_wins = int(gen4_match.group(1)) if gen4_match else 0

        # 解析平局数（可为 0）
        draw_match = re.search(r'平局:\s*(\d+)\s*局', stdout)
        draws = int(draw_match.group(1)) if draw_match else 0

        return {
            "gen4_wins": gen4_wins,
            "gen5_wins": gen5_wins,
            "draws": draws,
            "gen5_win_rate": gen5_win_rate,
        }

    # ============================================================
    # 内部方法：日志输出
    # ============================================================

    def _print_round_header(self):
        """打印轮次头：========== 第 {n} 轨迭代开始 =========="""
        print(f"========== 第 {self._round_num} 轨迭代开始 ==========")

    def _print_round_result(self, win_rate: float, promoted: bool):
        """打印轮次结果摘要：第 {n} 轨结果: cnn_v3.5 胜率 = {r}% → {晋升|停止}"""
        action = "晋升" if promoted else "停止"
        print(
            f"第 {self._round_num} 轨结果: "
            f"cnn_v3.5 胜率 = {win_rate:.1f}% → {action}"
        )

    def _print_final_report(self):
        """打印最终统计报告（总迭代次数、逐轮明细表、停止原因、最终优胜模型）。"""
        total_iterations = sum(1 for r in self._rounds if r["promoted"])

        print("\n========== 最终统计 ==========")
        print(f"总迭代次数: {total_iterations}")

        print("逐轮明细:")
        for r in self._rounds:
            action = "晋升" if r["promoted"] else "停止"
            print(
                f"  第{r['round_num']}轮: "
                f"cnn_v3.4 胜 {r['gen4_wins']} | "
                f"cnn_v3.5 胜 {r['gen5_wins']} | "
                f"平局 {r['draws']} | "
                f"cnn_v3.5 胜率 {r['gen5_win_rate']:.1f}% → {action}"
            )

        # 停止原因
        if self._rounds:
            last = self._rounds[-1]
            stop_reason = (
                f"第{last['round_num']}轮 "
                f"cnn_v3.5 胜率 {last['gen5_win_rate']:.1f}% "
                f"≤ {WIN_RATE_THRESHOLD}%，停止迭代"
            )
            print(f"停止原因: {stop_reason}")

        print(
            f"最终优胜模型: gen{PARENT_GEN}/cnn_v3.pth"
            f"（经 {total_iterations} 轮进化）"
        )

    # ============================================================
    # 内部方法：结果构建
    # ============================================================

    def _build_result_dict(self) -> dict:
        """构建 run() 返回的统计字典。

        Returns: {"total_iterations", "final_round", "rounds", "stop_reason", "final_model_path"}。
        """
        total_iterations = sum(1 for r in self._rounds if r["promoted"])

        if self._rounds:
            last = self._rounds[-1]
            stop_reason = (
                f"第{last['round_num']}轮 "
                f"cnn_v3.5 胜率 {last['gen5_win_rate']:.1f}% "
                f"≤ {WIN_RATE_THRESHOLD}%，停止迭代"
            )
        else:
            stop_reason = ""

        return {
            "total_iterations": total_iterations,
            "final_round": self._round_num,
            "rounds": list(self._rounds),
            "stop_reason": stop_reason,
            "final_model_path": os.path.abspath(self._gen4_path),
        }


def main():
    """CLI 入口：创建 IterativeTrainer 并运行。捕获异常后以非零退出码退出。

    KeyboardInterrupt 由 run() 内部处理，main() 不重复捕获。
    """
    try:
        trainer = IterativeTrainer()
        trainer.run()
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
