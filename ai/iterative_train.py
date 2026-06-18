"""迭代进化训练脚本 — 串联 create_samples / train_main / battle 为自动化迭代闭环。

用法: python ai/iterative_train.py --ai1 cnn_v4.99 --ai2 cnn_v4.100 --sample_dir ai/merge_samples2
前置: ai1 对应的 .pth 文件必须存在；不可多实例并发（.iterative_lock 保护）。
意图 13：模型标识和样本目录通过 CLI 参数指定，不再硬编码。
"""

import sys, os, re, json, shutil, subprocess, argparse
from pathlib import Path
from typing import Optional

# --- 模块级常量（意图 13：保留）---
SAMPLE_GAMES = 500
BATTLE_GAMES = 100
TRAIN_EPOCHS = 300
TRAIN_BATCH_SIZE = 4096
TEMP_MOVES = 2
TEMPERATURE = 2
PYTHON_EXECUTABLE = sys.executable


class IterativeTrainer:
    """迭代进化训练编排器。

    管理模型的迭代进化流程：复制权重→生成样本→训练→对战→无条件晋升→循环。
    意图 13：ai1/ai2 为模型 spec（如 "cnn_v4.99"），sample_dir 为样本输出目录基路径。
    """

    def __init__(self, ai1: str, ai2: str, sample_dir: str,
                 project_root: Optional[str] = None):
        """解析模型 spec、计算路径并校验前置条件。

        Args:
            ai1: 父代模型 spec，如 "cnn_v4.99"。格式为 模型名.代数。
            ai2: 子代模型 spec，如 "cnn_v4.100"。
            sample_dir: 样本输出目录基路径，如 "ai/merge_samples2"。
            project_root: 项目根目录。None 时自动推导。

        Raises:
            ValueError: ai1 与 ai2 模型名不一致。
            FileNotFoundError: ai1 的 .pth 文件不存在。
            RuntimeError: 锁文件存在（多实例并发）。
        """
        self._ai1 = ai1
        self._ai2 = ai2

        # 解析 spec：rsplit(".", 1) 仅分割最右侧点号
        self._model_name, gen_str1 = ai1.rsplit(".", 1)
        self._gen_of_ai1 = int(gen_str1)
        name2, gen_str2 = ai2.rsplit(".", 1)
        self._gen_of_ai2 = int(gen_str2)

        # 校验模型名一致
        if self._model_name != name2:
            raise ValueError(
                f"--ai1 和 --ai2 的模型名必须一致，"
                f"当前: {self._model_name} vs {name2}"
            )

        if project_root is None:
            project_root = str(Path(__file__).resolve().parent.parent)
        self.project_root = project_root

        # 动态计算路径: ai/models/{model_name}/gen{gen}/{model_name}.pth
        models_base = os.path.join(
            self.project_root, "ai", "models", self._model_name
        )
        self._ai1_path = os.path.join(
            models_base, f"gen{self._gen_of_ai1}", f"{self._model_name}.pth"
        )
        self._ai2_dir = os.path.join(models_base, f"gen{self._gen_of_ai2}")
        self._ai2_path = os.path.join(
            self._ai2_dir, f"{self._model_name}.pth"
        )
        self._lock_path = os.path.join(models_base, ".iterative_lock")
        self._sample_output_dir = os.path.join(
            self.project_root, sample_dir, f"v{self._gen_of_ai2}"
        )

        # 前置校验
        if not os.path.isfile(self._ai1_path):
            raise FileNotFoundError(
                f"{self._ai1} 权重文件不存在: {self._ai1_path}"
            )
        if os.path.isfile(self._lock_path):
            raise RuntimeError(
                f"检测到锁文件 {self._lock_path}，"
                f"可能有另一个迭代训练实例正在运行"
            )

        self._round_num = 1
        self._rounds: list = []

    # ============================================================
    # 公共方法
    # ============================================================

    def run(self) -> dict:
        """执行迭代循环：初始化→生成样本→校验→训练→对战→晋升→循环。

        Returns: 统计字典。KeyboardInterrupt 时正常返回已完成轮次统计。
        """
        # 创建锁文件
        self._create_lock()

        try:
            while True:
                # 步骤 a: 打印轮次头
                self._print_round_header()

                # 步骤 b: 模型初始化（copy ai1 → ai2）
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

                # 打印对战结果（ai1 胜场、ai2 胜场、平局数、ai2 胜率、轮次号）
                print(
                    f"{self._ai1} 胜 {gen4_wins} | "
                    f"{self._ai2} 胜 {gen5_wins} | "
                    f"平局 {draws} | "
                    f"胜率 {win_rate:.1f}% | "
                    f"i={self._round_num}"
                )

                # 步骤 h: 记录本轮统计
                round_record = {
                    "round_num": self._round_num,
                    "gen4_wins": gen4_wins,
                    "gen5_wins": gen5_wins,
                    "draws": draws,
                    "gen5_win_rate": win_rate,
                    "promoted": True,
                }

                # 步骤 i: 无条件晋升并继续循环（意图 12：不做胜率判定，始终 mv ai2→ai1）
                self._promote()
                self._rounds.append(round_record)
                self._print_round_result(win_rate)
                self._round_num += 1

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
        """创建锁文件（原子操作），防多实例并发。Raises: RuntimeError。"""
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(f"pid={os.getpid()}\n")
        except FileExistsError:
            raise RuntimeError(
                f"检测到锁文件 {self._lock_path}，"
                f"可能有另一个迭代训练实例正在运行"
            )

    def _remove_lock(self):
        """删除锁文件（幂等）。"""
        if os.path.isfile(self._lock_path):
            os.remove(self._lock_path)

    def _init_model(self):
        """复制 ai1 权重 → ai2。若 ai2 目录不存在则自动创建。"""
        os.makedirs(self._ai2_dir, exist_ok=True)
        shutil.copy2(self._ai1_path, self._ai2_path)

    def _ensure_sample_output_dir(self):
        """确保样本输出目录存在（自动创建）。"""
        os.makedirs(self._sample_output_dir, exist_ok=True)

    def _promote(self):
        """无条件晋升：mv ai2 → ai1（copy2 + remove）。

        意图 12：不做胜率判定，每轮结束后始终以 ai2 覆盖 ai1。
        这样下一轮 cp ai1→ai2 时起点为进化后的模型。

        copy2 失败时 ai1/ai2 均保留；copy2 成功后 remove 失败时 ai1 已更新、ai2 残留。
        """
        shutil.copy2(self._ai2_path, self._ai1_path)
        os.remove(self._ai2_path)

    # ============================================================
    # 内部方法：子进程调用
    # ============================================================

    def _run_subprocess(self, cmd: list) -> subprocess.CompletedProcess:
        """执行子进程（统一封装 cwd/project_root、encoding=utf-8 等公共参数）。

        Raises: RuntimeError（非零退出码）。
        """
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            encoding='utf-8',
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"子进程失败，退出码: {result.returncode}"
            )

        return result

    def _generate_samples(self) -> str:
        """调用 create_samples.py（ai1 自对弈）生成训练样本。

        Returns: 样本文件路径。Raises: RuntimeError。
        """
        self._ensure_sample_output_dir()

        sample_path = os.path.join(
            self._sample_output_dir, "path.json"
        )

        cmd = [
            PYTHON_EXECUTABLE,
            "ai/create_samples.py",
            "--model", self._ai1,
            "--sample_games", str(SAMPLE_GAMES),
            "--output", sample_path,
        ]

        result = self._run_subprocess(cmd)

        # 从 create_samples.py 的 stdout 中提取样本局数
        sample_games_count = None
        match = re.search(r'样本局数:\s*(\d+)', result.stdout)
        if match:
            sample_games_count = int(match.group(1))

        if sample_games_count is not None:
            print(f"样本生成完成: {sample_path}（样本局数: {sample_games_count}）")
        else:
            print(f"样本生成完成: {sample_path}")
        return sample_path

    def _validate_samples(self, sample_path: str):
        """校验样本文件：存在性、JSON 解析、数组类型、非空。Raises: RuntimeError。"""
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
        """调用 train_main.py 训练 ai2。Raises: RuntimeError。"""
        cmd = [
            PYTHON_EXECUTABLE,
            "ai/train_main.py",
            "--model", self._ai2,
            "--train-from-dir", self._sample_output_dir,
            "--epochs", str(TRAIN_EPOCHS),
            "--batch-size", str(TRAIN_BATCH_SIZE),
            "--shuffle",
        ]

        self._run_subprocess(cmd)
        print("训练完成")

    def _run_battle(self) -> str:
        """调用 battle.py 执行 ai1 vs ai2 对战。Returns: stdout。Raises: RuntimeError。"""
        cmd = [
            PYTHON_EXECUTABLE,
            "ai/battle.py",
            "--ai1", self._ai1,
            "--ai2", self._ai2,
            "--temp-moves", str(TEMP_MOVES),
            "--temperature", str(TEMPERATURE),
            "--temp-side", "both",
            "--games", str(BATTLE_GAMES),
        ]

        result = self._run_subprocess(cmd)
        return result.stdout

    # ============================================================
    # 内部方法：胜率解析
    # ============================================================

    def _parse_battle_output(self, stdout: str) -> dict:
        """动态正则解析对战统计。使用 re.escape() 支持任意模型名。

        Returns: {"gen4_wins", "gen5_wins", "draws", "gen5_win_rate"}。
        Raises: ValueError（无法解析 ai2 胜率）。
        """
        # 解析 ai2 胜场和胜率（优先匹配，此行为必需的解析目标）
        gen5_match = re.search(
            rf'{re.escape(self._ai2)} 胜:\s*(\d+)\s*局\s*\(([\d.]+)%\)',
            stdout
        )
        if not gen5_match:
            raise ValueError(
                f"无法解析 {self._ai2} 胜率，原始输出:\n{stdout}"
            )

        gen5_wins = int(gen5_match.group(1))
        gen5_win_rate = float(gen5_match.group(2))

        # 解析 ai1 胜场（可为 0）
        gen4_match = re.search(
            rf'{re.escape(self._ai1)} 胜:\s*(\d+)\s*局', stdout
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

    def _build_stop_reason(self, reason: str = "用户中断") -> str:
        """构建停止原因文本。_rounds 为空时返回空字符串。"""
        if not self._rounds:
            return ""
        last = self._rounds[-1]
        return (
            f"第{last['round_num']}轮 "
            f"{self._ai2} 胜率 {last['gen5_win_rate']:.1f}%，"
            f"{reason}"
        )

    def _print_round_header(self):
        """打印轮次头：========== 第 {n} 轨迭代开始 =========="""
        print(f"========== 第 {self._round_num} 轨迭代开始 ==========")

    def _print_round_result(self, win_rate: float):
        """打印轮次结果摘要：第 {n} 轨结果: {ai2} 胜率 = {r}% → 晋升"""
        print(
            f"第 {self._round_num} 轨结果: "
            f"{self._ai2} 胜率 = {win_rate:.1f}% → 晋升"
        )

    def _print_final_report(self):
        """打印最终统计报告。_rounds 为空时提前返回。"""
        if not self._rounds:
            return

        total_iterations = len(self._rounds)

        print("\n========== 最终统计 ==========")
        print(f"总迭代次数: {total_iterations}")

        print("逐轮明细:")
        for r in self._rounds:
            print(
                f"  第{r['round_num']}轮: "
                f"{self._ai1} 胜 {r['gen4_wins']} | "
                f"{self._ai2} 胜 {r['gen5_wins']} | "
                f"平局 {r['draws']} | "
                f"{self._ai2} 胜率 {r['gen5_win_rate']:.1f}%"
            )

        # 停止原因（通过 _build_stop_reason() 共用逻辑）
        stop_reason = self._build_stop_reason()
        print(f"停止原因: {stop_reason}")

        print(
            f"当前模型: gen{self._gen_of_ai1}/{self._model_name}.pth"
            f"（经 {total_iterations} 轮进化）"
        )

    # ============================================================
    # 内部方法：结果构建
    # ============================================================

    def _build_result_dict(self) -> dict:
        """构建 run() 返回的统计字典。final_round 取最后完成轮次编号。"""
        total_iterations = len(self._rounds)

        # 取最后完成轮次编号（KeyboardInterrupt 时 _round_num 可能指向未完成轮次）
        if self._rounds:
            final_round = self._rounds[-1]["round_num"]
        else:
            final_round = 0

        return {
            "total_iterations": total_iterations,
            "final_round": final_round,
            "rounds": list(self._rounds),
            "stop_reason": self._build_stop_reason(),
            "final_model_path": os.path.abspath(self._ai1_path),
        }


def main():
    """CLI 入口：解析 --ai1/--ai2/--sample_dir 参数并运行迭代训练。"""
    parser = argparse.ArgumentParser(description="迭代进化训练")
    parser.add_argument('--ai1', type=str, required=True, help='父代模型 spec，如 cnn_v4.99')
    parser.add_argument('--ai2', type=str, required=True, help='子代模型 spec，如 cnn_v4.100')
    parser.add_argument('--sample_dir', type=str, required=True, help='样本输出目录基路径，如 ai/merge_samples2')
    args = parser.parse_args()

    # 校验: ai1/ai2 格式（含 . 且 gen 为正整数）
    try:
        name1, gen_str1 = args.ai1.rsplit(".", 1)
        _gen1 = int(gen_str1)  # 仅用于格式校验，gen 值由 __init__ 重新解析
        name2, gen_str2 = args.ai2.rsplit(".", 1)
        _gen2 = int(gen_str2)  # 仅用于格式校验
    except ValueError:
        parser.error(
            "--ai1 和 --ai2 的格式必须为 模型名.代数，如 cnn_v4.99"
        )
        return  # parser.error() 会调用 sys.exit()，此行仅为静态分析

    # 校验: ai1 与 ai2 模型名一致
    if name1 != name2:
        parser.error(
            f"--ai1 和 --ai2 的模型名必须一致，"
            f"当前: {name1} vs {name2}"
        )

    try:
        trainer = IterativeTrainer(
            ai1=args.ai1,
            ai2=args.ai2,
            sample_dir=args.sample_dir,
        )
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
    except OSError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
