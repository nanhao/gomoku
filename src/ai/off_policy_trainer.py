"""CNN v11 off-policy 监督训练器模块

继承 CNNV11Trainer，仅覆盖样本加载逻辑（_load_samples），
训练能力（train_epochs/_train_step/_convert_inputs/_plot_loss_curve/save_model）
全部继承复用。面向静态样本集（如 renjunet_samples 人类棋谱）做多 epoch 全量训练。

样本来源优先级（与设计 58 interface.md 一致）：
1. manifest.json → 按 chunks 列表逐个加载 chunk JSON 文件
2. sample.json → 兼容旧样本目录
3. 目录扫描 → 加载全部 .json（跳过 manifest.json）
"""

import json
import os
from typing import List, Tuple

from ai.models.cnn_v11.trainer import CNNV11Trainer


class OffPolicyTrainer(CNNV11Trainer):
    """CNN v11 off-policy 监督训练器（静态数据集多 epoch）

    覆盖基类静态方法 _load_samples，支持 chunk JSON 目录
    （manifest.json + chunk_*.json，与 sample_buffer/renjunet_samples 同格式），
    并兼容旧 sample.json 目录。

    容错与校验规则（设计 58 interface.md）：
    - 损坏的 JSON 文件：跳过并打印警告，不中断加载
    - manifest 中记录但磁盘缺失的文件：静默跳过
    - 单条样本缺失 state/pi/z 字段：跳过并计数（打印提示）
    - state 长度 ≠ 226 或 pi 长度 ≠ 225：跳过并计数
    - 无有效样本时返回空列表（调用方据此报错）
    """

    # 样本字段维度常量（与模型输入/输出维度对应）
    STATE_DIM: int = 226
    PI_DIM: int = 225

    @staticmethod
    def _load_samples(
        sample_dirs: List[str],
    ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """加载目录列表中的全部样本（覆盖基类同名静态方法）

        Args:
            sample_dirs: 样本目录列表（每个目录独立应用加载规则）

        Returns:
            (states, pis, zs) 三元组；无有效样本时为空列表
        """
        all_states: List[List[int]] = []
        all_pis: List[List[float]] = []
        all_zs: List[float] = []
        for data_dir in sample_dirs:
            states, pis, zs = OffPolicyTrainer._load_dir(data_dir)
            all_states.extend(states)
            all_pis.extend(pis)
            all_zs.extend(zs)
        return all_states, all_pis, all_zs

    @staticmethod
    def _load_dir(data_dir: str) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """按优先级规则加载单个目录

        Args:
            data_dir: 样本目录路径

        Returns:
            (states, pis, zs) 三元组
        """
        # 规则 1：manifest.json 优先
        manifest_path = os.path.join(data_dir, "manifest.json")
        if os.path.exists(manifest_path):
            return OffPolicyTrainer._load_manifest(data_dir, manifest_path)

        # 规则 2：sample.json 兼容
        sample_path = os.path.join(data_dir, "sample.json")
        if os.path.exists(sample_path):
            return OffPolicyTrainer._load_json_file(sample_path)

        # 规则 3：目录扫描
        return OffPolicyTrainer._scan_dir(data_dir)

    @staticmethod
    def _load_manifest(
        data_dir: str, manifest_path: str
    ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """按 manifest 的 chunks 列表逐个加载 chunk 文件

        Args:
            data_dir: 样本目录路径（chunk 文件所在目录）
            manifest_path: manifest.json 完整路径

        Returns:
            (states, pis, zs) 三元组
        """
        states: List[List[int]] = []
        pis: List[List[float]] = []
        zs: List[float] = []

        # manifest 自身损坏时跳过并警告，不中断加载
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[警告] 跳过损坏的 manifest 文件: {manifest_path}（{exc}）")
            return states, pis, zs

        # 遍历 chunks 列表，逐个加载 chunk 文件
        chunks = manifest.get("chunks", []) if isinstance(manifest, dict) else []
        for entry in chunks:
            if not isinstance(entry, dict):
                continue
            chunk_name = entry.get("file")
            if not chunk_name:
                continue
            chunk_path = os.path.join(data_dir, chunk_name)
            # manifest 记录但磁盘缺失：静默跳过
            if not os.path.isfile(chunk_path):
                continue
            chunk_states, chunk_pis, chunk_zs = OffPolicyTrainer._load_json_file(chunk_path)
            states.extend(chunk_states)
            pis.extend(chunk_pis)
            zs.extend(chunk_zs)
        return states, pis, zs

    @staticmethod
    def _load_json_file(
        file_path: str,
    ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """加载单个样本 JSON 文件（含容错与字段校验）

        Args:
            file_path: 样本 JSON 文件路径

        Returns:
            (states, pis, zs) 三元组；损坏文件返回空列表
        """
        states: List[List[int]] = []
        pis: List[List[float]] = []
        zs: List[float] = []

        # 损坏 JSON：跳过并打印警告，不中断加载
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[警告] 跳过损坏的 JSON 文件: {file_path}（{exc}）")
            return states, pis, zs

        # 非列表内容：视为无效文件
        if not isinstance(samples, list):
            print(f"[警告] 跳过非列表 JSON 文件: {file_path}")
            return states, pis, zs

        # 逐条校验：字段完整性 + 维度匹配，非法样本跳过并计数
        skipped = 0
        for sample in samples:
            if not isinstance(sample, dict):
                skipped += 1
                continue
            if "state" not in sample or "pi" not in sample or "z" not in sample:
                skipped += 1
                continue
            state = sample["state"]
            pi = sample["pi"]
            if len(state) != OffPolicyTrainer.STATE_DIM or len(pi) != OffPolicyTrainer.PI_DIM:
                skipped += 1
                continue
            states.append(state)
            pis.append(pi)
            zs.append(sample["z"])

        # 打印跳过统计提示
        if skipped > 0:
            print(f"[提示] 跳过 {skipped} 条无效样本: {file_path}")
        return states, pis, zs

    @staticmethod
    def _scan_dir(
        data_dir: str,
    ) -> Tuple[List[List[int]], List[List[float]], List[float]]:
        """扫描目录下全部 .json 文件逐个加载（跳过 manifest.json）

        Args:
            data_dir: 样本目录路径

        Returns:
            (states, pis, zs) 三元组
        """
        states: List[List[int]] = []
        pis: List[List[float]] = []
        zs: List[float] = []
        for file_name in sorted(os.listdir(data_dir)):
            if not file_name.endswith(".json") or file_name == "manifest.json":
                continue
            file_path = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_path):
                continue
            file_states, file_pis, file_zs = OffPolicyTrainer._load_json_file(file_path)
            states.extend(file_states)
            pis.extend(file_pis)
            zs.extend(file_zs)
        return states, pis, zs
