"""SampleBufferManager 样本缓冲区管理器模块

管理 sample_buffer/ 目录下的 FIFO 滑动窗口缓冲区。
以 10 局 chunk 为原子单位进出，总样本数上限 80000。
"""

import json
import os
from typing import List, Dict


class SampleBufferManager:
    """样本缓冲区管理器

    管理样本缓冲区的 FIFO 队列，以 chunk（对局组）为单位进出。
    每 chunk 独立落盘为一个 JSON 文件，通过 manifest.json 追踪缓冲区状态。

    Attributes:
        _buffer_dir: 缓冲区目录绝对路径
        _max_size: 缓冲区最大样本数
        _chunk_games: 每个 chunk 包含的对局数
        _chunks: 当前缓冲区中的 chunk 元数据列表
        _total_samples: 当前缓冲区样本总数
    """

    # manifest.json 文件名常量
    _MANIFEST_FILENAME: str = "manifest.json"

    def __init__(self, buffer_dir: str, max_size: int = 80000, chunk_games: int = 10):
        """初始化缓冲区管理器

        若 buffer_dir 不存在则创建；若已存在 manifest.json 则加载已有状态。

        Args:
            buffer_dir: 缓冲区目录路径（如 "sample_buffer"）
            max_size: 缓冲区最大样本数（按局面计），默认 80000
            chunk_games: 每个 chunk 包含的对局数，默认 10
        """
        self._buffer_dir: str = os.path.abspath(buffer_dir)
        self._max_size: int = max_size
        self._chunk_games: int = chunk_games
        self._chunks: List[Dict] = []
        self._total_samples: int = 0

        # 创建缓冲区目录（若不存在）
        os.makedirs(self._buffer_dir, exist_ok=True)

        # 尝试加载已有 manifest
        loaded: bool = self._load_manifest()

        # 若为新缓冲区（无 manifest），写入初始空状态
        if not loaded:
            self._save_manifest()

    # ---------- 公开接口 ----------

    def ingest(self, game_start: int, game_end: int, samples: List[dict]) -> int:
        """写入一个 chunk 到缓冲区，必要时淘汰旧 chunk

        将 game_start 到 game_end（含）的对局产生的样本写入新 chunk 文件。
        若写入后缓冲区总样本数 >= max_size，从最旧 chunk 开始逐 chunk 淘汰，
        直到总样本数 < max_size。

        Args:
            game_start: 本 chunk 起始局编号（1-based）
            game_end: 本 chunk 结束局编号（含）
            samples: 样本列表，每项 {"state": [...], "pi": [...], "z": float}

        Returns:
            本次淘汰的样本总数（0 表示未触发淘汰）
        """
        sample_count: int = len(samples)
        chunk_filename: str = f"chunk_{game_start:06d}_{game_end:06d}.json"
        chunk_path: str = os.path.join(self._buffer_dir, chunk_filename)

        # 原子写入：先写 .tmp 再 os.replace
        tmp_path: str = chunk_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)
        os.replace(tmp_path, chunk_path)

        # 记录 chunk 元数据
        self._chunks.append({
            "game_start": game_start,
            "game_end": game_end,
            "sample_count": sample_count,
            "file": chunk_filename,
        })
        self._total_samples += sample_count

        # 若超上限，淘汰最旧 chunk
        evicted_total: int = 0
        while self._total_samples >= self._max_size and len(self._chunks) > 1:
            evicted_total += self._evict_oldest()

        # 持久化 manifest
        self._save_manifest()

        return evicted_total

    def get_total_samples(self) -> int:
        """返回缓冲区当前样本总数"""
        return self._total_samples

    def get_chunk_count(self) -> int:
        """返回当前 chunk 数量"""
        return len(self._chunks)

    def get_summary(self) -> dict:
        """返回缓冲区摘要信息

        Returns:
            {"total_samples": N, "chunk_count": N, "max_size": N,
             "chunks": [{"game_range": [start, end], "sample_count": N, "file": path}, ...]}
        """
        chunks_info: List[Dict] = []
        for c in self._chunks:
            chunks_info.append({
                "game_range": [c["game_start"], c["game_end"]],
                "sample_count": c["sample_count"],
                "file": c["file"],
            })
        return {
            "total_samples": self._total_samples,
            "chunk_count": len(self._chunks),
            "max_size": self._max_size,
            "chunks": chunks_info,
        }

    # ---------- 内部方法 ----------

    def _evict_oldest(self) -> int:
        """淘汰最旧的 chunk，删除文件并从内存和 manifest 中移除

        Returns:
            被淘汰 chunk 的样本数
        """
        if not self._chunks:
            return 0

        oldest: Dict = self._chunks.pop(0)
        oldest_path: str = os.path.join(self._buffer_dir, oldest["file"])
        evicted_count: int = oldest["sample_count"]

        # 删除 chunk 文件
        if os.path.isfile(oldest_path):
            os.remove(oldest_path)

        self._total_samples -= evicted_count
        return evicted_count

    def _save_manifest(self) -> None:
        """将当前缓冲区状态原子写入 manifest.json"""
        manifest: Dict = {
            "max_size": self._max_size,
            "chunk_games": self._chunk_games,
            "total_samples": self._total_samples,
            "chunks": [
                {
                    "game_start": c["game_start"],
                    "game_end": c["game_end"],
                    "sample_count": c["sample_count"],
                    "file": c["file"],
                }
                for c in self._chunks
            ],
        }
        manifest_path: str = os.path.join(self._buffer_dir, self._MANIFEST_FILENAME)
        tmp_path: str = manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, manifest_path)

    def _load_manifest(self) -> bool:
        """从 manifest.json 加载已有缓冲区状态

        若文件不存在则返回 False（保持初始化状态）。
        若文件存在但格式损坏，静默忽略并返回 False。

        加载时自动清理：
        - 同名文件去重，仅保留最后一条（处理多次重启导致的文件复用堆积）
        - 磁盘上不存在的 chunk 文件条目移除（处理残留的已淘汰条目）
        - 以上清理后会重新计算 total_samples

        Returns:
            True 表示成功加载已有状态，False 表示新缓冲区或文件损坏
        """
        manifest_path: str = os.path.join(self._buffer_dir, self._MANIFEST_FILENAME)
        if not os.path.isfile(manifest_path):
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data: Dict = json.load(f)

            self._max_size = data.get("max_size", self._max_size)
            self._chunk_games = data.get("chunk_games", self._chunk_games)

            raw_chunks: List[Dict] = data.get("chunks", [])
            raw_total: int = data.get("total_samples", 0)

            # 去重：同名文件只保留最后一次出现（处理多次重启复用文件名）
            seen: Dict[str, int] = {}  # filename → index in raw_chunks
            for i, c in enumerate(raw_chunks):
                filename = c.get("file", "")
                if filename:
                    seen[filename] = i
            deduped: List[Dict] = [raw_chunks[i] for i in sorted(seen.values())]

            # 验存：过滤掉磁盘上不存在的 chunk（处理残留的已淘汰条目）
            valid_chunks: List[Dict] = []
            for c in deduped:
                chunk_path = os.path.join(self._buffer_dir, c.get("file", ""))
                if os.path.isfile(chunk_path):
                    valid_chunks.append({
                        "game_start": c.get("game_start", 0),
                        "game_end": c.get("game_end", 0),
                        "sample_count": c.get("sample_count", 0),
                        "file": c.get("file", ""),
                    })

            self._chunks = valid_chunks
            # 重新计算总样本数（基于实际有效 chunk）
            self._total_samples = sum(c["sample_count"] for c in valid_chunks)

            # 诊断：若加载时做了清理则输出，并自动落盘修复
            removed = len(raw_chunks) - len(valid_chunks)
            if removed > 0:
                print(f"[SampleBufferManager] manifest 清理: 移除 {removed} 条无效/重复条目"
                      f"（原有 {len(raw_chunks)}，有效 {len(valid_chunks)}）")
                print(f"[SampleBufferManager] total_samples: {raw_total} → {self._total_samples}")
                self._save_manifest()

            return True
        except (json.JSONDecodeError, IOError, KeyError):
            # 文件损坏，重置为空状态
            self._chunks = []
            self._total_samples = 0
            return False
