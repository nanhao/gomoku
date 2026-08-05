"""SampleBufferManager 单元测试

测试样本缓冲区管理器的构造、写入、淘汰、摘要和持久化。
测试使用临时目录，测试结束后自动清理。
"""

import json
import os
import sys
import tempfile

# 确保项目根在 sys.path 中
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pytest

# 注：此时 SampleBufferManager 尚未实现，以下 import 在 RED 阶段会失败
# 在 GREEN 阶段实现后即可通过
from ai.sample_buffer_manager import SampleBufferManager


def _make_samples(count: int) -> list:
    """生成指定数量的模拟样本，每项含 state/pi/z"""
    samples = []
    for i in range(count):
        samples.append({
            "state": [0.0] * 225 + [float((i % 2) + 1)],  # 226 维
            "pi": [1.0 / 225] * 225,                       # 225 维
            "z": 1.0 if i % 2 == 0 else -1.0,
        })
    return samples


class TestSampleBufferManagerConstruction:
    """TC3.1, TC3.2 — 构造器行为"""

    def test_create_dir_and_manifest(self):
        """TC3.1: 临时目录不存在时，构造器创建目录和空 manifest"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            buffer_dir = os.path.join(tmp_dir, "sample_buffer")
            assert not os.path.exists(buffer_dir)

            mgr = SampleBufferManager(buffer_dir, max_size=100, chunk_games=2)

            # 目录和 manifest 已创建
            assert os.path.isdir(buffer_dir)
            manifest_path = os.path.join(buffer_dir, "manifest.json")
            assert os.path.isfile(manifest_path)

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["total_samples"] == 0
            assert manifest["chunks"] == []
            assert manifest["max_size"] == 100
            assert manifest["chunk_games"] == 2

    def test_load_existing_state(self):
        """TC3.2: 已有 manifest 和 chunk 文件时，构造器加载已有状态"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            buffer_dir = os.path.join(tmp_dir, "sample_buffer")
            os.makedirs(buffer_dir, exist_ok=True)

            # 手动写入 manifest 和两个 chunk 文件
            chunk1_file = "chunk_000001_000002.json"
            chunk2_file = "chunk_000003_000004.json"
            chunk1_path = os.path.join(buffer_dir, chunk1_file)
            chunk2_path = os.path.join(buffer_dir, chunk2_file)

            samples1 = _make_samples(80)
            samples2 = _make_samples(90)

            with open(chunk1_path, "w", encoding="utf-8") as f:
                json.dump(samples1, f)
            with open(chunk2_path, "w", encoding="utf-8") as f:
                json.dump(samples2, f)

            manifest = {
                "max_size": 200,
                "chunk_games": 2,
                "total_samples": 170,
                "chunks": [
                    {"game_start": 1, "game_end": 2, "sample_count": 80, "file": chunk1_file},
                    {"game_start": 3, "game_end": 4, "sample_count": 90, "file": chunk2_file},
                ],
            }
            manifest_path = os.path.join(buffer_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            mgr = SampleBufferManager(buffer_dir, max_size=200, chunk_games=2)
            assert mgr.get_total_samples() == 170
            assert mgr.get_chunk_count() == 2


class TestSampleBufferManagerIngest:
    """TC3.3 ~ TC3.6 — ingest 行为"""

    def test_ingest_first_chunk(self):
        """TC3.3: 写入首个 chunk"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = SampleBufferManager(tmp_dir, max_size=200, chunk_games=2)
            samples = _make_samples(80)
            evicted = mgr.ingest(1, 2, samples)

            assert evicted == 0
            assert mgr.get_total_samples() == 80
            assert mgr.get_chunk_count() == 1

            # 验证 chunk 文件存在且内容正确
            chunk_path = os.path.join(tmp_dir, "chunk_000001_000002.json")
            assert os.path.isfile(chunk_path)
            with open(chunk_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == 80

            # 验证 manifest 已更新
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["total_samples"] == 80
            assert len(manifest["chunks"]) == 1
            assert manifest["chunks"][0]["sample_count"] == 80

    def test_ingest_without_eviction(self):
        """TC3.4: 缓冲区未超上限，不触发淘汰"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = SampleBufferManager(tmp_dir, max_size=200, chunk_games=2)

            e1 = mgr.ingest(1, 2, _make_samples(80))
            assert e1 == 0
            e2 = mgr.ingest(3, 4, _make_samples(90))
            assert e2 == 0

            assert mgr.get_total_samples() == 170
            assert mgr.get_chunk_count() == 2

            # 两个 chunk 文件均存在
            assert os.path.isfile(os.path.join(tmp_dir, "chunk_000001_000002.json"))
            assert os.path.isfile(os.path.join(tmp_dir, "chunk_000003_000004.json"))

    def test_ingest_with_single_eviction(self):
        """TC3.5: 超上限时淘汰最旧的一个 chunk"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = SampleBufferManager(tmp_dir, max_size=150, chunk_games=2)

            mgr.ingest(1, 2, _make_samples(80))
            # 80 + 90 = 170 > 150，应淘汰 chunk1
            evicted = mgr.ingest(3, 4, _make_samples(90))

            assert evicted == 80
            assert mgr.get_total_samples() == 90
            assert mgr.get_chunk_count() == 1

            # chunk1 文件应被删除
            assert not os.path.isfile(os.path.join(tmp_dir, "chunk_000001_000002.json"))
            # chunk2 文件应存在
            assert os.path.isfile(os.path.join(tmp_dir, "chunk_000003_000004.json"))

    def test_ingest_with_multiple_evictions(self):
        """TC3.6: 超上限时逐次淘汰多个旧 chunk

        chunk2 ingest 时触发淘汰 chunk1(80)；chunk3 ingest 时触发淘汰 chunk2(70)。
        每次 ingest 返回当次淘汰的样本数。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = SampleBufferManager(tmp_dir, max_size=100, chunk_games=2)

            # chunk1: 80 样本，未超限
            e1 = mgr.ingest(1, 2, _make_samples(80))
            assert e1 == 0

            # chunk2: 80+70=150 >= 100，淘汰 chunk1(80)
            e2 = mgr.ingest(3, 4, _make_samples(70))
            assert e2 == 80
            assert not os.path.isfile(os.path.join(tmp_dir, "chunk_000001_000002.json"))

            # chunk3: 70+60=130 >= 100，淘汰 chunk2(70)
            e3 = mgr.ingest(5, 6, _make_samples(60))
            assert e3 == 70

            assert mgr.get_total_samples() == 60
            assert mgr.get_chunk_count() == 1

            # 仅 chunk3 保留
            assert not os.path.isfile(os.path.join(tmp_dir, "chunk_000003_000004.json"))
            assert os.path.isfile(os.path.join(tmp_dir, "chunk_000005_000006.json"))


class TestSampleBufferManagerSummary:
    """TC3.7 — get_summary"""

    def test_summary_returns_complete_info(self):
        """TC3.7: get_summary 返回完整摘要信息"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = SampleBufferManager(tmp_dir, max_size=200, chunk_games=2)
            mgr.ingest(1, 2, _make_samples(80))
            mgr.ingest(3, 4, _make_samples(90))

            summary = mgr.get_summary()
            assert summary["total_samples"] == 170
            assert summary["chunk_count"] == 2
            assert summary["max_size"] == 200
            assert len(summary["chunks"]) == 2

            for chunk in summary["chunks"]:
                assert "game_range" in chunk
                assert "sample_count" in chunk
                assert "file" in chunk
                assert len(chunk["game_range"]) == 2


class TestSampleBufferManagerPersistence:
    """TC3.8 — 持久化"""

    def test_manifest_reload(self):
        """TC3.8: write → new instance → state matches"""
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_obj.name
        try:
            mgr1 = SampleBufferManager(tmp_dir, max_size=200, chunk_games=2)
            mgr1.ingest(1, 2, _make_samples(80))
            total1 = mgr1.get_total_samples()

            # 新建实例加载同一目录
            mgr2 = SampleBufferManager(tmp_dir, max_size=200, chunk_games=2)
            assert mgr2.get_total_samples() == total1
            assert mgr2.get_chunk_count() == 1
        finally:
            tmp_dir_obj.cleanup()
