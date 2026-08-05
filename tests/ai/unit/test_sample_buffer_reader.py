"""SampleBufferReader 单元测试

测试从样本缓冲区读取 chunk JSON 文件并进行随机抽样的功能。
"""
import json
import os
import tempfile

import pytest

from ai.sample_buffer_reader import SampleBufferReader


def _make_sample(state_val: int = 1, pi_val: float = 0.5, z_val: float = 1.0) -> dict:
    """创建一条符合缓冲区格式的测试样本

    Args:
        state_val: state 向量的填充值
        pi_val: pi 向量的填充值
        z_val: z 值

    Returns:
        样本字典，含 state (226维), pi (225维), z
    """
    return {
        "state": [state_val] * 226,
        "pi": [pi_val] * 225,
        "z": z_val,
    }


def _write_chunk(buffer_dir: str, game_start: int, game_end: int, samples: list) -> str:
    """在缓冲区目录写入一个 chunk JSON 文件

    Args:
        buffer_dir: 缓冲区目录
        game_start: 起始局号
        game_end: 结束局号
        samples: 样本列表

    Returns:
        chunk 文件名
    """
    filename = f"chunk_{game_start:06d}_{game_end:06d}.json"
    path = os.path.join(buffer_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f)
    return filename


def _write_manifest(buffer_dir: str, chunks: list, max_size: int = 80000, chunk_games: int = 10) -> None:
    """写入 manifest.json

    Args:
        buffer_dir: 缓冲区目录
        chunks: chunk 元数据列表
        max_size: 缓冲区最大容量
        chunk_games: 每 chunk 对局数
    """
    total_samples = sum(c["sample_count"] for c in chunks)
    manifest = {
        "max_size": max_size,
        "chunk_games": chunk_games,
        "total_samples": total_samples,
        "chunks": chunks,
    }
    path = os.path.join(buffer_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ========== TC-SBR-01: 空缓冲区 ==========

def test_empty_buffer_returns_zero_samples():
    """空缓冲区：无 chunk 文件时 total_samples=0，random_sample 返回空列表"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 只创建 manifest.json，无任何 chunk
        _write_manifest(tmpdir, [], max_size=80000, chunk_games=10)

        reader = SampleBufferReader(tmpdir)
        assert reader.total_samples() == 0
        result = reader.random_sample(10)
        assert result == []


# ========== TC-SBR-02: 单 chunk 抽样 ==========

def test_single_chunk_random_sample():
    """单 chunk 含 100 条样本，抽取 50 条应返回 50 条且格式正确"""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples = [_make_sample(state_val=i, z_val=float(i % 3 - 1)) for i in range(100)]
        filename = _write_chunk(tmpdir, 1, 10, samples)
        _write_manifest(tmpdir, [{
            "game_start": 1,
            "game_end": 10,
            "sample_count": 100,
            "file": filename,
        }])

        reader = SampleBufferReader(tmpdir)
        assert reader.total_samples() == 100

        result = reader.random_sample(50)
        assert len(result) == 50
        for item in result:
            assert "state" in item
            assert "pi" in item
            assert "z" in item
            assert len(item["state"]) == 226
            assert len(item["pi"]) == 225
            assert isinstance(item["z"], (int, float))


# ========== TC-SBR-03: 请求数超过总量 ==========

def test_request_more_than_total():
    """总样本 30 条，random_sample(100) 返回 30 条"""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples = [_make_sample() for _ in range(30)]
        filename = _write_chunk(tmpdir, 1, 3, samples)
        _write_manifest(tmpdir, [{
            "game_start": 1, "game_end": 3,
            "sample_count": 30, "file": filename,
        }])

        reader = SampleBufferReader(tmpdir)
        result = reader.random_sample(100)
        assert len(result) == 30


# ========== TC-SBR-04: 不放回抽样 ==========

def test_sampling_without_replacement():
    """连续两次 random_sample 的结果不应重复"""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples = [_make_sample(state_val=i, z_val=float(i)) for i in range(100)]
        filename = _write_chunk(tmpdir, 1, 10, samples)
        _write_manifest(tmpdir, [{
            "game_start": 1, "game_end": 10,
            "sample_count": 100, "file": filename,
        }])

        reader = SampleBufferReader(tmpdir)
        first = reader.random_sample(60)
        second = reader.random_sample(60)

        # 第二次应该只能拿到剩余 40 条（已抽取 60 条）
        assert len(second) == 40

        # 两次的 z 值集合不应有交集
        first_z = {s["z"] for s in first}
        second_z = {s["z"] for s in second}
        assert first_z.isdisjoint(second_z)


# ========== TC-SBR-05: refresh 增量加载 ==========

def test_refresh_loads_new_chunks():
    """初始 2 chunk 共 200 条，外部新增 1 chunk 含 80 条，refresh 返回 80"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 初始 2 个 chunk
        chunk1_samples = [_make_sample(state_val=i) for i in range(100)]
        chunk2_samples = [_make_sample(state_val=i) for i in range(100, 200)]

        f1 = _write_chunk(tmpdir, 1, 10, chunk1_samples)
        f2 = _write_chunk(tmpdir, 11, 20, chunk2_samples)

        initial_chunks = [
            {"game_start": 1, "game_end": 10, "sample_count": 100, "file": f1},
            {"game_start": 11, "game_end": 20, "sample_count": 100, "file": f2},
        ]
        _write_manifest(tmpdir, initial_chunks)

        reader = SampleBufferReader(tmpdir)
        assert reader.total_samples() == 200

        # 外部新增第 3 个 chunk
        chunk3_samples = [_make_sample(state_val=i) for i in range(200, 280)]
        f3 = _write_chunk(tmpdir, 21, 30, chunk3_samples)

        updated_chunks = initial_chunks + [
            {"game_start": 21, "game_end": 30, "sample_count": 80, "file": f3},
        ]
        _write_manifest(tmpdir, updated_chunks)

        added = reader.refresh()
        assert added == 80
        assert reader.total_samples() == 280


# ========== TC-SBR-06: 损坏的 chunk 文件 ==========

def test_corrupted_chunk_skipped():
    """一个 chunk JSON 格式损坏时跳过，其他正常加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        good_samples = [_make_sample() for _ in range(50)]
        f_good = _write_chunk(tmpdir, 1, 5, good_samples)

        # 写入损坏的 chunk
        bad_path = os.path.join(tmpdir, "chunk_000006_000010.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("this is not valid json {{{")

        chunks_metadata = [
            {"game_start": 1, "game_end": 5, "sample_count": 50, "file": f_good},
            {"game_start": 6, "game_end": 10, "sample_count": 30, "file": "chunk_000006_000010.json"},
        ]
        _write_manifest(tmpdir, chunks_metadata)

        reader = SampleBufferReader(tmpdir)
        # 应该只加载了好的 chunk
        assert reader.total_samples() == 50
        result = reader.random_sample(30)
        assert len(result) == 30


# ========== TC-SBR-07: manifest 与实际文件不一致 ==========

def test_manifest_mismatch():
    """manifest 记录 3 chunk 但磁盘只有 2 个，按实际存在的加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples1 = [_make_sample() for _ in range(50)]
        samples2 = [_make_sample() for _ in range(30)]

        f1 = _write_chunk(tmpdir, 1, 5, samples1)
        f2 = _write_chunk(tmpdir, 6, 10, samples2)
        # 不创建 f3

        chunks_metadata = [
            {"game_start": 1, "game_end": 5, "sample_count": 50, "file": f1},
            {"game_start": 6, "game_end": 10, "sample_count": 30, "file": f2},
            {"game_start": 11, "game_end": 20, "sample_count": 40, "file": "chunk_000011_000020.json"},
        ]
        _write_manifest(tmpdir, chunks_metadata)

        reader = SampleBufferReader(tmpdir)
        # 只加载实际存在的 2 个 chunk
        assert reader.total_samples() == 80


# ========== 边界条件: refresh 后不放回抽样重置 ==========

def test_refresh_resets_sampling_pool():
    """refresh 后抽样池重置，可再次抽取全部样本"""
    with tempfile.TemporaryDirectory() as tmpdir:
        samples = [_make_sample(state_val=i, z_val=float(i)) for i in range(100)]
        filename = _write_chunk(tmpdir, 1, 10, samples)
        _write_manifest(tmpdir, [{
            "game_start": 1, "game_end": 10,
            "sample_count": 100, "file": filename,
        }])

        reader = SampleBufferReader(tmpdir)

        # 先抽 80 条
        first = reader.random_sample(80)
        assert len(first) == 80

        # refresh 应该重置抽样池
        reader.refresh()
        # 重新可以抽 100 条
        second = reader.random_sample(100)
        assert len(second) == 100
