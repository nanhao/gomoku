"""move_sampler 单元测试

对应设计文档 design/move_sampler_design.md 的用例 1-16。
"""

import random

import pytest

from ai.move_sampler import (
    TempConfig,
    SUPPORTED_TEMP_SIDES,
    TEMP_SIDE_BLACK,
    TEMP_SIDE_WHITE,
    TEMP_SIDE_BOTH,
    TEMP_SIDE_RANDOM,
    temperature_sample,
    should_sample,
    resolve_temp_side,
)


def _positions_line(n):
    """生成 n 个合法坐标 (0,0), (0,1), ..., (0,n-1)。"""
    return [(0, i) for i in range(n)]


# ===========================================================================
# temperature_sample 分布行为 & 边界
# ===========================================================================

def test_temperature_sample_T_eq_1_follows_distribution():
    """用例 1：T=1 时采样频率贴近原分布。"""
    positions = _positions_line(3)
    probs = [0.6, 0.3, 0.1]
    rng = random.Random(20260501)

    counts = [0, 0, 0]
    for _ in range(3000):
        _, y = temperature_sample(positions, probs, temperature=1.0, rng=rng)
        counts[y] += 1

    freqs = [c / 3000.0 for c in counts]
    assert abs(freqs[0] - 0.6) < 0.04
    assert abs(freqs[1] - 0.3) < 0.04
    assert abs(freqs[2] - 0.1) < 0.04


def test_temperature_sample_small_T_behaves_like_argmax():
    """用例 2：T 极小 → 等价 argmax。"""
    positions = _positions_line(3)
    probs = [0.1, 0.7, 0.2]
    rng = random.Random(42)

    for _ in range(200):
        pos = temperature_sample(positions, probs, temperature=0.01, rng=rng)
        assert pos == (0, 1)


def test_temperature_sample_large_T_is_near_uniform():
    """用例 3：T 极大 → 趋近均匀。"""
    positions = _positions_line(3)
    probs = [0.8, 0.15, 0.05]
    rng = random.Random(123)

    counts = [0, 0, 0]
    for _ in range(3000):
        _, y = temperature_sample(positions, probs, temperature=100.0, rng=rng)
        counts[y] += 1

    hi = max(counts)
    lo = min(counts)
    assert lo > 0
    assert hi / lo < 2.0


def test_temperature_sample_invalid_temperature_raises():
    """用例 4：非法温度 → ValueError。"""
    positions = _positions_line(2)
    probs = [0.5, 0.5]
    with pytest.raises(ValueError):
        temperature_sample(positions, probs, temperature=0.0)
    with pytest.raises(ValueError):
        temperature_sample(positions, probs, temperature=-1.0)


def test_temperature_sample_empty_positions_raises():
    """用例 5：空位置列表 → ValueError。"""
    with pytest.raises(ValueError):
        temperature_sample([], [], temperature=1.0)


def test_temperature_sample_length_mismatch_raises():
    """用例 5b：长度不一致 → ValueError。"""
    with pytest.raises(ValueError):
        temperature_sample([(0, 0), (0, 1)], [0.5], temperature=1.0)


def test_temperature_sample_top_k_respected():
    """用例 6：top_k 截断生效。"""
    positions = _positions_line(5)
    probs = [0.4, 0.3, 0.15, 0.10, 0.05]
    rng = random.Random(7)

    seen = set()
    for _ in range(300):
        _, y = temperature_sample(
            positions, probs, temperature=1.0, top_k=2, rng=rng
        )
        seen.add(y)

    assert seen.issubset({0, 1})
    assert 0 in seen and 1 in seen


def test_temperature_sample_all_zero_probs_uniform_fallback():
    """用例 7：全零概率 → 均匀回退。"""
    positions = _positions_line(4)
    probs = [0.0, 0.0, 0.0, 0.0]
    rng = random.Random(0)

    seen = set()
    for _ in range(400):
        _, y = temperature_sample(positions, probs, temperature=1.0, rng=rng)
        seen.add(y)
    assert seen == {0, 1, 2, 3}


def test_temperature_sample_seed_reproducible():
    """用例 8：同 seed 的两个 Random 采样序列完全一致。"""
    positions = _positions_line(5)
    probs = [0.3, 0.25, 0.2, 0.15, 0.1]

    rng1 = random.Random(2026)
    rng2 = random.Random(2026)

    seq1 = [
        temperature_sample(positions, probs, temperature=1.2, rng=rng1)
        for _ in range(50)
    ]
    seq2 = [
        temperature_sample(positions, probs, temperature=1.2, rng=rng2)
        for _ in range(50)
    ]
    assert seq1 == seq2


# ===========================================================================
# TempConfig 字段校验
# ===========================================================================

def test_TempConfig_validates_fields():
    """用例 9：TempConfig 字段校验。"""
    cfg = TempConfig()
    assert cfg.temperature == 1.0
    assert cfg.temp_moves == 12
    assert cfg.temp_side == TEMP_SIDE_BOTH
    assert cfg.top_k is None
    assert cfg.seed is None

    with pytest.raises(ValueError):
        TempConfig(temperature=0.0)
    with pytest.raises(ValueError):
        TempConfig(temperature=-0.5)

    with pytest.raises(ValueError):
        TempConfig(temp_moves=-1)

    with pytest.raises(ValueError):
        TempConfig(temp_side="invalid_side")

    with pytest.raises(ValueError):
        TempConfig(top_k=0)
    with pytest.raises(ValueError):
        TempConfig(top_k=-3)

    TempConfig(temp_moves=0)
    TempConfig(top_k=1)


# ===========================================================================
# should_sample 判定
# ===========================================================================

def test_should_sample_before_threshold_both():
    """用例 10：阈值前 side=both 均为 True。"""
    cfg = TempConfig(temp_moves=5, temp_side=TEMP_SIDE_BOTH)
    for mc in range(5):
        assert should_sample(mc, 1, cfg, TEMP_SIDE_BOTH) is True
        assert should_sample(mc, 2, cfg, TEMP_SIDE_BOTH) is True


def test_should_sample_at_or_after_threshold():
    """用例 11：阈值处及之后 → False。"""
    cfg = TempConfig(temp_moves=5, temp_side=TEMP_SIDE_BOTH)
    assert should_sample(5, 1, cfg, TEMP_SIDE_BOTH) is False
    assert should_sample(10, 2, cfg, TEMP_SIDE_BOTH) is False


def test_should_sample_side_filter():
    """用例 12：side 过滤。"""
    cfg = TempConfig(temp_moves=10, temp_side=TEMP_SIDE_BLACK)
    assert should_sample(0, 1, cfg, TEMP_SIDE_BLACK) is True
    assert should_sample(0, 2, cfg, TEMP_SIDE_BLACK) is False

    cfg2 = TempConfig(temp_moves=10, temp_side=TEMP_SIDE_WHITE)
    assert should_sample(0, 1, cfg2, TEMP_SIDE_WHITE) is False
    assert should_sample(0, 2, cfg2, TEMP_SIDE_WHITE) is True


def test_should_sample_temp_moves_zero_equals_off():
    """用例 13：temp_moves=0 等同关闭。"""
    cfg = TempConfig(temp_moves=0, temp_side=TEMP_SIDE_BOTH)
    for mc in range(20):
        assert should_sample(mc, 1, cfg, TEMP_SIDE_BOTH) is False
        assert should_sample(mc, 2, cfg, TEMP_SIDE_BOTH) is False


# ===========================================================================
# resolve_temp_side 分发
# ===========================================================================

def test_resolve_temp_side_random_splits_evenly():
    """用例 14：random 抽签均衡。"""
    rng = random.Random(2026)
    black_count = 0
    white_count = 0
    total = 400
    for _ in range(total):
        side = resolve_temp_side(TEMP_SIDE_RANDOM, rng=rng)
        assert side in (TEMP_SIDE_BLACK, TEMP_SIDE_WHITE)
        if side == TEMP_SIDE_BLACK:
            black_count += 1
        else:
            white_count += 1
    assert black_count + white_count == total
    assert abs(black_count - white_count) / total < 0.10


def test_resolve_temp_side_passthrough():
    """用例 15：passthrough。"""
    assert resolve_temp_side(TEMP_SIDE_BLACK) == TEMP_SIDE_BLACK
    assert resolve_temp_side(TEMP_SIDE_WHITE) == TEMP_SIDE_WHITE
    assert resolve_temp_side(TEMP_SIDE_BOTH) == TEMP_SIDE_BOTH


def test_resolve_temp_side_invalid_raises():
    """用例 16：非法 side → ValueError。"""
    with pytest.raises(ValueError):
        resolve_temp_side("unknown")


def test_supported_temp_sides_contents():
    """常量内容断言，防止未来误删。"""
    assert set(SUPPORTED_TEMP_SIDES) == {
        TEMP_SIDE_BLACK, TEMP_SIDE_WHITE,
        TEMP_SIDE_BOTH, TEMP_SIDE_RANDOM,
    }
