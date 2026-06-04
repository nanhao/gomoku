"""Flask app API 端点集成测试

测试 3 个 API 端点：
- GET /api/create-samples/models
- POST /api/create-samples/execute
- GET /api/create-samples/games/<game_id>/steps

覆盖 backend_api_design.md 中 T1-T16 测试点。
使用 mock 替代真实子进程调用，避免依赖 GPU/模型。
"""

import json
import os
import tempfile
from unittest import mock

import pytest

from tools.create_samples_tool.app import create_app


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def app():
    """创建测试 Flask 应用"""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def round_dir_with_data(tmp_path):
    """创建带测试数据的 round_dir"""
    rd = os.path.join(tmp_path, "test_round")
    os.makedirs(rd)
    # 创建 1 局正常数据
    game_dir = os.path.join(rd, "1")
    os.makedirs(game_dir)
    record = {
        "history": [
            {
                "step_id": "sid-1",
                "input": [0] * 225 + [1],
                "output": [0.01] * 225,
                "position": [7, 7],
                "strategy": "sampling",
                "sampling_probs": [0.02] * 225,
                "temperature": 1.0,
            },
        ],
        "winner": 1,
        "end_board": [0] * 226,
    }
    with open(os.path.join(game_dir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f)

    # 创建样本文件
    samples = [
        {
            "step_id": "sid-1",
            "input": [0] * 226,
            "output": [0.5] * 225,
            "mask": [1] * 225,
        }
    ]
    sample_path = os.path.join(tmp_path, "samples.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f)

    return rd, sample_path


# ------------------------------------------------------------------
# GET /api/create-samples/models
# ------------------------------------------------------------------

class TestGetModels:
    """测试模型列表端点"""

    def test_returns_model_list(self, client):
        """T1: 返回非空模型列表"""
        resp = client.get("/api/create-samples/models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)
        # 至少应包含 "random"
        assert "random" in data["models"]


# ------------------------------------------------------------------
# POST /api/create-samples/execute
# ------------------------------------------------------------------

class TestPostExecute:
    """测试执行端点"""

    def test_execute_returns_result(self, client):
        """T3: 执行返回完整结果结构（mock 子进程，避免真实执行）"""
        mock_result = {
            "status": "completed",
            "command": "python ai/create_samples.py --model random --output test.json --games 1",
            "output": "mock output",
            "returncode": 0,
            "round_dir": "ai/ai_data/mock_round",
            "output_path": "test.json",
        }
        params = {
            "model": "random",
            "output": "test.json",
            "games": 1,
        }
        with mock.patch(
            "tools.create_samples_tool.app.execute",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/create-samples/execute",
                json=params,
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "completed"
        assert data["command"] is not None
        assert data["output"] is not None
        assert data["returncode"] == 0

    def test_execute_error_result(self, client):
        """执行返回错误状态时 API 仍正确传递结果"""
        mock_result = {
            "status": "error",
            "command": "python ai/create_samples.py --model bad --output x.json",
            "output": "Model not found",
            "returncode": 1,
            "round_dir": None,
            "output_path": None,
            "games": None,
        }
        params = {"model": "bad_model", "output": "x.json"}
        with mock.patch(
            "tools.create_samples_tool.app.execute",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/create-samples/execute",
                json=params,
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["returncode"] == 1


# ------------------------------------------------------------------
# GET /api/create-samples/games/<game_id>/steps
# ------------------------------------------------------------------

class TestGetGameSteps:
    """测试步数据端点"""

    def test_steps_returns_data(self, client, round_dir_with_data):
        """T10: 步数据完整"""
        rd, sample_path = round_dir_with_data
        resp = client.get(
            f"/api/create-samples/games/1/steps"
            f"?round_dir={rd}&sample_path={sample_path}"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["game_id"] == 1
        assert data["winner"] == 1
        assert data["winner_display"] == "黑胜"
        assert data["is_draw"] is False
        assert len(data["steps"]) == 1
        # 步数据字段齐全
        step = data["steps"][0]
        assert step["step_index"] == 0
        assert step["player"] == 1
        assert step["move_position"] == [7, 7]
        assert step["board"] is not None
        assert step["raw_output"] is not None
        assert step["sampling_probs"] is not None
        assert step["strategy"] == "sampling"
        assert step["sample_output"] is not None  # step_id 匹配成功

    def test_game_id_not_found(self, client, round_dir_with_data):
        """T14: 不存在的 game_id → 404"""
        rd, sample_path = round_dir_with_data
        resp = client.get(
            f"/api/create-samples/games/999/steps"
            f"?round_dir={rd}&sample_path={sample_path}"
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    def test_missing_round_dir(self, client):
        """T15: 缺少 round_dir 参数 → 400"""
        resp = client.get("/api/create-samples/games/1/steps")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_missing_sample_path(self, client, round_dir_with_data):
        """缺少 sample_path 参数 → 400"""
        rd, _ = round_dir_with_data
        resp = client.get(
            f"/api/create-samples/games/1/steps?round_dir={rd}"
        )
        assert resp.status_code == 400

    def test_corrupted_record(self, client, tmp_path):
        """T16: record.json 损坏 → 500"""
        rd = os.path.join(tmp_path, "corrupt_round")
        os.makedirs(rd)
        game_dir = os.path.join(rd, "1")
        os.makedirs(game_dir)
        with open(os.path.join(game_dir, "record.json"), "w") as f:
            f.write("BAD JSON")
        sample_path = os.path.join(tmp_path, "samples.json")
        with open(sample_path, "w") as f:
            json.dump([], f)

        resp = client.get(
            f"/api/create-samples/games/1/steps"
            f"?round_dir={rd}&sample_path={sample_path}"
        )
        assert resp.status_code == 500