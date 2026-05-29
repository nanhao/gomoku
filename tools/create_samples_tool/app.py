"""Flask 应用 — create_samples 验收工具

3 个 API 端点：
- GET  /api/create-samples/models:        获取可用模型列表
- POST /api/create-samples/execute:        同步执行子进程
- GET  /api/create-samples/games/<id>/steps: 获取步数据

零耦合：不 import 任何 ai/ 包模块。
模型列表通过子进程调用 registry 获取。
"""

import json
import os
import subprocess

from flask import Flask, request, jsonify

from tools.create_samples_tool.executor import build_cli_command, execute
from tools.create_samples_tool.data_parser import (
    scan_round_dir,
    parse_game_steps,
)


def create_app():
    """创建 Flask 应用实例"""
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        static_url_path="/static",
    )

    # 项目根目录（app.py 的父目录的父目录的父目录）
    # tools/create_samples_tool/app.py → gomoku/
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))
    )
    )

    # ------------------------------------------------------------------
    # GET /api/create-samples/models
    # ------------------------------------------------------------------
    @app.route("/api/create-samples/models", methods=["GET"])
    def get_models():
        """获取可用模型列表

        通过子进程调用 ai.registry.list_models_with_full_name()。
        """
        try:
            # 使用列表形式传参，避免 Windows shell 引号问题
            cmd = [
                "python", "-c",
                "import json; from ai.registry import list_models_with_full_name; "
                "print(json.dumps(list_models_with_full_name()))",
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=_project_root,
            )
            stdout_bytes, stderr_bytes = process.communicate()
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")

            if process.returncode != 0:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace")
                return jsonify({
                    "error": f"无法获取模型列表: {stderr_text}"
                }), 500

            # 解析输出为 Python 列表
            models = json.loads(stdout_text.strip())
            return jsonify({"models": models})

        except Exception as e:
            return jsonify({
                "error": f"无法获取模型列表: {str(e)}"
            }), 500

    # ------------------------------------------------------------------
    # POST /api/create-samples/execute
    # ------------------------------------------------------------------
    @app.route("/api/create-samples/execute", methods=["POST"])
    def post_execute():
        """同步执行 create_samples.py

        构造 CLI 命令，Popen + communicate() 阻塞等待，
        完成后一次性返回完整结果（含对局列表）。
        """
        params = request.get_json()
        if not params:
            return jsonify({"error": "请求体为空"}), 400

        # ai_data_dir 路径
        ai_data_dir = os.path.join(_project_root, "ai", "ai_data")

        result = execute(params, project_root=_project_root, ai_data_dir=ai_data_dir)

        # 成功时扫描对局列表
        if result["status"] == "completed" and result.get("round_dir"):
            try:
                games = scan_round_dir(result["round_dir"])
                result["games"] = games
                result["output_path"] = params.get("output")
            except Exception:
                result["games"] = []

        return jsonify(result)

    # ------------------------------------------------------------------
    # GET /api/create-samples/games/<game_id>/steps
    # ------------------------------------------------------------------
    @app.route(
        "/api/create-samples/games/<int:game_id>/steps", methods=["GET"]
    )
    def get_game_steps(game_id):
        """获取指定对局的步数据

        参数：round_dir 和 sample_path（query string）。
        """
        round_dir = request.args.get("round_dir")
        sample_path = request.args.get("sample_path")

        if not round_dir:
            return jsonify({
                "error": "缺少 round_dir 参数"
            }), 400
        if not sample_path:
            return jsonify({
                "error": "缺少 sample_path 参数"
            }), 400

        try:
            result = parse_game_steps(round_dir, game_id, sample_path=sample_path)
            return jsonify(result)
        except FileNotFoundError:
            return jsonify({
                "error": f"对局 {game_id} 不存在"
            }), 404
        except json.JSONDecodeError:
            return jsonify({
                "error": f"对局 {game_id} 数据损坏"
            }), 500
        except Exception as e:
            return jsonify({
                "error": str(e)
            }), 500

    # ------------------------------------------------------------------
    # 主页面
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        """返回主页面"""
        return app.send_static_file("index.html")

    return app


# ------------------------------------------------------------------
# 启动入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=True)