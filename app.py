"""
VisionForge — Model Training Studio
Flask backend: dataset download, YOLOv8 training, live log streaming, cancel support.

Runs on ANY machine — automatically uses GPU if available, falls back to CPU.
No manual configuration needed.
"""

import os
import re
import json
import signal
import subprocess
import threading
import queue
import glob

from flask import Flask, request, jsonify, Response, send_file, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Global training state ─────────────────────────────────────────────────────
training_process: subprocess.Popen | None = None
log_queue: queue.Queue = queue.Queue()
training_status = {
    "running": False,
    "weights_path": None,
    "error": None,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def enqueue_log(line: str):
    log_queue.put(line)


def find_best_weights() -> str | None:
    candidates = sorted(
        glob.glob("runs/detect/train*/weights/best.pt"),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_results_image() -> str | None:
    candidates = sorted(
        glob.glob("runs/detect/train*/results.png"),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parse_snippet", methods=["POST"])
def parse_snippet():
    data = request.get_json(force=True)
    snippet: str = data.get("snippet", "")

    api_key   = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', snippet)
    workspace = re.search(r'\.workspace\(["\']([^"\']+)["\']\)', snippet)
    project   = re.search(r'\.project\(["\']([^"\']+)["\']\)', snippet)
    version   = re.search(r'\.version\((\d+)\)', snippet)

    missing = []
    if not api_key:   missing.append("api_key")
    if not workspace: missing.append("workspace")
    if not project:   missing.append("project name")
    if not version:   missing.append("version number")

    if missing:
        return jsonify({"success": False, "error": f"Could not parse: {', '.join(missing)}"}), 200

    return jsonify({
        "success":      True,
        "api_key":      api_key.group(1),
        "workspace":    workspace.group(1),
        "project_name": project.group(1),
        "version_num":  int(version.group(1)),
    })


@app.route("/start_training", methods=["POST"])
def start_training():
    global training_process, training_status

    if training_status["running"]:
        return jsonify({"success": False, "error": "Training already in progress."}), 200

    cfg = request.get_json(force=True)
    required = ["api_key", "workspace", "project_name", "version_num", "epochs", "image_size", "model_variant"]
    for key in required:
        if not cfg.get(key):
            return jsonify({"success": False, "error": f"Missing field: {key}"}), 200

    training_status = {"running": True, "weights_path": None, "error": None}

    while not log_queue.empty():
        try: log_queue.get_nowait()
        except queue.Empty: break

    thread = threading.Thread(target=_run_training, args=(cfg,), daemon=True)
    thread.start()

    return jsonify({"success": True})


def _run_training(cfg: dict):
    global training_process, training_status

    script = _build_training_script(cfg)

    try:
        enqueue_log("🔍 Downloading dataset from Roboflow…")
        training_process = subprocess.Popen(
            ["python", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        for line in training_process.stdout:
            line = line.rstrip()
            if line:
                enqueue_log(line)

        training_process.wait()

        if training_process.returncode == 0:
            weights = find_best_weights()
            training_status["weights_path"] = weights
            enqueue_log("✅ Training complete!")
            if weights:
                enqueue_log(f"🚀 Best weights saved at: {weights}")
        else:
            enqueue_log(f"❌ Process exited with code {training_process.returncode}")
            training_status["error"] = f"Exit code {training_process.returncode}"

    except Exception as exc:
        enqueue_log(f"❌ Exception: {exc}")
        training_status["error"] = str(exc)
    finally:
        training_status["running"] = False
        training_process = None
        enqueue_log("__DONE__")


def _build_training_script(cfg: dict) -> str:
    return f"""
import subprocess, sys

# ── Install dependencies ──────────────────────────────────────────────────────
for pkg in ["roboflow", "ultralytics"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

# ── Auto-detect device (GPU if available, else CPU) ───────────────────────────
import torch

if torch.cuda.is_available():
    device = "0"
    amp    = True
    print("=" * 50)
    print(f"✅ GPU detected: {{torch.cuda.get_device_name(0)}}")
    print(f"   VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
    print(f"   Training on: GPU")
    print("=" * 50)
else:
    device = "cpu"
    amp    = False
    print("=" * 50)
    print("ℹ️  No GPU detected — training on CPU")
    print(f"   PyTorch: {{torch.__version__}}")
    print("=" * 50)

# ── Download dataset ──────────────────────────────────────────────────────────
from roboflow import Roboflow
from ultralytics import YOLO

print("📦 Connecting to Roboflow…")
rf      = Roboflow(api_key="{cfg['api_key']}")
project = rf.workspace("{cfg['workspace']}").project("{cfg['project_name']}")
version = project.version({cfg['version_num']})
dataset = version.download("yolov8")
print(f"✅ Dataset downloaded to: {{dataset.location}}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("🏋️  Starting YOLOv8 training…")
model   = YOLO("{cfg['model_variant']}")
results = model.train(
    data=f"{{dataset.location}}/data.yaml",
    epochs={cfg['epochs']},
    imgsz={cfg['image_size']},
    device=device,
    amp=amp,
    verbose=True,
)
print("✅ Training finished.")
"""


@app.route("/cancel_training", methods=["POST"])
def cancel_training():
    global training_process, training_status

    if training_process is None:
        return jsonify({"success": False, "error": "No training process running."}), 200

    try:
        os.killpg(os.getpgid(training_process.pid), signal.SIGINT)
        training_status["running"] = False
        enqueue_log("⚠ Training cancelled by user (SIGINT sent).")
        enqueue_log("__DONE__")
        return jsonify({"success": True})
    except ProcessLookupError:
        return jsonify({"success": False, "error": "Process already finished."}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/logs_stream")
def logs_stream():
    """Server-Sent Events endpoint — streams log lines to the frontend."""
    def generate():
        while True:
            try:
                line = log_queue.get(timeout=30)
                if line == "__DONE__":
                    yield "data: \"__DONE__\"\n\n"
                    break
                yield f"data: {json.dumps(line)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/status")
def status():
    return jsonify({
        "running":      training_status["running"],
        "weights_path": training_status["weights_path"],
        "error":        training_status["error"],
    })


@app.route("/results_image")
def results_image():
    path = find_results_image()
    if path and os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "", 404


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
