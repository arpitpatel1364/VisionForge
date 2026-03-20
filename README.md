# ⚡ VisionForge — Model Training Studio

> A sleek, professional web UI for training custom **YOLOv8** object detection models using **Roboflow** datasets — with live log streaming, real-time epoch progress, and one-click cancel.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-0000FF?style=flat-square)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🖼️ Features

- **Paste & Parse** — Drop in your Roboflow code snippet; credentials are auto-extracted
- **Configurable Hyperparameters** — Epochs, image size, model variant (Nano / Small / Medium)
- **Live Log Streaming** — Server-Sent Events stream training output in real time
- **Epoch Progress Bar** — Automatically parsed from training output
- **Cancel Run** — Sends `SIGINT` (Ctrl+C equivalent) to kill the training process cleanly
- **Results Viewer** — Auto-loads `results.png` and best weights path after training

---

## 📁 Project Structure

```
visionforge/
├── app.py                        # Flask backend
├── requirements.txt              # Python dependencies
├── templates/
│   └── index.html                # Frontend UI
└── README.md
```

---

## ⚠️ Before You Run — Configure `app.py`

> `app.py` has separate **GPU** and **CPU** sections that must be configured before starting.
>
> **GPU user** → uncomment the GPU sections, comment out the CPU sections.
> **CPU user** → uncomment the CPU sections, comment out the GPU sections.
>
> Each section in `app.py` is clearly labeled with instructions inside the file.

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/visionforge.git
cd visionforge
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) before the above step.

### 4. Run the server

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🔧 How to Use the Web UI

### Step 1 — Paste Roboflow Snippet

Get your download snippet from the Roboflow dashboard and paste it:

```python
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("your-project")
version = project.version(1)
dataset = version.download("yolov8")
```

Click **Parse Snippet** — the app extracts your credentials automatically.

### Step 2 — Configure Hyperparameters

| Parameter | Options | Default |
|-----------|---------|---------|
| Epochs | 1–500 | 25 |
| Image Size | 640 / 800 / 1024 | 640 |
| Model Variant | Nano / Small / Medium | Nano |

Click **Start Training** to begin.

### Step 3 — Monitor Live Logs

Training output streams in real time with color-coded severity:
- 🟢 Green — success / completion events
- 🔵 Blue — dataset / download info
- 🟡 Amber — warnings
- 🔴 Red — errors

The **epoch progress bar** fills automatically.

### Step 4 — Cancel or Complete

- Click **Cancel Run** at any time to send `SIGINT` and stop training cleanly
- On completion, best weights path and `results.png` are displayed automatically

---

## 🛠️ API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Serves the VisionForge UI |
| `POST` | `/parse_snippet` | Extracts Roboflow credentials from a code snippet |
| `POST` | `/start_training` | Starts a training subprocess |
| `POST` | `/cancel_training` | Sends `SIGINT` to kill the training process |
| `GET` | `/logs_stream` | SSE stream of live training logs |
| `GET` | `/status` | Returns training state and best weights path |
| `GET` | `/results_image` | Serves the latest `results.png` |
| `GET` | `/gpu_status` | Returns detected CUDA GPUs *(GPU users only)* |

---

## 📦 Requirements

- Python 3.10+
- CUDA-compatible GPU recommended (NVIDIA)
- Roboflow account — [sign up free](https://roboflow.com)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ⚡ by VisionForge</p>
