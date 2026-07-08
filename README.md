# YOLO Live (Rust)

Real-time object detection for **Apple Silicon (M4)** — written entirely in **Rust**.

- **Model:** YOLO11n (ONNX)
- **Acceleration:** ONNX Runtime **CoreML** execution provider (Neural Engine / GPU / CPU)
- **Server:** Axum + WebSockets
- **Modes:** Live webcam · Upload video
- **UI:** Video stage + **labels panel on the right**

## Requirements

- macOS on Apple Silicon (M1–M4 recommended)
- [Rust](https://rustup.rs/) 1.75+
- [ffmpeg](https://ffmpeg.org/) (video upload mode) — `brew install ffmpeg`

## Quick start

```bash
git clone https://github.com/SametEmin/yolo-live.git
cd yolo-live
cargo run --release
```

Open **http://127.0.0.1:8000**

On first run the app loads `models/yolo11n.onnx` (auto-downloads if missing).

## Features

| Mode | Description |
|------|-------------|
| **Live Camera** | Browser webcam frames → Rust YOLO inference → annotated stream |
| **Upload Video** | MP4/MOV/WebM/AVI → ffmpeg decode → per-frame detection → optional annotated export |

Right-side panel shows:

- Grouped **labels** with counts and average confidence  
- Per-instance list with confidence scores  

## Project layout

```
yolo-live/
├── Cargo.toml
├── src/
│   ├── main.rs        # Axum HTTP/WebSocket server
│   ├── detector.rs    # YOLO11 ONNX + CoreML inference
│   └── coco.rs        # COCO-80 class names
├── models/            # yolo11n.onnx
├── static/            # UI (HTML/CSS/JS)
├── assets/            # Embedded font for box labels
├── uploads/           # Temporary uploads
└── outputs/           # Annotated videos
```

## Configuration

- Confidence slider in the UI (default `0.35`)
- Model path: `models/yolo11n.onnx`
- Bind address: `127.0.0.1:8000` (see `src/main.rs`)

## License

MIT

## Multi-threaded pipeline

Live detection runs a **4-stage OS-thread pipeline**:

| Thread | Stage |
|--------|--------|
| `yolo-capture` | JPEG decode (camera / network frames) |
| `yolo-preprocess` | Letterbox + NCHW tensor |
| `yolo-inference` | YOLO11 ONNX (CoreML) |
| `yolo-render` | Draw boxes + JPEG encode |

Upload-video processing still uses the sequential path for simplicity; live uses the pipeline.

### FPS benchmark (before vs after)

In the UI footer click **Compare FPS (before/after MT)**, or:

```bash
curl "http://127.0.0.1:8000/api/benchmark?frames=48"
```

`before` = single-thread sequential `predict_jpeg`  
`after` = 4-thread pipeline throughput  
