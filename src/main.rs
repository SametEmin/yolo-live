//! YOLO Live — pure Rust real-time object detection (CoreML on Apple Silicon).

mod coco;
mod detector;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::extract::multipart::Multipart;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, State};
use axum::response::{Html, IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use detector::ObjectDetector;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;
use tracing::{error, info};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    detector: Arc<Mutex<ObjectDetector>>,
    root: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "yolo_live=info,tower_http=info".into()),
        )
        .init();

    let root = std::env::current_dir()?;
    std::fs::create_dir_all(root.join("uploads"))?;
    std::fs::create_dir_all(root.join("outputs"))?;
    std::fs::create_dir_all(root.join("models"))?;

    let model_path = root.join("models/yolo11n.onnx");
    info!("Initializing YOLO detector…");
    let detector = tokio::task::spawn_blocking(move || ObjectDetector::load(model_path))
        .await??;

    info!(
        "Ready · device={} · model={}",
        detector.device(),
        detector.model_name()
    );

    let state = AppState {
        detector: Arc::new(Mutex::new(detector)),
        root: root.clone(),
    };

    let static_dir = root.join("static");
    let app = Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/confidence", post(set_confidence))
        .route(
            "/api/upload",
            post(upload_video).layer(DefaultBodyLimit::max(500 * 1024 * 1024)),
        )
        .route("/api/download/{job_id}", get(download_annotated))
        .route("/ws/detect", get(ws_detect_upgrade))
        .route("/ws/video/{job_id}", get(ws_video_upgrade))
        .nest_service("/static", ServeDir::new(static_dir))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8000));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("Listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let det = state.detector.lock().await;
    Json(json!({
        "ok": true,
        "device": det.device(),
        "model": det.model_name(),
        "confidence": det.confidence(),
        "language": "rust"
    }))
}

#[derive(Deserialize)]
struct ConfBody {
    confidence: f32,
}

async fn set_confidence(
    State(state): State<AppState>,
    Json(body): axum::Json<ConfBody>,
) -> impl IntoResponse {
    let mut det = state.detector.lock().await;
    det.set_confidence(body.confidence);
    Json(json!({ "confidence": det.confidence() }))
}

async fn upload_video(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Note: request body limit is raised on this route (500 MiB).
    // Without that, Axum's default 2 MiB limit surfaces as:
    // "Error parsing `multipart/form-data` request".
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                error!("multipart field error: {e}");
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    Json(json!({
                        "error": format!(
                            "Failed to read upload ({e}). Large videos need the raised body limit; try again after restarting the server."
                        )
                    })),
                )
                    .into_response();
            }
        };

        let name = field.name().unwrap_or("").to_string();
        if name != "file" {
            continue;
        }

        let filename = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "video.mp4".into());
        let ext = Path::new(&filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp4")
            .to_lowercase();
        let allowed = ["mp4", "mov", "avi", "mkv", "webm", "m4v"];
        if !allowed.contains(&ext.as_str()) {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Unsupported format: .{ext}") })),
            )
                .into_response();
        }

        let job_id = Uuid::new_v4().simple().to_string()[..12].to_string();
        let dest = state.root.join("uploads").join(format!("{job_id}.{ext}"));

        // Stream chunks to disk (avoids holding whole video in RAM).
        let mut file = match tokio::fs::File::create(&dest).await {
            Ok(f) => f,
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Cannot create upload file: {e}") })),
                )
                    .into_response();
            }
        };

        let mut total: u64 = 0;
        let mut field = field;
        loop {
            match field.chunk().await {
                Ok(Some(chunk)) => {
                    total += chunk.len() as u64;
                    if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await {
                        let _ = tokio::fs::remove_file(&dest).await;
                        return (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": format!("Write failed: {e}") })),
                        )
                            .into_response();
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&dest).await;
                    error!("chunk read error: {e}");
                    return (
                        axum::http::StatusCode::BAD_REQUEST,
                        Json(json!({
                            "error": format!(
                                "Upload interrupted ({e}). If the file is large, restart the updated server and retry."
                            )
                        })),
                    )
                        .into_response();
                }
            }
        }
        if let Err(e) = tokio::io::AsyncWriteExt::flush(&mut file).await {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response();
        }

        if total == 0 {
            let _ = tokio::fs::remove_file(&dest).await;
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Empty file" })),
            )
                .into_response();
        }

        info!("Saved upload {filename} → {} ({total} bytes)", dest.display());

        let (fps, frames, width, height) = probe_video(&dest);
        let duration_s = if fps > 0.0 {
            Some(((frames as f64) / fps * 100.0).round() / 100.0)
        } else {
            None
        };

        return Json(json!({
            "job_id": job_id,
            "filename": filename,
            "path": dest.file_name().and_then(|s| s.to_str()),
            "bytes": total,
            "fps": fps,
            "frames": frames,
            "width": width,
            "height": height,
            "duration_s": duration_s
        }))
        .into_response();
    }

    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(json!({ "error": "No file field in form (expected name=\"file\")" })),
    )
        .into_response()
}

async fn download_annotated(
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> impl IntoResponse {
    let path = state
        .root
        .join("outputs")
        .join(format!("{job_id}_annotated.mp4"));
    if !path.exists() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({ "error": "Not ready" })),
        )
            .into_response();
    }
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let disp = format!("attachment; filename=\"{job_id}_annotated.mp4\"");
            (
                axum::http::StatusCode::OK,
                [
                    (axum::http::header::CONTENT_TYPE, "video/mp4".to_string()),
                    (axum::http::header::CONTENT_DISPOSITION, disp),
                ],
                bytes,
            )
                .into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn ws_detect_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_live_ws(socket, state))
}

struct LiveRecording {
    job_id: String,
    frames_dir: PathBuf,
    frame_count: u64,
    /// Estimated capture FPS for ffmpeg (from client or EMA).
    fps: f32,
}

async fn handle_live_ws(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Auto-record live session so user can download annotated video on stop.
    let job_id = Uuid::new_v4().simple().to_string()[..12].to_string();
    let frames_dir = state.root.join("outputs").join(format!("{job_id}_live_frames"));
    let _ = std::fs::create_dir_all(&frames_dir);
    let mut recording = LiveRecording {
        job_id: job_id.clone(),
        frames_dir,
        frame_count: 0,
        fps: 12.0,
    };
    let mut recording_active = true;

    let _ = sender
        .send(Message::Text(
            json!({
                "type": "recording_started",
                "job_id": recording.job_id,
                "message": "Recording annotated live frames"
            })
            .to_string()
            .into(),
        ))
        .await;

    while let Some(Ok(msg)) = receiver.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => continue,
        };

        match v.get("type").and_then(|t| t.as_str()) {
            Some("ping") => {
                let _ = sender
                    .send(Message::Text(r#"{"type":"pong"}"#.into()))
                    .await;
            }
            Some("config") => {
                if let Some(c) = v.get("confidence").and_then(|c| c.as_f64()) {
                    state.detector.lock().await.set_confidence(c as f32);
                }
                let conf = state.detector.lock().await.confidence();
                let _ = sender
                    .send(Message::Text(
                        json!({"type":"config_ok","confidence": conf}).to_string().into(),
                    ))
                    .await;
            }
            Some("stop") | Some("record_stop") => {
                // Finalize annotated MP4 and notify client before socket ends.
                if recording_active {
                    let out = finalize_live_recording(&state, &recording).await;
                    let _ = sender
                        .send(Message::Text(
                            json!({
                                "type": "recording_ready",
                                "job_id": recording.job_id,
                                "frames": recording.frame_count,
                                "output": out,
                            })
                            .to_string()
                            .into(),
                        ))
                        .await;
                    recording_active = false;
                }
                break;
            }
            Some("frame") | None => {
                let Some(b64) = v.get("frame").and_then(|f| f.as_str()) else {
                    continue;
                };
                let b64 = b64.split_once(',').map(|(_, d)| d).unwrap_or(b64);
                let Ok(raw) = B64.decode(b64) else {
                    continue;
                };

                let det = state.detector.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut d = det.blocking_lock();
                    d.predict_jpeg(&raw)
                })
                .await;

                match result {
                    Ok(Ok((jpeg, meta))) => {
                        if recording_active {
                            recording.frame_count += 1;
                            if meta.fps > 1.0 {
                                // Blend reported end-to-end fps for export framerate
                                recording.fps = 0.8 * recording.fps + 0.2 * meta.fps.max(5.0);
                            }
                            let fp = recording
                                .frames_dir
                                .join(format!("frame_{:06}.jpg", recording.frame_count));
                            let _ = std::fs::write(&fp, &jpeg);
                        }

                        let frame = format!("data:image/jpeg;base64,{}", B64.encode(&jpeg));
                        let payload = json!({
                            "type": "result",
                            "frame": frame,
                            "detections": meta.detections,
                            "summary": meta.summary,
                            "fps": meta.fps,
                            "inference_ms": meta.inference_ms,
                            "device": meta.device,
                            "count": meta.count,
                            "recording": recording_active,
                            "recorded_frames": recording.frame_count,
                        });
                        if sender
                            .send(Message::Text(payload.to_string().into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(Err(e)) => {
                        error!("inference error: {e:#}");
                        let _ = sender
                            .send(Message::Text(
                                json!({"type":"error","message": e.to_string()})
                                    .to_string()
                                    .into(),
                            ))
                            .await;
                    }
                    Err(e) => error!("join error: {e}"),
                }
            }
            _ => {}
        }
    }

    // If client disconnected without stop, still try to finalize.
    if recording_active && recording.frame_count > 0 {
        let out = finalize_live_recording(&state, &recording).await;
        let _ = sender
            .send(Message::Text(
                json!({
                    "type": "recording_ready",
                    "job_id": recording.job_id,
                    "frames": recording.frame_count,
                    "output": out,
                })
                .to_string()
                .into(),
            ))
            .await;
    } else if recording.frame_count == 0 {
        let _ = std::fs::remove_dir_all(&recording.frames_dir);
    }
}

async fn finalize_live_recording(state: &AppState, rec: &LiveRecording) -> Option<String> {
    if rec.frame_count == 0 {
        let _ = std::fs::remove_dir_all(&rec.frames_dir);
        return None;
    }

    let out_video = state
        .root
        .join("outputs")
        .join(format!("{}_annotated.mp4", rec.job_id));
    let pattern = rec.frames_dir.join("frame_%06d.jpg");
    let fps = rec.fps.clamp(5.0, 30.0);

    let job_id = rec.job_id.clone();
    let frames_dir = rec.frames_dir.clone();
    let out_clone = out_video.clone();
    let pattern_s = pattern.to_string_lossy().to_string();
    let out_s = out_video.to_string_lossy().to_string();

    let ok = tokio::task::spawn_blocking(move || {
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                &format!("{fps:.2}"),
                "-i",
                &pattern_s,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                &out_s,
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let _ = std::fs::remove_dir_all(&frames_dir);
        status && out_clone.exists()
    })
    .await
    .unwrap_or(false);

    if ok {
        info!(
            "Live annotated video ready: {} ({} frames @ {:.1} fps)",
            out_video.display(),
            rec.frame_count,
            fps
        );
        Some(format!("/api/download/{job_id}"))
    } else {
        error!("Failed to assemble live annotated video for {job_id}");
        None
    }
}

async fn ws_video_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    AxumPath(job_id): AxumPath<String>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_video_ws(socket, state, job_id))
}

async fn handle_video_ws(socket: WebSocket, state: AppState, job_id: String) {
    let (mut sender, mut receiver) = socket.split();

    let uploads = state.root.join("uploads");
    let video_path = match find_upload(&uploads, &job_id) {
        Some(p) => p,
        None => {
            let _ = sender
                .send(Message::Text(
                    json!({"type":"error","message":"Video not found"})
                        .to_string()
                        .into(),
                ))
                .await;
            return;
        }
    };

    let (src_fps, total_frames, _, _) = probe_video(&video_path);
    let target_fps = 15.0_f64;
    let skip = if src_fps > 0.0 {
        ((src_fps / target_fps).round() as i64).max(1)
    } else {
        1
    };

    let _ = sender
        .send(Message::Text(
            json!({
                "type": "start",
                "total_frames": total_frames,
                "src_fps": src_fps,
                "skip": skip
            })
            .to_string()
            .into(),
        ))
        .await;

    // Extract JPEG frames via ffmpeg pipe
    let vf = if skip > 1 {
        format!("fps={target_fps}")
    } else {
        "null".to_string()
    };

    let mut child = match Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path.to_str().unwrap_or(""),
            "-vf",
            &vf,
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "5",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = sender
                .send(Message::Text(
                    json!({"type":"error","message": format!("ffmpeg failed: {e} (is ffmpeg installed?)")})
                        .to_string()
                        .into(),
                ))
                .await;
            return;
        }
    };

    let mut stdout = child.stdout.take().expect("stdout");
    let mut jpeg_buf: Vec<u8> = Vec::new();
    let mut read_buf = [0u8; 65536];
    let mut frame_index: u64 = 0;
    let mut processed: u64 = 0;
    let mut stop = false;

    // Annotated frames dir for optional re-encode
    let frames_dir = state.root.join("outputs").join(format!("{job_id}_frames"));
    let _ = std::fs::create_dir_all(&frames_dir);

    use std::io::Read;
    loop {
        // Non-blocking-ish client control: try drain stop messages
        while let Ok(Some(Ok(msg))) =
            tokio::time::timeout(std::time::Duration::from_millis(0), receiver.next()).await
        {
            if let Message::Text(t) = msg {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                    match v.get("type").and_then(|x| x.as_str()) {
                        Some("stop") => stop = true,
                        Some("config") => {
                            if let Some(c) = v.get("confidence").and_then(|c| c.as_f64()) {
                                state.detector.lock().await.set_confidence(c as f32);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        if stop {
            break;
        }

        let n = match stdout.read(&mut read_buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        jpeg_buf.extend_from_slice(&read_buf[..n]);

        // Split MJPEG stream on SOI/EOI markers
        while let Some((start, end)) = find_jpeg(&jpeg_buf) {
            let jpeg = jpeg_buf[start..=end].to_vec();
            jpeg_buf.drain(..=end);
            frame_index += 1;

            let det = state.detector.clone();
            let jpeg_clone = jpeg.clone();
            let result = tokio::task::spawn_blocking(move || {
                let mut d = det.blocking_lock();
                d.predict_jpeg(&jpeg_clone)
            })
            .await;

            match result {
                Ok(Ok((out_jpeg, meta))) => {
                    processed += 1;
                    // Save annotated frame for video assembly
                    let fp = frames_dir.join(format!("frame_{:06}.jpg", processed));
                    let _ = std::fs::write(&fp, &out_jpeg);

                    let progress = if total_frames > 0 {
                        // approximate progress by processed * skip / total
                        ((processed as f64 * skip as f64) / total_frames as f64).min(1.0)
                    } else {
                        0.0
                    };

                    let frame = format!("data:image/jpeg;base64,{}", B64.encode(&out_jpeg));
                    let payload = json!({
                        "type": "result",
                        "frame": frame,
                        "detections": meta.detections,
                        "summary": meta.summary,
                        "fps": meta.fps,
                        "inference_ms": meta.inference_ms,
                        "device": meta.device,
                        "count": meta.count,
                        "frame_index": frame_index,
                        "progress": (progress * 10000.0).round() / 10000.0,
                        "processed": processed
                    });
                    if sender
                        .send(Message::Text(payload.to_string().into()))
                        .await
                        .is_err()
                    {
                        stop = true;
                        break;
                    }
                }
                Ok(Err(e)) => {
                    error!("frame inference: {e:#}");
                }
                Err(e) => error!("join: {e}"),
            }
        }
    }

    let _ = child.kill();
    let _ = child.wait();

    // Assemble annotated video if we have frames
    let out_video = state
        .root
        .join("outputs")
        .join(format!("{job_id}_annotated.mp4"));
    let mut output_url = None;
    if processed > 0 {
        let pattern = frames_dir.join("frame_%06d.jpg");
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                &format!("{}", target_fps.min(src_fps.max(1.0))),
                "-i",
                pattern.to_str().unwrap_or(""),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                out_video.to_str().unwrap_or(""),
            ])
            .status();
        if status.map(|s| s.success()).unwrap_or(false) {
            output_url = Some(format!("/api/download/{job_id}"));
        }
        let _ = std::fs::remove_dir_all(&frames_dir);
    }

    let _ = sender
        .send(Message::Text(
            json!({
                "type": "done",
                "processed": processed,
                "output": output_url
            })
            .to_string()
            .into(),
        ))
        .await;
}

fn find_upload(dir: &Path, job_id: &str) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with(job_id) {
            return Some(e.path());
        }
    }
    None
}

fn find_jpeg(buf: &[u8]) -> Option<(usize, usize)> {
    let start = buf.windows(2).position(|w| w == [0xFF, 0xD8])?;
    let rest = &buf[start + 2..];
    let end_rel = rest.windows(2).position(|w| w == [0xFF, 0xD9])?;
    let end = start + 2 + end_rel + 1;
    Some((start, end))
}

fn probe_video(path: &Path) -> (f64, u64, u32, u32) {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            path.to_str().unwrap_or(""),
        ])
        .output();

    let Ok(out) = output else {
        return (0.0, 0, 0, 0);
    };
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or(json!({}));
    let stream = v
        .get("streams")
        .and_then(|s| s.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(json!({}));

    let width = stream.get("width").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let height = stream.get("height").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let frames = stream
        .get("nb_frames")
        .and_then(|x| x.as_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0u64);
    let fps = stream
        .get("r_frame_rate")
        .and_then(|x| x.as_str())
        .and_then(|s| {
            let mut parts = s.split('/');
            let n: f64 = parts.next()?.parse().ok()?;
            let d: f64 = parts.next().unwrap_or("1").parse().ok()?;
            if d == 0.0 {
                None
            } else {
                Some(n / d)
            }
        })
        .unwrap_or(0.0);

    (fps, frames, width, height)
}
