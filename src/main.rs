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
    // Sanitize job id (only hex from uuid simple)
    if !job_id.chars().all(|c| c.is_ascii_hexdigit()) || job_id.is_empty() || job_id.len() > 32 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Invalid job id" })),
        )
            .into_response();
    }

    let path = state
        .root
        .join("outputs")
        .join(format!("{job_id}_annotated.mp4"));
    if !path.exists() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({ "error": "Annotated video not found. Processing may have failed." })),
        )
            .into_response();
    }

    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let filename = format!("yolo_annotated_{job_id}.mp4");
            let disp = format!("attachment; filename=\"{filename}\"");
            (
                axum::http::StatusCode::OK,
                [
                    (
                        axum::http::header::CONTENT_TYPE,
                        "video/mp4".to_string(),
                    ),
                    (
                        axum::http::header::CONTENT_DISPOSITION,
                        disp,
                    ),
                    (
                        axum::http::header::CACHE_CONTROL,
                        "no-store".to_string(),
                    ),
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

    let meta = probe_video_meta(&video_path);
    let process_fps = 15.0_f64.min(if meta.fps > 1.0 { meta.fps } else { 15.0 });
    let duration_s = if meta.duration_s > 0.0 {
        meta.duration_s
    } else if meta.frames > 0 && meta.fps > 0.0 {
        meta.frames as f64 / meta.fps
    } else {
        0.0
    };
    // Expected *processed* frames at process_fps (what the progress bar should track).
    let expected_frames = if duration_s > 0.0 {
        (duration_s * process_fps).ceil().max(1.0) as u64
    } else if meta.frames > 0 {
        // Fallback: scale source frame count to process rate
        let ratio = if meta.fps > 0.0 {
            (process_fps / meta.fps).clamp(0.05, 1.0)
        } else {
            1.0
        };
        ((meta.frames as f64) * ratio).ceil().max(1.0) as u64
    } else {
        1
    };

    let frames_dir = state.root.join("outputs").join(format!("{job_id}_frames"));
    // Fresh session: clear any leftover frames for this job id.
    let _ = std::fs::remove_dir_all(&frames_dir);
    let _ = std::fs::create_dir_all(&frames_dir);
    let mut processed: u64 = 0;

    let _ = sender
        .send(Message::Text(
            json!({
                "type": "start",
                "total_source_frames": meta.frames,
                "src_fps": meta.fps,
                "process_fps": process_fps,
                "duration_s": duration_s,
                "expected_frames": expected_frames,
                "processed": processed,
                "progress": progress_value(processed, expected_frames, false),
            })
            .to_string()
            .into(),
        ))
        .await;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Ctrl {
        Run,
        Pause,
        Stop,
    }

    let mut finished_naturally = false;
    let mut last_export_url: Option<String> = None;

    'session: loop {
        // Seek to resume point (seconds into source video).
        let start_sec = if process_fps > 0.0 {
            processed as f64 / process_fps
        } else {
            0.0
        };

        // If already past expected end, finish.
        if expected_frames > 0 && processed >= expected_frames {
            finished_naturally = true;
            break 'session;
        }

        let mut child = match spawn_frame_extractor(&video_path, process_fps, start_sec) {
            Ok(c) => c,
            Err(e) => {
                let _ = sender
                    .send(Message::Text(
                        json!({"type":"error","message": e}).to_string().into(),
                    ))
                    .await;
                return;
            }
        };

        let mut stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let _ = sender
                    .send(Message::Text(
                        json!({"type":"error","message":"ffmpeg stdout missing"})
                            .to_string()
                            .into(),
                    ))
                    .await;
                return;
            }
        };

        let mut jpeg_buf: Vec<u8> = Vec::new();
        let mut read_buf = [0u8; 65536];
        let mut ctrl = Ctrl::Run;
        let mut export_requested = false;
        let mut reached_eof = false;

        use std::io::Read;
        'extract: loop {
            // Drain client control messages (non-blocking)
            while let Ok(Some(Ok(msg))) =
                tokio::time::timeout(std::time::Duration::from_millis(0), receiver.next()).await
            {
                if let Message::Text(t) = msg {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                        match v.get("type").and_then(|x| x.as_str()) {
                            Some("pause") => ctrl = Ctrl::Pause,
                            Some("stop") => ctrl = Ctrl::Stop,
                            Some("export") => export_requested = true,
                            Some("config") => {
                                if let Some(c) = v.get("confidence").and_then(|c| c.as_f64()) {
                                    state.detector.lock().await.set_confidence(c as f32);
                                }
                            }
                            // ignore resume while running
                            _ => {}
                        }
                    }
                }
            }

            if ctrl != Ctrl::Run {
                break 'extract;
            }

            if export_requested {
                export_requested = false;
                if processed > 0 {
                    match assemble_annotated_video(
                        &state.root,
                        &job_id,
                        &frames_dir,
                        process_fps,
                        false, // keep frames
                    )
                    .await
                    {
                        Ok(url) => {
                            last_export_url = Some(url.clone());
                            let _ = sender
                                .send(Message::Text(
                                    json!({
                                        "type": "export_ready",
                                        "output": url,
                                        "processed": processed,
                                        "expected_frames": expected_frames,
                                        "progress": progress_value(processed, expected_frames, false),
                                        "partial": processed < expected_frames,
                                    })
                                    .to_string()
                                    .into(),
                                ))
                                .await;
                        }
                        Err(e) => {
                            let _ = sender
                                .send(Message::Text(
                                    json!({"type":"error","message": format!("Export failed: {e}")})
                                        .to_string()
                                        .into(),
                                ))
                                .await;
                        }
                    }
                }
            }

            let n = match stdout.read(&mut read_buf) {
                Ok(0) => {
                    reached_eof = true;
                    break 'extract;
                }
                Ok(n) => n,
                Err(_) => {
                    reached_eof = true;
                    break 'extract;
                }
            };
            jpeg_buf.extend_from_slice(&read_buf[..n]);

            while let Some((start, end)) = find_jpeg(&jpeg_buf) {
                // Re-check control between frames
                while let Ok(Some(Ok(msg))) =
                    tokio::time::timeout(std::time::Duration::from_millis(0), receiver.next()).await
                {
                    if let Message::Text(t) = msg {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                            match v.get("type").and_then(|x| x.as_str()) {
                                Some("pause") => ctrl = Ctrl::Pause,
                                Some("stop") => ctrl = Ctrl::Stop,
                                Some("export") => export_requested = true,
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
                if ctrl != Ctrl::Run {
                    break 'extract;
                }

                let jpeg = jpeg_buf[start..=end].to_vec();
                jpeg_buf.drain(..=end);

                let det = state.detector.clone();
                let jpeg_clone = jpeg.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut d = det.blocking_lock();
                    d.predict_jpeg(&jpeg_clone)
                })
                .await;

                match result {
                    Ok(Ok((out_jpeg, meta_out))) => {
                        processed += 1;
                        let fp = frames_dir.join(format!("frame_{:06}.jpg", processed));
                        let _ = std::fs::write(&fp, &out_jpeg);

                        let progress = progress_value(processed, expected_frames, false);
                        let frame = format!("data:image/jpeg;base64,{}", B64.encode(&out_jpeg));
                        let payload = json!({
                            "type": "result",
                            "frame": frame,
                            "detections": meta_out.detections,
                            "summary": meta_out.summary,
                            "fps": meta_out.fps,
                            "inference_ms": meta_out.inference_ms,
                            "device": meta_out.device,
                            "count": meta_out.count,
                            "frame_index": processed,
                            "processed": processed,
                            "expected_frames": expected_frames,
                            "progress": progress,
                            "partial": true,
                        });
                        if sender
                            .send(Message::Text(payload.to_string().into()))
                            .await
                            .is_err()
                        {
                            ctrl = Ctrl::Stop;
                            break 'extract;
                        }

                        // Soft stop if we met/exceeded expected (ffmpeg may overshoot slightly)
                        if processed >= expected_frames {
                            reached_eof = true;
                            break 'extract;
                        }
                    }
                    Ok(Err(e)) => error!("frame inference: {e:#}"),
                    Err(e) => error!("join: {e}"),
                }

                if export_requested {
                    // handle after this frame via top of loop
                    break;
                }
            }
        }

        let _ = child.kill();
        let _ = child.wait();

        if ctrl == Ctrl::Stop {
            break 'session;
        }

        if reached_eof && ctrl == Ctrl::Run {
            finished_naturally = true;
            break 'session;
        }

        if ctrl == Ctrl::Pause {
            // Build partial video for download, keep frames for resume.
            let (output_url, assemble_error) = if processed > 0 {
                match assemble_annotated_video(
                    &state.root,
                    &job_id,
                    &frames_dir,
                    process_fps,
                    false,
                )
                .await
                {
                    Ok(url) => {
                        last_export_url = Some(url.clone());
                        (Some(url), None)
                    }
                    Err(e) => (None, Some(e)),
                }
            } else {
                (None, Some("No frames processed yet".into()))
            };

            let progress = progress_value(processed, expected_frames, false);
            let _ = sender
                .send(Message::Text(
                    json!({
                        "type": "paused",
                        "processed": processed,
                        "expected_frames": expected_frames,
                        "progress": progress,
                        "output": output_url,
                        "error": assemble_error,
                        "can_resume": processed < expected_frames && !finished_naturally,
                    })
                    .to_string()
                    .into(),
                ))
                .await;

            // Wait for resume / stop / export
            loop {
                match receiver.next().await {
                    Some(Ok(Message::Text(t))) => {
                        let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else {
                            continue;
                        };
                        match v.get("type").and_then(|x| x.as_str()) {
                            Some("resume") => {
                                let _ = sender
                                    .send(Message::Text(
                                        json!({
                                            "type": "resumed",
                                            "processed": processed,
                                            "expected_frames": expected_frames,
                                            "progress": progress_value(processed, expected_frames, false),
                                        })
                                        .to_string()
                                        .into(),
                                    ))
                                    .await;
                                continue 'session;
                            }
                            Some("stop") => break 'session,
                            Some("export") => {
                                if processed > 0 {
                                    if let Ok(url) = assemble_annotated_video(
                                        &state.root,
                                        &job_id,
                                        &frames_dir,
                                        process_fps,
                                        false,
                                    )
                                    .await
                                    {
                                        last_export_url = Some(url.clone());
                                        let _ = sender
                                            .send(Message::Text(
                                                json!({
                                                    "type": "export_ready",
                                                    "output": url,
                                                    "processed": processed,
                                                    "expected_frames": expected_frames,
                                                    "progress": progress_value(processed, expected_frames, false),
                                                    "partial": true,
                                                })
                                                .to_string()
                                                .into(),
                                            ))
                                            .await;
                                    }
                                }
                            }
                            Some("config") => {
                                if let Some(c) = v.get("confidence").and_then(|c| c.as_f64()) {
                                    state.detector.lock().await.set_confidence(c as f32);
                                }
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break 'session,
                    _ => {}
                }
            }
        }
    }

    // Final assembly (partial or complete)
    let mut output_url = last_export_url;
    let mut assemble_error = None;
    if processed > 0 {
        match assemble_annotated_video(&state.root, &job_id, &frames_dir, process_fps, true).await {
            Ok(url) => output_url = Some(url),
            Err(e) => {
                if output_url.is_none() {
                    assemble_error = Some(e);
                }
            }
        }
    } else {
        assemble_error = Some("No frames were processed".into());
        let _ = std::fs::remove_dir_all(&frames_dir);
    }

    let _ = sender
        .send(Message::Text(
            json!({
                "type": "done",
                "processed": processed,
                "expected_frames": expected_frames,
                "progress": 1.0,
                "complete": finished_naturally,
                "output": output_url,
                "error": assemble_error,
            })
            .to_string()
            .into(),
        ))
        .await;
}

fn progress_value(processed: u64, expected: u64, done: bool) -> f64 {
    if done {
        return 1.0;
    }
    if expected == 0 {
        return 0.0;
    }
    // Never report 100% until the job is fully finished.
    ((processed as f64) / (expected as f64)).clamp(0.0, 0.999)
}


fn spawn_frame_extractor(
    video_path: &Path,
    process_fps: f64,
    start_sec: f64,
) -> Result<std::process::Child, String> {
    let mut args: Vec<String> = vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
    ];
    // Accurate-ish seek before input for resume.
    if start_sec > 0.05 {
        args.push("-ss".into());
        args.push(format!("{start_sec:.3}"));
    }
    args.extend([
        "-i".into(),
        video_path.to_string_lossy().to_string(),
        "-vf".into(),
        format!("fps={process_fps:.3}"),
        "-f".into(),
        "image2pipe".into(),
        "-vcodec".into(),
        "mjpeg".into(),
        "-q:v".into(),
        "5".into(),
        "-".into(),
    ]);

    Command::new("ffmpeg")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("ffmpeg failed: {e} (is ffmpeg installed?)"))
}

/// Assemble JPEG sequence into MP4. If `cleanup_frames`, deletes the frames dir on success.
async fn assemble_annotated_video(
    root: &Path,
    job_id: &str,
    frames_dir: &Path,
    fps: f64,
    cleanup_frames: bool,
) -> Result<String, String> {
    let out_video = root.join("outputs").join(format!("{job_id}_annotated.mp4"));
    let pattern = frames_dir.join("frame_%06d.jpg");
    let pattern_s = pattern.to_string_lossy().to_string();
    let out_s = out_video.to_string_lossy().to_string();
    let frames_dir = frames_dir.to_path_buf();
    let out_clone = out_video.clone();
    let fps = fps.clamp(1.0, 60.0);

    let result = tokio::task::spawn_blocking(move || {
        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                &format!("{fps:.3}"),
                "-i",
                &pattern_s,
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                &out_s,
            ])
            .output();
        match output {
            Ok(o) if o.status.success() && out_clone.exists() => {
                if cleanup_frames {
                    let _ = std::fs::remove_dir_all(&frames_dir);
                }
                Ok(())
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
                Err(if stderr.is_empty() {
                    format!("ffmpeg exited with status {}", o.status)
                } else {
                    stderr
                })
            }
            Err(e) => Err(format!("ffmpeg spawn failed: {e}")),
        }
    })
    .await
    .map_err(|e| format!("assemble join: {e}"))?;

    result?;
    Ok(format!("/api/download/{job_id}"))
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

#[derive(Debug, Clone)]
struct VideoMeta {
    fps: f64,
    frames: u64,
    width: u32,
    height: u32,
    duration_s: f64,
}

fn probe_video_meta(path: &Path) -> VideoMeta {
    let path_s = path.to_str().unwrap_or("");
    let stream_out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            path_s,
        ])
        .output();
    let format_out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path_s,
        ])
        .output();

    let mut meta = VideoMeta {
        fps: 0.0,
        frames: 0,
        width: 0,
        height: 0,
        duration_s: 0.0,
    };

    if let Ok(out) = stream_out {
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or(json!({}));
        let stream = v
            .get("streams")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .cloned()
            .unwrap_or(json!({}));
        meta.width = stream.get("width").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        meta.height = stream.get("height").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        meta.frames = stream
            .get("nb_frames")
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse().ok())
            .or_else(|| stream.get("nb_frames").and_then(|x| x.as_u64()))
            .unwrap_or(0);
        meta.fps = stream
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
        if let Some(d) = stream
            .get("duration")
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse().ok())
            .or_else(|| stream.get("duration").and_then(|x| x.as_f64()))
        {
            if d > 0.0 {
                meta.duration_s = d;
            }
        }
    }

    if let Ok(out) = format_out {
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or(json!({}));
        if let Some(d) = v
            .pointer("/format/duration")
            .and_then(|x| x.as_str())
            .and_then(|s| s.parse().ok())
            .or_else(|| v.pointer("/format/duration").and_then(|x| x.as_f64()))
        {
            if d > 0.0 {
                meta.duration_s = d;
            }
        }
    }

    // Derive duration from frames if still missing
    if meta.duration_s <= 0.0 && meta.frames > 0 && meta.fps > 0.0 {
        meta.duration_s = meta.frames as f64 / meta.fps;
    }

    meta
}

/// Used by upload endpoint metadata.
fn probe_video(path: &Path) -> (f64, u64, u32, u32) {
    let m = probe_video_meta(path);
    (m.fps, m.frames, m.width, m.height)
}
