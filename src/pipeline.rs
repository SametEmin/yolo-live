//! Four-stage multi-threaded detection pipeline.
//!
//! ```text
//!  [capture] → [preprocess] → [inference] → [render] → results
//!     thr1         thr2           thr3         thr4
//! ```
//!
//! Bounded channels provide back-pressure so a slow stage does not unbounded-queue
//! frames (latest-frame semantics: queue capacity 2).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ab_glyph::FontArc;
use anyhow::Result;
use tracing::{error, info};

use crate::detector::{
    stage_capture_decode, stage_preprocess, stage_render, Detection, FrameResult, ObjectDetector,
    PreprocessedFrame,
};

const QUEUE: usize = 8;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PipelineStats {
    pub mode: String,
    pub frames: u64,
    pub elapsed_s: f64,
    pub avg_fps: f64,
    pub avg_inference_ms: f64,
    pub threads: Vec<String>,
}

pub struct PipelineOutput {
    pub frame_id: u64,
    pub jpeg: Vec<u8>,
    pub meta: FrameResult,
    pub stage_ms: StageTimings,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StageTimings {
    pub capture_ms: f32,
    pub preprocess_ms: f32,
    pub inference_ms: f32,
    pub render_ms: f32,
}

enum CaptureMsg {
    Frame { id: u64, jpeg: Vec<u8> },
    Shutdown,
}

struct PreprocessMsg {
    id: u64,
    rgb: image::RgbImage,
    capture_ms: f32,
}

struct InferMsg {
    id: u64,
    prepared: PreprocessedFrame,
    capture_ms: f32,
    preprocess_ms: f32,
}

struct RenderMsg {
    id: u64,
    rgb: image::RgbImage,
    detections: Vec<Detection>,
    inference_ms: f32,
    capture_ms: f32,
    preprocess_ms: f32,
}

/// Multi-threaded YOLO pipeline with dedicated OS threads per stage.
pub struct DetectionPipeline {
    capture_tx: SyncSender<CaptureMsg>,
    /// Mutex so AppState/Pipeline is Shareable across axum tasks (Receiver is !Sync).
    result_rx: Mutex<Receiver<PipelineOutput>>,
    conf_bits: Arc<AtomicU32>,
    frame_counter: AtomicU64,
    completed: Arc<AtomicU64>,
    sum_infer_ms: Arc<AtomicU64>, // fixed-point ×1000
    shutdown: Arc<AtomicBool>,
    joins: Mutex<Vec<JoinHandle<()>>>,
    device: String,
}

impl DetectionPipeline {
    pub fn start(mut detector: ObjectDetector) -> Result<Self> {
        let device = detector.device().to_string();
        let font: FontArc = detector.font().clone();
        let conf_bits = Arc::new(AtomicU32::new(detector.confidence().to_bits()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicU64::new(0));
        let sum_infer_ms = Arc::new(AtomicU64::new(0));

        let (cap_tx, cap_rx) = sync_channel::<CaptureMsg>(QUEUE);
        let (pre_tx, pre_rx) = sync_channel::<PreprocessMsg>(QUEUE);
        let (inf_tx, inf_rx) = sync_channel::<InferMsg>(QUEUE);
        let (ren_tx, ren_rx) = sync_channel::<RenderMsg>(QUEUE);
        let (out_tx, out_rx) = sync_channel::<PipelineOutput>(QUEUE);

        // ── 1. Capture thread ──────────────────────────────────────────────
        let shut_c = shutdown.clone();
        let t_capture = thread::Builder::new()
            .name("yolo-capture".into())
            .spawn(move || {
                info!("[pipeline] capture thread started");
                while let Ok(msg) = cap_rx.recv() {
                    match msg {
                        CaptureMsg::Shutdown => break,
                        CaptureMsg::Frame { id, jpeg } => {
                            if shut_c.load(Ordering::Relaxed) {
                                break;
                            }
                            let t0 = Instant::now();
                            match stage_capture_decode(&jpeg) {
                                Ok(rgb) => {
                                    let capture_ms = t0.elapsed().as_secs_f32() * 1000.0;
                                    // Blocking send: back-pressure instead of dropping mid-pipeline.
                                    if pre_tx
                                        .send(PreprocessMsg {
                                            id,
                                            rgb,
                                            capture_ms,
                                        })
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                                Err(e) => error!("[capture] {e:#}"),
                            }
                        }
                    }
                }
                // Propagate shutdown
                drop(pre_tx);
                info!("[pipeline] capture thread stopped");
            })?;

        // ── 2. Preprocess thread ───────────────────────────────────────────
        let shut_p = shutdown.clone();
        let t_pre = thread::Builder::new()
            .name("yolo-preprocess".into())
            .spawn(move || {
                info!("[pipeline] preprocess thread started");
                while let Ok(msg) = pre_rx.recv() {
                    if shut_p.load(Ordering::Relaxed) {
                        break;
                    }
                    let t0 = Instant::now();
                    let prepared = stage_preprocess(msg.rgb);
                    let preprocess_ms = t0.elapsed().as_secs_f32() * 1000.0;
                    if inf_tx
                        .send(InferMsg {
                            id: msg.id,
                            prepared,
                            capture_ms: msg.capture_ms,
                            preprocess_ms,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                drop(inf_tx);
                info!("[pipeline] preprocess thread stopped");
            })?;

        // ── 3. Inference thread (owns ONNX session) ────────────────────────
        let conf_i = conf_bits.clone();
        let shut_i = shutdown.clone();
        let sum_i = sum_infer_ms.clone();
        let t_inf = thread::Builder::new()
            .name("yolo-inference".into())
            .spawn(move || {
                info!("[pipeline] inference thread started ({})", detector.device());
                while let Ok(msg) = inf_rx.recv() {
                    if shut_i.load(Ordering::Relaxed) {
                        break;
                    }
                    let conf = f32::from_bits(conf_i.load(Ordering::Relaxed));
                    detector.set_confidence(conf);
                    match detector.stage_inference(&msg.prepared) {
                        Ok((dets, inference_ms)) => {
                            sum_i.fetch_add((inference_ms * 1000.0) as u64, Ordering::Relaxed);
                            if ren_tx
                                .send(RenderMsg {
                                    id: msg.id,
                                    rgb: msg.prepared.rgb,
                                    detections: dets,
                                    inference_ms,
                                    capture_ms: msg.capture_ms,
                                    preprocess_ms: msg.preprocess_ms,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => error!("[inference] {e:#}"),
                    }
                }
                drop(ren_tx);
                info!("[pipeline] inference thread stopped");
            })?;

        // ── 4. Render thread ───────────────────────────────────────────────
        let shut_r = shutdown.clone();
        let completed_r = completed.clone();
        let device_r = device.clone();
        let mut last_out = Instant::now();
        let mut ema_fps = 0.0f32;
        let t_ren = thread::Builder::new()
            .name("yolo-render".into())
            .spawn(move || {
                info!("[pipeline] render thread started");
                while let Ok(msg) = ren_rx.recv() {
                    if shut_r.load(Ordering::Relaxed) {
                        break;
                    }
                    let t0 = Instant::now();
                    let now = Instant::now();
                    let dt = now.duration_since(last_out).as_secs_f32();
                    last_out = now;
                    if dt > 0.0 {
                        let inst = 1.0 / dt;
                        ema_fps = if ema_fps <= 0.0 {
                            inst
                        } else {
                            0.85 * ema_fps + 0.15 * inst
                        };
                    }
                    match stage_render(
                        msg.rgb,
                        &msg.detections,
                        &font,
                        &device_r,
                        msg.inference_ms,
                        ema_fps,
                    ) {
                        Ok((jpeg, meta)) => {
                            let render_ms = t0.elapsed().as_secs_f32() * 1000.0;
                            completed_r.fetch_add(1, Ordering::Relaxed);
                            if out_tx
                                .send(PipelineOutput {
                                    frame_id: msg.id,
                                    jpeg,
                                    meta,
                                    stage_ms: StageTimings {
                                        capture_ms: msg.capture_ms,
                                        preprocess_ms: msg.preprocess_ms,
                                        inference_ms: msg.inference_ms,
                                        render_ms,
                                    },
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => error!("[render] {e:#}"),
                    }
                }
                info!("[pipeline] render thread stopped");
            })?;

        Ok(Self {
            capture_tx: cap_tx,
            result_rx: Mutex::new(out_rx),
            conf_bits,
            frame_counter: AtomicU64::new(0),
            completed,
            sum_infer_ms,
            shutdown,
            joins: Mutex::new(vec![t_capture, t_pre, t_inf, t_ren]),
            device,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn set_confidence(&self, conf: f32) {
        let c = conf.clamp(0.05, 0.95);
        self.conf_bits.store(c.to_bits(), Ordering::Relaxed);
    }

    pub fn confidence(&self) -> f32 {
        f32::from_bits(self.conf_bits.load(Ordering::Relaxed))
    }

    /// Push a captured JPEG into the pipeline (non-blocking; may drop if full).
    pub fn submit_capture(&self, jpeg: Vec<u8>) -> u64 {
        let id = self.frame_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.capture_tx.try_send(CaptureMsg::Frame { id, jpeg });
        id
    }

    /// Like `submit_capture`, but reports whether the frame was accepted.
    pub fn submit_capture_ok(&self, jpeg: Vec<u8>) -> bool {
        let id = self.frame_counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.capture_tx
            .try_send(CaptureMsg::Frame { id, jpeg })
            .is_ok()
    }

    /// Blocking submit used by sequential-comparison harness (still uses pipeline threads).
    pub fn submit_capture_blocking(&self, jpeg: Vec<u8>) -> u64 {
        let id = self.frame_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.capture_tx.send(CaptureMsg::Frame { id, jpeg });
        id
    }

    pub fn try_recv(&self) -> Option<PipelineOutput> {
        self.result_rx.lock().ok()?.try_recv().ok()
    }

    pub fn recv_timeout(&self, dur: Duration) -> Option<PipelineOutput> {
        // Poll without holding the mutex across long waits (keeps the channel free).
        let deadline = Instant::now() + dur;
        loop {
            if let Some(v) = self.try_recv() {
                return Some(v);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    pub fn completed_frames(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }
}

impl Drop for DetectionPipeline {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let _ = self.capture_tx.send(CaptureMsg::Shutdown);
        if let Ok(mut joins) = self.joins.lock() {
            for h in joins.drain(..) {
                let _ = h.join();
            }
        }
    }
}

/// Run sequential (single-thread) vs multi-thread pipeline FPS comparison.
pub fn run_benchmark(model_path: &std::path::Path, frames: u32) -> Result<serde_json::Value> {
    use image::{Rgb, RgbImage};
    use serde_json::json;

    let frames = frames.clamp(10, 300);
    info!("Benchmark: loading model for sequential baseline…");
    let mut seq = ObjectDetector::load(model_path)?;

    // Synthetic "camera" frames (varied noise so decode/preproc do real work).
    let mut test_jpegs: Vec<Vec<u8>> = Vec::with_capacity(frames as usize);
    for i in 0..frames {
        let mut img = RgbImage::from_fn(640, 480, |x, y| {
            let v = ((x + y + i * 3) % 255) as u8;
            Rgb([v, v.wrapping_mul(3), 255u8.wrapping_sub(v)])
        });
        // Draw a bright rectangle so YOLO has structure (optional).
        for x in 200..400 {
            for y in 150..350 {
                img.put_pixel(x, y, Rgb([200, 40, 40]));
            }
        }
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 85);
        enc.encode(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )?;
        test_jpegs.push(buf);
    }

    // ── BEFORE: single-threaded sequential stages ──────────────────────────
    // Warm-up
    for j in test_jpegs.iter().take(3) {
        let _ = seq.predict_jpeg(j);
    }
    let t0 = Instant::now();
    let mut seq_infer_sum = 0.0f64;
    for j in &test_jpegs {
        let (_jpeg, meta) = seq.predict_jpeg(j)?;
        seq_infer_sum += meta.inference_ms as f64;
    }
    let seq_elapsed = t0.elapsed().as_secs_f64();
    let seq_fps = frames as f64 / seq_elapsed;
    let seq_stats = PipelineStats {
        mode: "single_thread_sequential".into(),
        frames: frames as u64,
        elapsed_s: (seq_elapsed * 1000.0).round() / 1000.0,
        avg_fps: (seq_fps * 100.0).round() / 100.0,
        avg_inference_ms: ((seq_infer_sum / frames as f64) * 100.0).round() / 100.0,
        threads: vec![
            "capture+preprocess+inference+render (1 OS thread)".into(),
        ],
    };

    // Drop sequential detector before loading pipeline detector (free CoreML mem).
    drop(seq);

    // ── AFTER: 4-thread pipeline ───────────────────────────────────────────
    info!("Benchmark: starting 4-thread pipeline…");
    let pipe_det = ObjectDetector::load(model_path)?;
    let pipeline = DetectionPipeline::start(pipe_det)?;

    // Warm-up pipeline
    for j in test_jpegs.iter().take(3) {
        pipeline.submit_capture_blocking(j.clone());
    }
    let mut warmed = 0u32;
    let warm_deadline = Instant::now() + Duration::from_secs(30);
    while warmed < 3 && Instant::now() < warm_deadline {
        if pipeline.recv_timeout(Duration::from_millis(200)).is_some() {
            warmed += 1;
        }
    }

    let pipeline = Arc::new(pipeline);
    let before_completed = pipeline.completed_frames();
    let t1 = Instant::now();
    // Producer + consumer run concurrently so the pipeline can stay full.
    let jpegs = test_jpegs.clone();
    let prod = pipeline.clone();
    let producer = thread::spawn(move || {
        for j in jpegs {
            prod.submit_capture_blocking(j);
        }
    });
    let mut got = 0u32;
    let mut pipe_infer_sum = 0.0f64;
    let deadline = Instant::now() + Duration::from_secs(120);
    while got < frames && Instant::now() < deadline {
        if let Some(out) = pipeline.recv_timeout(Duration::from_millis(500)) {
            pipe_infer_sum += out.meta.inference_ms as f64;
            got += 1;
        }
    }
    let _ = producer.join();
    // Drain any remaining
    while got < frames {
        if let Some(out) = pipeline.recv_timeout(Duration::from_millis(200)) {
            pipe_infer_sum += out.meta.inference_ms as f64;
            got += 1;
        } else {
            break;
        }
    }
    let pipe_elapsed = t1.elapsed().as_secs_f64();
    let pipe_fps = if pipe_elapsed > 0.0 {
        got as f64 / pipe_elapsed
    } else {
        0.0
    };
    let after_completed = pipeline.completed_frames() - before_completed;

    let pipe_stats = PipelineStats {
        mode: "multi_thread_pipeline".into(),
        frames: got as u64,
        elapsed_s: (pipe_elapsed * 1000.0).round() / 1000.0,
        avg_fps: (pipe_fps * 100.0).round() / 100.0,
        avg_inference_ms: if got > 0 {
            ((pipe_infer_sum / got as f64) * 100.0).round() / 100.0
        } else {
            0.0
        },
        threads: vec![
            "yolo-capture".into(),
            "yolo-preprocess".into(),
            "yolo-inference".into(),
            "yolo-render".into(),
        ],
    };

    let speedup = if seq_stats.avg_fps > 0.0 {
        (pipe_stats.avg_fps / seq_stats.avg_fps * 100.0).round() / 100.0
    } else {
        0.0
    };

    Ok(json!({
        "device": pipeline.device(),
        "frames_requested": frames,
        "before": seq_stats,
        "after": pipe_stats,
        "speedup_x": speedup,
        "completed_pipeline_frames": after_completed,
        "notes": [
            "before = all 4 stages on one thread (predict_jpeg)",
            "after = dedicated OS threads: capture | preprocess | inference | render",
            "FPS = completed frames / wall-clock time (throughput)",
            "Pipeline benefits when capture/pre/render overlap with inference"
        ]
    }))
}
