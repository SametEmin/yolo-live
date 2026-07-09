//! Low-latency 4-stage multi-threaded live detection pipeline.
//!
//! ```text
//!  capture  →  preprocess  →  inference  →  render  →  result
//!  (decode)    (letterbox)    (YOLO/CoreML)  (draw)
//! ```
//!
//! **Latest-frame semantics**: each stage keeps only the newest item. When a
//! stage is busy, older frames are overwritten so output tracks the camera
//! (no multi-frame queue lag).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ab_glyph::FontArc;
use anyhow::Result;
use tracing::{error, info};

use crate::detector::{
    stage_capture_decode, stage_preprocess, stage_preprocess_fast, stage_render,
    stage_render_fast, Detection, FrameResult, ObjectDetector, PreprocessedFrame,
};
use crate::tracker::BoxTracker;

// ─── Latest-frame slot ─────────────────────────────────────────────────────

struct LatestSlot<T> {
    data: Mutex<Option<T>>,
    cv: Condvar,
    closed: AtomicBool,
}

impl<T> LatestSlot<T> {
    fn new() -> Self {
        Self {
            data: Mutex::new(None),
            cv: Condvar::new(),
            closed: AtomicBool::new(false),
        }
    }

    /// Overwrite with the newest value (drops stale).
    fn push(&self, value: T) {
        let mut g = self.data.lock().unwrap();
        *g = Some(value);
        self.cv.notify_one();
    }

    /// Block until a value is available (or closed).
    fn take_wait(&self) -> Option<T> {
        let mut g = self.data.lock().unwrap();
        loop {
            if let Some(v) = g.take() {
                return Some(v);
            }
            if self.closed.load(Ordering::Relaxed) {
                return None;
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    fn try_take(&self) -> Option<T> {
        self.data.lock().unwrap().take()
    }

    fn clear(&self) {
        *self.data.lock().unwrap() = None;
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        self.cv.notify_all();
    }
}

// ─── Internal messages ─────────────────────────────────────────────────────

struct CapIn {
    id: u64,
    jpeg: Vec<u8>,
}

struct PreIn {
    id: u64,
    rgb: image::RgbImage,
    capture_ms: f32,
}

struct InfIn {
    id: u64,
    prepared: PreprocessedFrame,
    capture_ms: f32,
    preprocess_ms: f32,
}

struct RenIn {
    id: u64,
    rgb: image::RgbImage,
    detections: Vec<Detection>,
    inference_ms: f32,
    capture_ms: f32,
    preprocess_ms: f32,
}

// ─── Public types ──────────────────────────────────────────────────────────

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

/// Multi-threaded YOLO pipeline with **latest-frame** handoff (low latency).
pub struct DetectionPipeline {
    cap_in: Arc<LatestSlot<CapIn>>,
    result_slot: Arc<LatestSlot<PipelineOutput>>,
    conf_bits: Arc<AtomicU32>,
    frame_counter: AtomicU64,
    completed: Arc<AtomicU64>,
    sum_infer_ms: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
    joins: Mutex<Vec<JoinHandle<()>>>,
    device: String,
    live_fast: Arc<AtomicBool>,
    /// Intermediate slots so `clear()` can flush stale work after mode switches.
    pre_in: Arc<LatestSlot<PreIn>>,
    inf_in: Arc<LatestSlot<InfIn>>,
    ren_in: Arc<LatestSlot<RenIn>>,
    /// Signal render thread to reset box smoother (new live session).
    tracker_reset: Arc<AtomicBool>,
}

impl DetectionPipeline {
    pub fn start(mut detector: ObjectDetector) -> Result<Self> {
        let device = detector.device().to_string();
        let font: FontArc = detector.font().clone();
        let conf_bits = Arc::new(AtomicU32::new(detector.confidence().to_bits()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicU64::new(0));
        let sum_infer_ms = Arc::new(AtomicU64::new(0));
        let live_fast = Arc::new(AtomicBool::new(true));
        let tracker_reset = Arc::new(AtomicBool::new(false));

        let cap_in = Arc::new(LatestSlot::<CapIn>::new());
        let pre_in = Arc::new(LatestSlot::<PreIn>::new());
        let inf_in = Arc::new(LatestSlot::<InfIn>::new());
        let ren_in = Arc::new(LatestSlot::<RenIn>::new());
        let result_slot = Arc::new(LatestSlot::<PipelineOutput>::new());

        let mut joins = Vec::new();

        // ── 1. Capture / decode ────────────────────────────────────────────
        {
            let shut = shutdown.clone();
            let cap_in_t = cap_in.clone();
            let pre_in_t = pre_in.clone();
            joins.push(
                thread::Builder::new()
                    .name("yolo-capture".into())
                    .spawn(move || {
                        info!("[pipeline] capture thread started (latest-frame)");
                        while !shut.load(Ordering::Relaxed) {
                            let Some(msg) = cap_in_t.take_wait() else { break };
                            if shut.load(Ordering::Relaxed) {
                                break;
                            }
                            let t0 = Instant::now();
                            match stage_capture_decode(&msg.jpeg) {
                                Ok(rgb) => {
                                    pre_in_t.push(PreIn {
                                        id: msg.id,
                                        rgb,
                                        capture_ms: t0.elapsed().as_secs_f32() * 1000.0,
                                    });
                                }
                                Err(e) => error!("[capture] {e:#}"),
                            }
                        }
                        pre_in_t.close();
                        info!("[pipeline] capture thread stopped");
                    })?,
            );
        }

        // ── 2. Preprocess ──────────────────────────────────────────────────
        {
            let shut = shutdown.clone();
            let pre_in_t = pre_in.clone();
            let inf_in_t = inf_in.clone();
            let live_fast_t = live_fast.clone();
            joins.push(
                thread::Builder::new()
                    .name("yolo-preprocess".into())
                    .spawn(move || {
                        info!("[pipeline] preprocess thread started (latest-frame)");
                        while !shut.load(Ordering::Relaxed) {
                            let Some(msg) = pre_in_t.take_wait() else { break };
                            if shut.load(Ordering::Relaxed) {
                                break;
                            }
                            let t0 = Instant::now();
                            let prepared = if live_fast_t.load(Ordering::Relaxed) {
                                stage_preprocess_fast(msg.rgb)
                            } else {
                                stage_preprocess(msg.rgb)
                            };
                            inf_in_t.push(InfIn {
                                id: msg.id,
                                prepared,
                                capture_ms: msg.capture_ms,
                                preprocess_ms: t0.elapsed().as_secs_f32() * 1000.0,
                            });
                        }
                        inf_in_t.close();
                        info!("[pipeline] preprocess thread stopped");
                    })?,
            );
        }

        // ── 3. Inference (owns ONNX session) ───────────────────────────────
        {
            let shut = shutdown.clone();
            let conf_i = conf_bits.clone();
            let sum_i = sum_infer_ms.clone();
            let inf_in_t = inf_in.clone();
            let ren_in_t = ren_in.clone();
            joins.push(
                thread::Builder::new()
                    .name("yolo-inference".into())
                    .spawn(move || {
                        info!(
                            "[pipeline] inference thread started ({}) latest-frame",
                            detector.device()
                        );
                        while !shut.load(Ordering::Relaxed) {
                            let Some(msg) = inf_in_t.take_wait() else { break };
                            if shut.load(Ordering::Relaxed) {
                                break;
                            }
                            let conf = f32::from_bits(conf_i.load(Ordering::Relaxed));
                            detector.set_confidence(conf);
                            match detector.stage_inference(&msg.prepared) {
                                Ok((dets, inference_ms)) => {
                                    sum_i.fetch_add(
                                        (inference_ms * 1000.0) as u64,
                                        Ordering::Relaxed,
                                    );
                                    ren_in_t.push(RenIn {
                                        id: msg.id,
                                        rgb: msg.prepared.rgb,
                                        detections: dets,
                                        inference_ms,
                                        capture_ms: msg.capture_ms,
                                        preprocess_ms: msg.preprocess_ms,
                                    });
                                }
                                Err(e) => error!("[inference] {e:#}"),
                            }
                        }
                        ren_in_t.close();
                        info!("[pipeline] inference thread stopped");
                    })?,
            );
        }

        // ── 4. Render ──────────────────────────────────────────────────────
        {
            let shut = shutdown.clone();
            let completed_r = completed.clone();
            let device_r = device.clone();
            let ren_in_t = ren_in.clone();
            let result_t = result_slot.clone();
            let live_fast_t = live_fast.clone();
            let tracker_reset_t = tracker_reset.clone();
            let mut last_out = Instant::now();
            let mut ema_fps = 0.0f32;
            let mut tracker = BoxTracker::new();
            joins.push(
                thread::Builder::new()
                    .name("yolo-render".into())
                    .spawn(move || {
                        info!("[pipeline] render thread started (latest-frame + smooth tracker)");
                        while !shut.load(Ordering::Relaxed) {
                            let Some(msg) = ren_in_t.take_wait() else { break };
                            if shut.load(Ordering::Relaxed) {
                                break;
                            }
                            if tracker_reset_t.swap(false, Ordering::Relaxed) {
                                tracker.reset();
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
                            // Mathematical smooth tracking (IoU match + CV-EMA). Cheap vs inference.
                            let smooth_dets = tracker.update(&msg.detections);
                            let rendered = if live_fast_t.load(Ordering::Relaxed) {
                                stage_render_fast(
                                    msg.rgb,
                                    &smooth_dets,
                                    &font,
                                    &device_r,
                                    msg.inference_ms,
                                    ema_fps,
                                )
                            } else {
                                stage_render(
                                    msg.rgb,
                                    &smooth_dets,
                                    &font,
                                    &device_r,
                                    msg.inference_ms,
                                    ema_fps,
                                )
                            };
                            match rendered {
                                Ok((jpeg, meta)) => {
                                    let render_ms = t0.elapsed().as_secs_f32() * 1000.0;
                                    completed_r.fetch_add(1, Ordering::Relaxed);
                                    result_t.push(PipelineOutput {
                                        frame_id: msg.id,
                                        jpeg,
                                        meta,
                                        stage_ms: StageTimings {
                                            capture_ms: msg.capture_ms,
                                            preprocess_ms: msg.preprocess_ms,
                                            inference_ms: msg.inference_ms,
                                            render_ms,
                                        },
                                    });
                                }
                                Err(e) => error!("[render] {e:#}"),
                            }
                        }
                        result_t.close();
                        info!("[pipeline] render thread stopped");
                    })?,
            );
        }

        Ok(Self {
            cap_in,
            result_slot,
            conf_bits,
            frame_counter: AtomicU64::new(0),
            completed,
            sum_infer_ms,
            shutdown,
            joins: Mutex::new(joins),
            device,
            live_fast,
            pre_in,
            inf_in,
            ren_in,
            tracker_reset,
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

    pub fn set_live_fast(&self, fast: bool) {
        self.live_fast.store(fast, Ordering::Relaxed);
    }

    /// Drop any stale in-flight frames (call when starting a new live session).
    pub fn clear_pending(&self) {
        self.cap_in.clear();
        self.pre_in.clear();
        self.inf_in.clear();
        self.ren_in.clear();
        self.result_slot.clear();
        // Reset box smoother so a new live session does not inherit old tracks.
        self.tracker_reset.store(true, Ordering::Relaxed);
    }

    /// Push camera JPEG; always keeps only the **latest** frame (overwrites).
    pub fn submit_latest(&self, jpeg: Vec<u8>) -> u64 {
        let id = self.frame_counter.fetch_add(1, Ordering::Relaxed) + 1;
        self.cap_in.push(CapIn { id, jpeg });
        id
    }

    /// Non-blocking take of the newest annotated result.
    pub fn try_recv(&self) -> Option<PipelineOutput> {
        self.result_slot.try_take()
    }

    /// Wait briefly for a result (polling).
    pub fn recv_timeout(&self, dur: Duration) -> Option<PipelineOutput> {
        let deadline = Instant::now() + dur;
        loop {
            if let Some(v) = self.try_recv() {
                return Some(v);
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    pub fn completed_frames(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }

    // Compatibility aliases used by older call sites / benchmark.
    pub fn submit_capture(&self, jpeg: Vec<u8>) -> u64 {
        self.submit_latest(jpeg)
    }

    pub fn submit_capture_ok(&self, jpeg: Vec<u8>) -> bool {
        self.submit_latest(jpeg);
        true
    }

    pub fn submit_capture_blocking(&self, jpeg: Vec<u8>) -> u64 {
        self.submit_latest(jpeg)
    }
}

impl Drop for DetectionPipeline {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.cap_in.close();
        self.pre_in.close();
        self.inf_in.close();
        self.ren_in.close();
        self.result_slot.close();
        // Wake threads stuck in take_wait by pushing dummy-clear + close already done
        if let Ok(mut joins) = self.joins.lock() {
            for h in joins.drain(..) {
                let _ = h.join();
            }
        }
    }
}

/// Sequential vs multi-thread pipeline FPS comparison (throughput).
pub fn run_benchmark(model_path: &std::path::Path, frames: u32) -> Result<serde_json::Value> {
    use image::{Rgb, RgbImage};
    use serde_json::json;

    let frames = frames.clamp(10, 300);
    info!("Benchmark: loading model for sequential baseline…");
    let mut seq = ObjectDetector::load(model_path)?;

    let mut test_jpegs: Vec<Vec<u8>> = Vec::with_capacity(frames as usize);
    for i in 0..frames {
        let mut img = RgbImage::from_fn(640, 480, |x, y| {
            let v = ((x + y + i * 3) % 255) as u8;
            Rgb([v, v.wrapping_mul(3), 255u8.wrapping_sub(v)])
        });
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
        threads: vec!["capture+preprocess+inference+render (1 OS thread)".into()],
    };
    drop(seq);

    info!("Benchmark: starting 4-thread latest-frame pipeline…");
    let pipe_det = ObjectDetector::load(model_path)?;
    let pipeline = Arc::new(DetectionPipeline::start(pipe_det)?);
    pipeline.set_live_fast(true);

    for j in test_jpegs.iter().take(5) {
        pipeline.submit_latest(j.clone());
        let _ = pipeline.recv_timeout(Duration::from_millis(500));
    }
    pipeline.clear_pending();

    let before_completed = pipeline.completed_frames();
    let t1 = Instant::now();
    // Feed as fast as possible; latest-frame pipeline may skip some — measure
    // wall-clock of completed outputs over the feed window + drain.
    let jpegs = test_jpegs.clone();
    let prod = pipeline.clone();
    let producer = thread::spawn(move || {
        for j in jpegs {
            prod.submit_latest(j);
            // Small gap simulates ~30 FPS camera
            thread::sleep(Duration::from_millis(5));
        }
    });

    let mut got = 0u32;
    let mut pipe_infer_sum = 0.0f64;
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Some(out) = pipeline.recv_timeout(Duration::from_millis(50)) {
            pipe_infer_sum += out.meta.inference_ms as f64;
            got += 1;
        } else if producer.is_finished() && got > 0 {
            // drain a bit more
            if pipeline.recv_timeout(Duration::from_millis(100)).is_none() {
                break;
            }
        }
        if producer.is_finished() && got >= frames.saturating_sub(5) {
            // latest-frame may complete fewer than submitted
            if pipeline.try_recv().is_none() {
                break;
            }
        }
    }
    let _ = producer.join();
    while let Some(out) = pipeline.try_recv() {
        pipe_infer_sum += out.meta.inference_ms as f64;
        got += 1;
    }
    let pipe_elapsed = t1.elapsed().as_secs_f64().max(1e-6);
    let pipe_fps = got as f64 / pipe_elapsed;
    let after_completed = pipeline.completed_frames() - before_completed;

    let pipe_stats = PipelineStats {
        mode: "multi_thread_latest_frame_pipeline".into(),
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
            "after = 4 OS threads with latest-frame slots (stale frames dropped)",
            "Live path optimizes for latency + sustained FPS, not processing every frame",
            "FPS = completed outputs / wall-clock (camera-tracking throughput)"
        ]
    }))
}
