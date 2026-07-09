//! YOLO11n ONNX detector optimized for Apple Silicon via CoreML EP.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ab_glyph::{FontArc, PxScale};
use anyhow::{Context, Result};
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use tracing::info;

use crate::coco::{self, COCO_NAMES};

const INPUT_SIZE: u32 = 640;
const MODEL_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx";

/// Convert ort errors that embed non-Send types into anyhow.
fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
    pub color: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LabelSummary {
    pub label: String,
    pub count: usize,
    pub avg_confidence: f32,
    pub color: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FrameResult {
    pub detections: Vec<Detection>,
    pub summary: Vec<LabelSummary>,
    pub fps: f32,
    pub inference_ms: f32,
    pub device: String,
    pub count: usize,
}

pub struct ObjectDetector {
    session: Session,
    conf: f32,
    iou: f32,
    device: String,
    font: FontArc,
    last_instant: Instant,
    fps: f32,
}

impl ObjectDetector {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = ensure_model(model_path.as_ref())?;
        info!("Loading YOLO model from {}", path.display());

        let mut device = "cpu".to_string();

        let session = {
            #[cfg(target_os = "macos")]
            {
                use ort::ep::CoreML;
                let coreml_try = Session::builder()
                    .map_err(ort_err)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(ort_err)?
                    .with_intra_threads(4)
                    .map_err(ort_err)?
                    .with_execution_providers([CoreML::default().build()])
                    .map_err(ort_err)
                    .and_then(|mut b| b.commit_from_file(&path).map_err(ort_err));

                match coreml_try {
                    Ok(s) => {
                        device = "coreml".to_string();
                        info!("CoreML execution provider enabled (Apple Silicon / M4)");
                        s
                    }
                    Err(e) => {
                        tracing::warn!("CoreML unavailable ({e}), falling back to CPU");
                        Session::builder()
                            .map_err(ort_err)?
                            .with_optimization_level(GraphOptimizationLevel::Level3)
                            .map_err(ort_err)?
                            .with_intra_threads(4)
                            .map_err(ort_err)?
                            .commit_from_file(&path)
                            .map_err(ort_err)?
                    }
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                Session::builder()
                    .map_err(ort_err)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(ort_err)?
                    .with_intra_threads(4)
                    .map_err(ort_err)?
                    .commit_from_file(&path)
                    .map_err(ort_err)?
            }
        };

        let font = FontArc::try_from_slice(include_bytes!("../assets/DejaVuSans.ttf"))
            .map_err(|e| anyhow::anyhow!("embed font: {e}"))?;

        let mut det = Self {
            session,
            conf: 0.35,
            iou: 0.45,
            device,
            font,
            last_instant: Instant::now(),
            fps: 0.0,
        };

        let warm = RgbImage::new(640, 480);
        let _ = det.predict_image(&DynamicImage::ImageRgb8(warm))?;
        info!("YOLO warm-up complete on {}", det.device);

        Ok(det)
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn model_name(&self) -> &str {
        "yolo11n.onnx"
    }

    pub fn confidence(&self) -> f32 {
        self.conf
    }

    pub fn set_confidence(&mut self, conf: f32) {
        self.conf = conf.clamp(0.05, 0.95);
    }

    pub fn font(&self) -> &FontArc {
        &self.font
    }

    /// Single-threaded end-to-end (baseline): capture → preprocess → infer → render.
    pub fn predict_jpeg(&mut self, jpeg_bytes: &[u8]) -> Result<(Vec<u8>, FrameResult)> {
        let rgb = stage_capture_decode(jpeg_bytes)?;
        let prepared = stage_preprocess(rgb);
        let (dets, inference_ms) = self.stage_inference(&prepared)?;
        self.note_fps();
        stage_render(
            prepared.rgb,
            &dets,
            &self.font,
            &self.device,
            inference_ms,
            self.fps,
        )
    }

    pub fn predict_image(&mut self, img: &DynamicImage) -> Result<(DynamicImage, FrameResult)> {
        let rgb = img.to_rgb8();
        let prepared = stage_preprocess(rgb);
        let (dets, inference_ms) = self.stage_inference(&prepared)?;
        self.note_fps();
        let (jpeg, meta) = stage_render(
            prepared.rgb,
            &dets,
            &self.font,
            &self.device,
            inference_ms,
            self.fps,
        )?;
        let annotated = image::load_from_memory(&jpeg).context("reload annotated")?;
        let _ = jpeg;
        Ok((annotated, meta))
    }

    /// Stage 3 — Inference (must run on the thread that owns the ONNX session).
    pub fn stage_inference(
        &mut self,
        prepared: &PreprocessedFrame,
    ) -> Result<(Vec<Detection>, f32)> {
        let t0 = Instant::now();
        let shape = [1i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let input_ref = TensorRef::from_array_view((
            shape,
            prepared.tensor.as_slice().unwrap(),
        ))
        .map_err(ort_err)?;
        let outputs = self.session.run(ort::inputs![input_ref]).map_err(ort_err)?;
        let inference_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let (_k, output) = outputs.iter().next().context("no model outputs")?;
        let (out_shape, data) = output.try_extract_tensor::<f32>().map_err(ort_err)?;
        let shape_vec: Vec<usize> = out_shape.iter().map(|d| *d as usize).collect();
        let detections = postprocess(
            data,
            &shape_vec,
            &prepared.letterbox,
            prepared.orig_w,
            prepared.orig_h,
            self.conf,
            self.iou,
        )?;
        Ok((detections, inference_ms))
    }

    fn note_fps(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_instant).as_secs_f32();
        self.last_instant = now;
        if dt > 0.0 {
            let instant = 1.0 / dt;
            self.fps = if self.fps <= 0.0 {
                instant
            } else {
                0.85 * self.fps + 0.15 * instant
            };
        }
    }
}

#[derive(Debug, Clone)]
pub struct LetterboxMeta {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Output of the preprocessing stage (letterbox + NCHW tensor).
pub struct PreprocessedFrame {
    pub rgb: RgbImage,
    pub tensor: Array4<f32>,
    pub letterbox: LetterboxMeta,
    pub orig_w: u32,
    pub orig_h: u32,
}

// ─── Pipeline stages (used by multi-threaded pipeline) ─────────────────────

/// Stage 1 — Capture: decode camera / network JPEG bytes to RGB.
pub fn stage_capture_decode(jpeg_bytes: &[u8]) -> Result<RgbImage> {
    let img = image::load_from_memory(jpeg_bytes).context("decode jpeg")?;
    Ok(img.to_rgb8())
}

/// Stage 2 — Preprocessing: letterbox resize + normalize to NCHW float tensor.
pub fn stage_preprocess(rgb: RgbImage) -> PreprocessedFrame {
    stage_preprocess_with_filter(rgb, image::imageops::FilterType::Triangle)
}

/// Faster preprocess for live (Nearest filter) — targets 20+ FPS with low latency.
pub fn stage_preprocess_fast(rgb: RgbImage) -> PreprocessedFrame {
    stage_preprocess_with_filter(rgb, image::imageops::FilterType::Nearest)
}

fn stage_preprocess_with_filter(
    rgb: RgbImage,
    filter: image::imageops::FilterType,
) -> PreprocessedFrame {
    let (orig_w, orig_h) = (rgb.width(), rgb.height());
    let (input, letterbox) = letterbox_with_filter(&rgb, INPUT_SIZE, filter);
    let tensor = Array4::from_shape_fn(
        (1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize),
        |(_, c, y, x)| input[(x as u32, y as u32)][c] as f32 / 255.0,
    );
    PreprocessedFrame {
        rgb,
        tensor,
        letterbox,
        orig_w,
        orig_h,
    }
}

/// Stage 4 — Rendering: draw boxes and encode JPEG (does not need the session).
pub fn stage_render(
    rgb: RgbImage,
    detections: &[Detection],
    font: &FontArc,
    device: &str,
    inference_ms: f32,
    pipeline_fps: f32,
) -> Result<(Vec<u8>, FrameResult)> {
    stage_render_quality(rgb, detections, font, device, inference_ms, pipeline_fps, 80)
}

/// Live path: lower JPEG quality for faster encode / smaller WS payloads.
pub fn stage_render_fast(
    rgb: RgbImage,
    detections: &[Detection],
    font: &FontArc,
    device: &str,
    inference_ms: f32,
    pipeline_fps: f32,
) -> Result<(Vec<u8>, FrameResult)> {
    stage_render_quality(rgb, detections, font, device, inference_ms, pipeline_fps, 65)
}

fn stage_render_quality(
    mut rgb: RgbImage,
    detections: &[Detection],
    font: &FontArc,
    device: &str,
    inference_ms: f32,
    pipeline_fps: f32,
    jpeg_quality: u8,
) -> Result<(Vec<u8>, FrameResult)> {
    draw_detections(&mut rgb, detections, font);
    let summary = summarize(detections);
    let count = detections.len();
    let result = FrameResult {
        detections: detections.to_vec(),
        summary,
        fps: (pipeline_fps * 10.0).round() / 10.0,
        inference_ms: (inference_ms * 10.0).round() / 10.0,
        device: device.to_string(),
        count,
    };
    let mut buf = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut buf);
        let mut encoder =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
        encoder
            .encode(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                image::ExtendedColorType::Rgb8,
            )
            .context("encode jpeg")?;
    }
    Ok((buf, result))
}

fn letterbox_with_filter(
    img: &RgbImage,
    size: u32,
    filter: image::imageops::FilterType,
) -> (RgbImage, LetterboxMeta) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let scale = (size as f32 / w).min(size as f32 / h);
    let nw = (w * scale).round().max(1.0) as u32;
    let nh = (h * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(img, nw, nh, filter);

    let mut out = RgbImage::from_pixel(size, size, Rgb([114, 114, 114]));
    let pad_x = (size - nw) / 2;
    let pad_y = (size - nh) / 2;
    image::imageops::replace(&mut out, &resized, pad_x as i64, pad_y as i64);

    (
        out,
        LetterboxMeta {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        },
    )
}

fn postprocess(
    data: &[f32],
    shape: &[usize],
    meta: &LetterboxMeta,
    orig_w: u32,
    orig_h: u32,
    conf_thresh: f32,
    iou_thresh: f32,
) -> Result<Vec<Detection>> {
    let (num_attrs, num_preds, transposed) = if shape.len() == 3 {
        if shape[1] <= shape[2] {
            (shape[1], shape[2], false)
        } else {
            (shape[2], shape[1], true)
        }
    } else if shape.len() == 2 {
        if shape[0] <= shape[1] {
            (shape[0], shape[1], false)
        } else {
            (shape[1], shape[0], true)
        }
    } else {
        anyhow::bail!("unexpected output shape: {shape:?}");
    };

    let num_classes = num_attrs.saturating_sub(4);
    let mut candidates: Vec<(f32, usize, [f32; 4])> = Vec::new();

    for i in 0..num_preds {
        let get = |a: usize| -> f32 {
            if transposed {
                data[i * num_attrs + a]
            } else {
                data[a * num_preds + i]
            }
        };

        let cx = get(0);
        let cy = get(1);
        let bw = get(2);
        let bh = get(3);

        let mut best_cls = 0usize;
        let mut best_score = 0.0f32;
        for c in 0..num_classes {
            let s = get(4 + c);
            if s > best_score {
                best_score = s;
                best_cls = c;
            }
        }
        if best_score < conf_thresh {
            continue;
        }

        let x1 = cx - bw / 2.0;
        let y1 = cy - bh / 2.0;
        let x2 = cx + bw / 2.0;
        let y2 = cy + bh / 2.0;
        candidates.push((best_score, best_cls, [x1, y1, x2, y2]));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let kept = nms(candidates, iou_thresh);

    let mut out = Vec::with_capacity(kept.len());
    for (score, cls, box_lb) in kept {
        let mut x1 = (box_lb[0] - meta.pad_x) / meta.scale;
        let mut y1 = (box_lb[1] - meta.pad_y) / meta.scale;
        let mut x2 = (box_lb[2] - meta.pad_x) / meta.scale;
        let mut y2 = (box_lb[3] - meta.pad_y) / meta.scale;

        x1 = x1.clamp(0.0, orig_w as f32);
        y1 = y1.clamp(0.0, orig_h as f32);
        x2 = x2.clamp(0.0, orig_w as f32);
        y2 = y2.clamp(0.0, orig_h as f32);

        let label = COCO_NAMES
            .get(cls)
            .copied()
            .unwrap_or("object")
            .to_string();

        out.push(Detection {
            label,
            confidence: (score * 1000.0).round() / 1000.0,
            bbox: [
                x1 / orig_w as f32,
                y1 / orig_h as f32,
                x2 / orig_w as f32,
                y2 / orig_h as f32,
            ],
            color: coco::hex_color(cls),
        });
    }
    Ok(out)
}

fn nms(mut dets: Vec<(f32, usize, [f32; 4])>, iou_thresh: f32) -> Vec<(f32, usize, [f32; 4])> {
    let mut keep = Vec::new();
    while let Some(best) = dets.first().cloned() {
        keep.push(best);
        dets.remove(0);
        dets.retain(|d| {
            if d.1 != best.1 {
                return true;
            }
            iou(&best.2, &d.2) < iou_thresh
        });
    }
    keep
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}

fn summarize(dets: &[Detection]) -> Vec<LabelSummary> {
    let mut map: HashMap<String, (usize, f32, String)> = HashMap::new();
    for d in dets {
        let e = map
            .entry(d.label.clone())
            .or_insert((0, 0.0, d.color.clone()));
        e.0 += 1;
        e.1 += d.confidence;
    }
    let mut v: Vec<LabelSummary> = map
        .into_iter()
        .map(|(label, (count, sum, color))| LabelSummary {
            label,
            count,
            avg_confidence: ((sum / count as f32) * 1000.0).round() / 1000.0,
            color,
        })
        .collect();
    v.sort_by(|a, b| b.count.cmp(&a.count).then(a.label.cmp(&b.label)));
    v
}

fn draw_detections(img: &mut RgbImage, dets: &[Detection], font: &FontArc) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let thickness = ((w.min(h) / 400.0).round() as i32).max(2);
    let scale = PxScale::from((w.min(h) / 35.0).clamp(14.0, 28.0));

    for d in dets {
        let x1 = (d.bbox[0] * w) as i32;
        let y1 = (d.bbox[1] * h) as i32;
        let x2 = (d.bbox[2] * w) as i32;
        let y2 = (d.bbox[3] * h) as i32;
        let bw = (x2 - x1).max(1) as u32;
        let bh = (y2 - y1).max(1) as u32;

        let hex = d.color.trim_start_matches('#');
        let r = u8::from_str_radix(hex.get(0..2).unwrap_or("ff"), 16).unwrap_or(255);
        let g = u8::from_str_radix(hex.get(2..4).unwrap_or("ff"), 16).unwrap_or(255);
        let b = u8::from_str_radix(hex.get(4..6).unwrap_or("00"), 16).unwrap_or(0);
        let color = Rgb([r, g, b]);

        for t in 0..thickness {
            let ww = bw.saturating_sub((t * 2) as u32);
            let hh = bh.saturating_sub((t * 2) as u32);
            if ww > 0 && hh > 0 {
                let rect = Rect::at(x1 + t, y1 + t).of_size(ww, hh);
                draw_hollow_rect_mut(img, rect, color);
            }
        }

        let text = format!("{} {:.0}%", d.label, d.confidence * 100.0);
        let tx = x1.max(0);
        let ty = (y1 - (scale.y as i32) - 4).max(0);
        let tw = ((text.len() as f32) * scale.x * 0.55) as u32 + 8;
        let th = scale.y as u32 + 6;
        let bg = Rect::at(tx, ty).of_size(tw.max(1), th.max(1));
        draw_filled_rect_mut(img, bg, color);
        draw_text_mut(img, Rgb([0, 0, 0]), tx + 4, ty + 2, scale, font, &text);
    }
}

fn ensure_model(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    info!("Downloading YOLO model from {MODEL_URL}");
    let resp = reqwest::blocking::get(MODEL_URL)?.error_for_status()?;
    let bytes = resp.bytes()?;
    std::fs::write(path, &bytes)?;
    info!("Saved model to {}", path.display());
    Ok(path.to_path_buf())
}
