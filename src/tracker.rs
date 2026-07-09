//! Smooth multi-object box tracking for stable live overlays.
//!
//! Pipeline (very cheap, does not affect inference FPS):
//! 1. Predict each track with constant-velocity model
//! 2. Associate detections ↔ tracks by label + IoU (greedy)
//! 3. Update matched tracks with EMA blended measurement + prediction
//! 4. Coast unmatched tracks briefly; spawn new tracks for new detections
//!
//! This removes jittery box jumps between frames while keeping full throughput.

use crate::detector::Detection;

#[derive(Clone)]
struct Track {
    id: u64,
    label: String,
    color: String,
    /// Smoothed xyxy (normalized 0..1)
    bbox: [f32; 4],
    /// Velocity of xyxy per frame
    vel: [f32; 4],
    conf: f32,
    /// Frames since last successful match
    time_since_update: u32,
    /// Total successful matches
    hits: u32,
    age: u32,
}

/// Online box smoother / tracker.
pub struct BoxTracker {
    tracks: Vec<Track>,
    next_id: u64,
    /// Measurement blend for center (0 = pure predict/smooth, 1 = pure measure)
    alpha_center: f32,
    /// Measurement blend for size
    alpha_size: f32,
    /// Velocity EMA
    alpha_vel: f32,
    iou_thresh: f32,
    /// Keep coasting after miss
    max_age: u32,
    /// Min hits before emitting a box (reduces flicker of one-frame FPs)
    min_hits: u32,
}

impl Default for BoxTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BoxTracker {
    pub fn new() -> Self {
        Self {
            tracks: Vec::with_capacity(32),
            next_id: 1,
            // Tuned for smooth live boxes without rubber-banding lag
            // Lower alpha => smoother (less jitter), still responsive.
            alpha_center: 0.28,
            alpha_size: 0.32,
            alpha_vel: 0.40,
            iou_thresh: 0.12,
            max_age: 10,
            min_hits: 1,
        }
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    /// Update tracker with raw detections; returns smoothed boxes to draw.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Detection> {
        // 1) Predict
        for t in &mut self.tracks {
            for i in 0..4 {
                t.bbox[i] = (t.bbox[i] + t.vel[i]).clamp(0.0, 1.0);
            }
            // Keep box ordered
            if t.bbox[0] > t.bbox[2] {
                t.bbox.swap(0, 2);
            }
            if t.bbox[1] > t.bbox[3] {
                t.bbox.swap(1, 3);
            }
            t.age = t.age.saturating_add(1);
            t.time_since_update = t.time_since_update.saturating_add(1);
        }

        // 2) Greedy association: same label, highest IoU first
        let n_det = detections.len();
        let n_trk = self.tracks.len();
        let mut det_used = vec![false; n_det];
        let mut trk_used = vec![false; n_trk];

        // Build candidate pairs
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            for (di, d) in detections.iter().enumerate() {
                if t.label != d.label {
                    continue;
                }
                let iou = iou_xyxy(&t.bbox, &d.bbox);
                if iou >= self.iou_thresh {
                    pairs.push((iou, ti, di));
                } else {
                    // Also allow center-distance match for fast motion / low IoU
                    let dist = center_dist(&t.bbox, &d.bbox);
                    if dist < 0.12 {
                        pairs.push((0.05 + (0.12 - dist), ti, di));
                    }
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_score, ti, di) in pairs {
            if trk_used[ti] || det_used[di] {
                continue;
            }
            trk_used[ti] = true;
            det_used[di] = true;
            self.apply_measurement(ti, &detections[di]);
        }

        // 3) New tracks for unmatched detections
        for (di, d) in detections.iter().enumerate() {
            if det_used[di] {
                continue;
            }
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.tracks.push(Track {
                id,
                label: d.label.clone(),
                color: d.color.clone(),
                bbox: d.bbox,
                vel: [0.0; 4],
                conf: d.confidence,
                time_since_update: 0,
                hits: 1,
                age: 1,
            });
        }

        // 4) Drop stale tracks
        self.tracks
            .retain(|t| t.time_since_update <= self.max_age);

        // 5) Emit active tracks (matched recently or confirmed)
        let mut out = Vec::with_capacity(self.tracks.len());
        for t in &self.tracks {
            // Show if recently updated, or still coasting with enough history
            let visible = t.hits >= self.min_hits
                && (t.time_since_update <= 3 || t.hits >= 3);
            if !visible {
                continue;
            }
            // Confidence fades slightly while coasting
            let conf = if t.time_since_update == 0 {
                t.conf
            } else {
                (t.conf * (1.0 - 0.08 * t.time_since_update as f32)).max(0.15)
            };
            out.push(Detection {
                label: t.label.clone(),
                confidence: (conf * 1000.0).round() / 1000.0,
                bbox: t.bbox,
                color: t.color.clone(),
            });
        }
        out
    }

    fn apply_measurement(&mut self, ti: usize, d: &Detection) {
        let t = &mut self.tracks[ti];
        let prev = t.bbox;
        let meas = d.bbox;

        // Convert to center/size for more natural smoothing
        let (pcx, pcy, pw, ph) = xyxy_to_cxcywh(prev);
        let (mcx, mcy, mw, mh) = xyxy_to_cxcywh(meas);

        // Predicted center already advanced in predict step; blend measurement
        let ac = self.alpha_center;
        let asz = self.alpha_size;
        let cx = (1.0 - ac) * pcx + ac * mcx;
        let cy = (1.0 - ac) * pcy + ac * mcy;
        let w = (1.0 - asz) * pw + asz * mw;
        let h = (1.0 - asz) * ph + asz * mh;

        let new_bbox = cxcywh_to_xyxy(cx, cy, w, h);

        // Velocity update from measurement residual
        let av = self.alpha_vel;
        for i in 0..4 {
            let meas_vel = meas[i] - prev[i];
            t.vel[i] = (1.0 - av) * t.vel[i] + av * meas_vel;
            // Dampen velocity to avoid overshoot
            t.vel[i] *= 0.85;
        }

        t.bbox = new_bbox;
        t.conf = 0.6 * t.conf + 0.4 * d.confidence;
        t.color = d.color.clone();
        t.time_since_update = 0;
        t.hits = t.hits.saturating_add(1);
    }
}

fn xyxy_to_cxcywh(b: [f32; 4]) -> (f32, f32, f32, f32) {
    let w = (b[2] - b[0]).max(1e-4);
    let h = (b[3] - b[1]).max(1e-4);
    let cx = b[0] + w * 0.5;
    let cy = b[1] + h * 0.5;
    (cx, cy, w, h)
}

fn cxcywh_to_xyxy(cx: f32, cy: f32, w: f32, h: f32) -> [f32; 4] {
    let w = w.max(1e-4);
    let h = h.max(1e-4);
    [
        (cx - w * 0.5).clamp(0.0, 1.0),
        (cy - h * 0.5).clamp(0.0, 1.0),
        (cx + w * 0.5).clamp(0.0, 1.0),
        (cy + h * 0.5).clamp(0.0, 1.0),
    ]
}

fn iou_xyxy(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    inter / (area_a + area_b - inter + 1e-6)
}

fn center_dist(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (acx, acy, _, _) = xyxy_to_cxcywh(*a);
    let (bcx, bcy, _, _) = xyxy_to_cxcywh(*b);
    let dx = acx - bcx;
    let dy = acy - bcy;
    (dx * dx + dy * dy).sqrt()
}
