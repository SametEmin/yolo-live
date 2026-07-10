//! Multi-object Kalman filter tracker for upload-video detection.
//!
//! Per-track state (constant velocity on box center + size):
//!   x = [cx, cy, w, h, vx, vy]^T
//!
//! Each processed frame:
//!   1. Predict all tracks (motion model)
//!   2. Associate YOLO detections via greedy IoU (same class)
//!   3. Update matched tracks with the Kalman measurement update
//!   4. Coast unmatched tracks (keep box visible through brief misses)
//!   5. Spawn tracks for new detections
//!
//! This prevents boxes from blinking off when the model briefly fails
//! on the same object.

use crate::detector::Detection;

const STATE_DIM: usize = 6; // cx, cy, w, h, vx, vy
const MEAS_DIM: usize = 4; // cx, cy, w, h

#[derive(Clone)]
struct KalmanBox {
    /// State mean
    x: [f32; STATE_DIM],
    /// State covariance (row-major 6x6)
    p: [[f32; STATE_DIM]; STATE_DIM],
}

impl KalmanBox {
    fn from_measurement(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let mut p = [[0.0f32; STATE_DIM]; STATE_DIM];
        // High initial uncertainty on velocity
        for i in 0..4 {
            p[i][i] = 0.05;
        }
        p[4][4] = 1.0;
        p[5][5] = 1.0;
        Self {
            x: [cx, cy, w.max(1e-3), h.max(1e-3), 0.0, 0.0],
            p,
        }
    }

    fn predict(&mut self, dt: f32) {
        // F: constant velocity
        // cx' = cx + vx*dt, cy' = cy + vy*dt, w'=w, h'=h, v'=v
        let dt = dt.max(1e-3);
        self.x[0] += self.x[4] * dt;
        self.x[1] += self.x[5] * dt;
        // Clamp center into image, keep size positive
        self.x[2] = self.x[2].max(1e-3);
        self.x[3] = self.x[3].max(1e-3);
        self.x[0] = self.x[0].clamp(0.0, 1.0);
        self.x[1] = self.x[1].clamp(0.0, 1.0);

        // P = F P F^T + Q  (F only couples pos with vel)
        // Process noise Q
        let q_pos = 1e-4_f32;
        let q_size = 5e-5_f32;
        let q_vel = 5e-4_f32;

        // Manual FPF^T for our sparse F
        let p = self.p;
        let mut np = [[0.0f32; STATE_DIM]; STATE_DIM];
        // For i,j in positions involving velocity coupling:
        // State indices: 0:cx 1:cy 2:w 3:h 4:vx 5:vy
        // F row0: [1,0,0,0,dt,0]
        // F row1: [0,1,0,0,0,dt]
        // F row2: [0,0,1,0,0,0]
        // F row3: [0,0,0,1,0,0]
        // F row4: [0,0,0,0,1,0]
        // F row5: [0,0,0,0,0,1]

        // Compute F P (temporary) then (FP) F^T
        let mut fp = [[0.0f32; STATE_DIM]; STATE_DIM];
        for j in 0..STATE_DIM {
            fp[0][j] = p[0][j] + dt * p[4][j];
            fp[1][j] = p[1][j] + dt * p[5][j];
            fp[2][j] = p[2][j];
            fp[3][j] = p[3][j];
            fp[4][j] = p[4][j];
            fp[5][j] = p[5][j];
        }
        for i in 0..STATE_DIM {
            np[i][0] = fp[i][0] + dt * fp[i][4];
            np[i][1] = fp[i][1] + dt * fp[i][5];
            np[i][2] = fp[i][2];
            np[i][3] = fp[i][3];
            np[i][4] = fp[i][4];
            np[i][5] = fp[i][5];
        }
        // Add Q
        np[0][0] += q_pos;
        np[1][1] += q_pos;
        np[2][2] += q_size;
        np[3][3] += q_size;
        np[4][4] += q_vel;
        np[5][5] += q_vel;
        self.p = np;
    }

    fn update(&mut self, cx: f32, cy: f32, w: f32, h: f32) {
        let z = [cx, cy, w.max(1e-3), h.max(1e-3)];
        // H picks first 4 state components
        // Innovation y = z - H x
        let mut y = [0.0f32; MEAS_DIM];
        for i in 0..MEAS_DIM {
            y[i] = z[i] - self.x[i];
        }

        // Measurement noise R (diagonal) — YOLO noise
        let r = [2e-3_f32, 2e-3, 4e-3, 4e-3];

        // S = H P H^T + R  → top-left 4x4 of P + R
        let mut s = [[0.0f32; MEAS_DIM]; MEAS_DIM];
        for i in 0..MEAS_DIM {
            for j in 0..MEAS_DIM {
                s[i][j] = self.p[i][j];
            }
            s[i][i] += r[i];
        }

        // Invert S (4x4) via Gauss-Jordan
        let s_inv = match invert4(s) {
            Some(m) => m,
            None => return,
        };

        // K = P H^T S^{-1}  → 6x4: rows use P[i][0..4] * S_inv
        let mut k = [[0.0f32; MEAS_DIM]; STATE_DIM];
        for i in 0..STATE_DIM {
            for j in 0..MEAS_DIM {
                let mut sum = 0.0;
                for t in 0..MEAS_DIM {
                    sum += self.p[i][t] * s_inv[t][j];
                }
                k[i][j] = sum;
            }
        }

        // x = x + K y
        for i in 0..STATE_DIM {
            let mut sum = 0.0;
            for j in 0..MEAS_DIM {
                sum += k[i][j] * y[j];
            }
            self.x[i] += sum;
        }
        self.x[2] = self.x[2].max(1e-3);
        self.x[3] = self.x[3].max(1e-3);
        self.x[0] = self.x[0].clamp(0.0, 1.0);
        self.x[1] = self.x[1].clamp(0.0, 1.0);

        // P = (I - K H) P
        // KH is 6x6 with only first 4 columns of K affecting first 4 state parts
        let p = self.p;
        let mut khp = [[0.0f32; STATE_DIM]; STATE_DIM];
        // (I - KH) has: row i: e_i - K[i, 0..4] on first 4 cols
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                // (I - K H) * P  with H = [I_4 | 0]
                let mut sum = 0.0;
                for tt in 0..STATE_DIM {
                    let mut m = if i == tt { 1.0 } else { 0.0 };
                    if tt < MEAS_DIM {
                        m -= k[i][tt];
                    }
                    sum += m * p[tt][j];
                }
                khp[i][j] = sum;
            }
        }
        self.p = khp;
        // Symmetrize & keep PD-ish
        for i in 0..STATE_DIM {
            for j in 0..i {
                let v = 0.5 * (self.p[i][j] + self.p[j][i]);
                self.p[i][j] = v;
                self.p[j][i] = v;
            }
            self.p[i][i] = self.p[i][i].max(1e-6);
        }
    }

    fn bbox_xyxy(&self) -> [f32; 4] {
        let (cx, cy, w, h) = (self.x[0], self.x[1], self.x[2].max(1e-3), self.x[3].max(1e-3));
        [
            (cx - w * 0.5).clamp(0.0, 1.0),
            (cy - h * 0.5).clamp(0.0, 1.0),
            (cx + w * 0.5).clamp(0.0, 1.0),
            (cy + h * 0.5).clamp(0.0, 1.0),
        ]
    }
}

fn invert4(mut a: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0.0f32; 4]; 4];
    for i in 0..4 {
        inv[i][i] = 1.0;
    }
    for col in 0..4 {
        // Pivot
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..4 {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-8 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            inv.swap(col, piv);
        }
        let diag = a[col][col];
        for j in 0..4 {
            a[col][j] /= diag;
            inv[col][j] /= diag;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            for j in 0..4 {
                a[r][j] -= f * a[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    Some(inv)
}

#[derive(Clone)]
struct Track {
    id: u64,
    label: String,
    color: String,
    kf: KalmanBox,
    conf: f32,
    time_since_update: u32,
    hits: u32,
    age: u32,
}

/// Multi-object Kalman tracker for continuous boxes on upload video.
#[derive(Clone)]
pub struct KalmanBoxTracker {
    tracks: Vec<Track>,
    next_id: u64,
    iou_thresh: f32,
    /// Frames to keep after last match (at process FPS)
    max_age: u32,
    /// Coast display window
    max_coast_display: u32,
    min_hits: u32,
    min_conf: f32,
    /// Time step between processed frames (1.0 = one process frame)
    dt: f32,
}

impl Default for KalmanBoxTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanBoxTracker {
    pub fn new() -> Self {
        Self {
            tracks: Vec::with_capacity(32),
            next_id: 1,
            iou_thresh: 0.15,
            // ~1s at 15 FPS process rate
            max_age: 15,
            max_coast_display: 12,
            min_hits: 1,
            min_conf: 0.35,
            dt: 1.0,
        }
    }

    pub fn set_min_conf(&mut self, min_conf: f32) {
        self.min_conf = min_conf.clamp(0.05, 0.95);
    }

    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt.clamp(0.1, 5.0);
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    /// Predict + associate + update; returns boxes to draw this frame.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Detection> {
        let detections: Vec<&Detection> = detections
            .iter()
            .filter(|d| d.confidence + 1e-6 >= self.min_conf)
            .collect();

        // 1) Predict
        for t in &mut self.tracks {
            t.kf.predict(self.dt);
            t.age = t.age.saturating_add(1);
            t.time_since_update = t.time_since_update.saturating_add(1);
        }

        // 2) Associate
        let n_det = detections.len();
        let n_trk = self.tracks.len();
        let mut det_used = vec![false; n_det];
        let mut trk_used = vec![false; n_trk];
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();

        for (ti, t) in self.tracks.iter().enumerate() {
            let tb = t.kf.bbox_xyxy();
            for (di, d) in detections.iter().enumerate() {
                if t.label != d.label {
                    continue;
                }
                let iou = iou_xyxy(&tb, &d.bbox);
                if iou >= self.iou_thresh {
                    pairs.push((iou, ti, di));
                } else {
                    let dist = center_dist(&tb, &d.bbox);
                    if dist < 0.15 {
                        pairs.push((0.05 + (0.15 - dist), ti, di));
                    }
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_s, ti, di) in pairs {
            if trk_used[ti] || det_used[di] {
                continue;
            }
            trk_used[ti] = true;
            det_used[di] = true;
            let d = detections[di];
            let (cx, cy, w, h) = xyxy_to_cxcywh(d.bbox);
            self.tracks[ti].kf.update(cx, cy, w, h);
            // Prefer stronger real detector scores (no artificial decay).
            let blended = 0.3 * self.tracks[ti].conf + 0.7 * d.confidence;
            self.tracks[ti].conf = blended.max(d.confidence).max(self.min_conf);
            self.tracks[ti].color = d.color.clone();
            self.tracks[ti].time_since_update = 0;
            self.tracks[ti].hits = self.tracks[ti].hits.saturating_add(1);
        }

        // 3) New tracks
        for (di, d) in detections.iter().enumerate() {
            if det_used[di] {
                continue;
            }
            let (cx, cy, w, h) = xyxy_to_cxcywh(d.bbox);
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.tracks.push(Track {
                id,
                label: d.label.clone(),
                color: d.color.clone(),
                kf: KalmanBox::from_measurement(cx, cy, w, h),
                conf: d.confidence,
                time_since_update: 0,
                hits: 1,
                age: 1,
            });
        }

        // 4) Drop old
        self.tracks
            .retain(|t| t.time_since_update <= self.max_age);

        // 5) Emit — coast through brief misses so boxes don't blink off
        let mut out = Vec::with_capacity(self.tracks.len());
        for t in &self.tracks {
            if t.conf + 1e-6 < self.min_conf {
                continue;
            }
            if t.hits < self.min_hits {
                continue;
            }
            // Keep showing while recently matched OR coasting within window
            if t.time_since_update > self.max_coast_display {
                continue;
            }
            out.push(Detection {
                label: t.label.clone(),
                confidence: (t.conf * 1000.0).round() / 1000.0,
                bbox: t.kf.bbox_xyxy(),
                color: t.color.clone(),
            });
        }
        out
    }
}

fn xyxy_to_cxcywh(b: [f32; 4]) -> (f32, f32, f32, f32) {
    let w = (b[2] - b[0]).max(1e-4);
    let h = (b[3] - b[1]).max(1e-4);
    (b[0] + w * 0.5, b[1] + h * 0.5, w, h)
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
