(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    btnLive: $("btnLive"),
    btnUpload: $("btnUpload"),
    btnStart: $("btnStart"),
    btnPause: $("btnPause"),
    btnResume: $("btnResume"),
    btnStop: $("btnStop"),
    confSlider: $("confSlider"),
    confValue: $("confValue"),
    webcam: $("webcam"),
    previewVideo: $("previewVideo"),
    captureCanvas: $("captureCanvas"),
    outputFrame: $("outputFrame"),
    placeholder: $("placeholder"),
    placeholderTitle: $("placeholderTitle"),
    placeholderText: $("placeholderText"),
    uploadZone: $("uploadZone"),
    fileInput: $("fileInput"),
    btnPickFile: $("btnPickFile"),
    fileName: $("fileName"),
    downloadBtn: $("downloadBtn"),
    downloadHint: $("downloadHint"),
    summaryList: $("summaryList"),
    instanceList: $("instanceList"),
    labelCount: $("labelCount"),
    fpsValue: $("fpsValue"),
    infValue: $("infValue"),
    objValue: $("objValue"),
    deviceLabel: $("deviceLabel"),
    deviceChip: $("deviceChip"),
    statusText: $("statusText"),
    progressBar: $("progressBar"),
    progressFill: $("progressFill"),
    progressLabel: $("progressLabel"),
    recordPill: $("recordPill"),
    recFrames: $("recFrames"),
    btnBenchmark: $("btnBenchmark"),
    benchResult: $("benchResult"),
  };

  const state = {
    mode: "live",
    running: false,
    paused: false,
    ws: null,
    stream: null,
    loopId: null,
    jobId: null,
    selectedFile: null,
    previewObjectUrl: null,
    liveWarmupLeft: 0,
    busyFrame: false,
    inFlight: 0,
    maxInFlight: 1,
    stopping: false,
    expectedFrames: 0,
    processed: 0,
  };

  let pendingDownloadUrl = null;

  fetch("/api/health")
    .then((r) => r.json())
    .then((d) => {
      const dev = (d.device || "cpu").toLowerCase();
      els.deviceLabel.textContent = dev === "coreml" ? "CoreML · M4" : dev.toUpperCase();
      if (dev === "coreml" || dev === "mps") els.deviceChip.classList.add("live");
      els.statusText.textContent = `Rust · ${d.model} ready`;
    })
    .catch(() => {
      els.deviceLabel.textContent = "offline";
      els.statusText.textContent = "Server unreachable";
    });

  function setMode(mode) {
    if (state.running || state.paused) stopAll(true);
    // Always tear down any leftover upload/live socket so modes cannot interfere.
    closeSocket();
    resetLiveClientState();
    state.mode = mode;
    state.paused = false;
    state.stopping = false;
    els.btnLive.classList.toggle("active", mode === "live");
    els.btnUpload.classList.toggle("active", mode === "upload");
    els.uploadZone.hidden = mode !== "upload";
    els.progressBar.hidden = true;
    hideDownload();
    setRecordingUi(false, 0);
    // Always clear detection frame when switching modes.
    if (els.outputFrame) {
      els.outputFrame.hidden = true;
      els.outputFrame.removeAttribute("src");
    }
    updateControlVisibility();
    if (mode === "live") {
      // Preview belongs only to Upload mode — never keep it on Live.
      clearPreviewVideo();
      els.placeholderTitle.textContent = "Live camera mode";
      els.placeholderText.innerHTML =
        "Click <strong>Start</strong> to open your webcam. Annotated frames are recorded automatically — press <strong>Stop &amp; save</strong> to download the video.";
      els.btnStart.textContent = "Start camera";
      els.btnStop.textContent = "Stop & save";
      showPlaceholder(true);
    } else {
      els.placeholderTitle.textContent = "Upload a video";
      els.placeholderText.innerHTML =
        "Choose a video, then <strong>Start detection</strong>. Use <strong>Pause</strong> / <strong>Resume</strong>, and download a partial annotated video anytime.";
      els.btnStart.textContent = "Start detection";
      els.btnStop.textContent = "Stop & save";
      // Restore preview only in upload mode if a file is already selected.
      if (state.selectedFile) {
        showSelectedVideoPreview(state.selectedFile);
      } else {
        clearPreviewVideo();
        showPlaceholder(true);
      }
    }
  }

  function updateControlVisibility() {
    const upload = state.mode === "upload";
    const active = state.running || state.paused;
    if (els.btnPause) {
      els.btnPause.hidden = !upload;
      els.btnPause.disabled = !state.running || state.paused;
    }
    if (els.btnResume) {
      els.btnResume.hidden = !upload;
      els.btnResume.disabled = !state.paused;
    }
    els.btnStart.disabled = active;
    els.btnStop.disabled = !active && !(state.mode === "live" && state.running);
    if (state.mode === "live") {
      els.btnStop.disabled = !state.running && !state.stopping;
    }
    els.btnLive.disabled = active;
    els.btnUpload.disabled = active;
  }

  els.btnLive.addEventListener("click", () => setMode("live"));
  els.btnUpload.addEventListener("click", () => setMode("upload"));
  els.confSlider.addEventListener("input", () => {
    const v = (Number(els.confSlider.value) / 100).toFixed(2);
    els.confValue.textContent = v;
    if (state.ws && state.ws.readyState === WebSocket.OPEN)
      state.ws.send(JSON.stringify({ type: "config", confidence: Number(v) }));
  });
  els.btnPickFile.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => {
    const f = els.fileInput.files?.[0];
    state.selectedFile = f || null;
    els.fileName.textContent = f ? f.name : "No file selected";
    hideDownload();
    if (state.mode === "upload") {
      showSelectedVideoPreview(f || null);
    } else {
      clearPreviewVideo();
    }
  });
  els.btnStart.addEventListener("click", () =>
    state.mode === "live" ? startLive() : startUpload()
  );
  els.btnStop.addEventListener("click", () => stopAll(false));
  if (els.btnPause) {
    els.btnPause.addEventListener("click", () => {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN || !state.running) return;
      state.ws.send(JSON.stringify({ type: "pause" }));
      els.statusText.textContent = "Pausing…";
      els.btnPause.disabled = true;
    });
  }
  if (els.btnResume) {
    els.btnResume.addEventListener("click", () => {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN || !state.paused) return;
      state.ws.send(JSON.stringify({ type: "resume" }));
      els.statusText.textContent = "Resuming detection…";
      els.btnResume.disabled = true;
    });
  }

  function closeSocket() {
    if (!state.ws) return;
    try {
      state.ws.onopen = null;
      state.ws.onmessage = null;
      state.ws.onerror = null;
      state.ws.onclose = null;
      if (
        state.ws.readyState === WebSocket.OPEN ||
        state.ws.readyState === WebSocket.CONNECTING
      ) {
        try {
          state.ws.send(JSON.stringify({ type: "stop" }));
        } catch (_) {}
        try {
          state.ws.close();
        } catch (_) {}
      }
    } catch (_) {}
    state.ws = null;
  }

  function resetLiveClientState() {
    state.busyFrame = false;
    state.inFlight = 0;
    state.liveWarmupLeft = 0;
    state.stopping = false;
    if (state.loopId) {
      cancelAnimationFrame(state.loopId);
      state.loopId = null;
    }
  }

  function setRunningFlags({ running = false, paused = false } = {}) {
    state.running = running;
    state.paused = paused;
    if (running) els.deviceChip.classList.add("live");
    updateControlVisibility();
  }

  function showPlaceholder(show) {
    els.placeholder.hidden = !show;
    if (show) {
      els.outputFrame.hidden = true;
      els.outputFrame.removeAttribute("src");
      // Never leave a stuck preview under the placeholder (esp. Live mode).
      if (state.mode !== "upload" && els.previewVideo) {
        els.previewVideo.hidden = true;
        els.previewVideo.pause();
      }
    }
  }

  function showFrame(dataUrl) {
    // Detection frames replace preview in the same stage slot.
    els.placeholder.hidden = true;
    if (els.previewVideo) {
      els.previewVideo.hidden = true;
      els.previewVideo.pause();
    }
    els.outputFrame.hidden = false;
    els.outputFrame.src = dataUrl;
  }

  function clearPreviewVideo() {
    if (!els.previewVideo) return;
    els.previewVideo.pause();
    els.previewVideo.removeAttribute("src");
    try { els.previewVideo.load(); } catch (_) {}
    els.previewVideo.hidden = true;
    if (state.previewObjectUrl) {
      URL.revokeObjectURL(state.previewObjectUrl);
      state.previewObjectUrl = null;
    }
  }

  function showSelectedVideoPreview(file) {
    if (!els.previewVideo) return;
    // Only show uploaded file preview in Upload Video mode.
    if (state.mode !== "upload") {
      clearPreviewVideo();
      return;
    }
    clearPreviewVideo();
    if (!file) {
      showPlaceholder(true);
      return;
    }
    // Same stage area as detection (#stage grid cell).
    const url = URL.createObjectURL(file);
    state.previewObjectUrl = url;
    els.previewVideo.src = url;
    els.previewVideo.hidden = false;
    els.outputFrame.hidden = true;
    els.outputFrame.removeAttribute("src");
    els.placeholder.hidden = true;
    els.previewVideo.currentTime = 0;
    els.previewVideo.play().catch(() => {
      // Autoplay may be blocked; controls allow manual play.
    });
    els.statusText.textContent = `Selected: ${file.name}`;
  }

  function hideDownload() {
    pendingDownloadUrl = null;
    if (els.downloadBtn) {
      els.downloadBtn.hidden = true;
      els.downloadBtn.disabled = false;
      els.downloadBtn.textContent = "Download annotated video";
    }
    if (els.downloadHint) {
      els.downloadHint.hidden = true;
      els.downloadHint.textContent = "";
    }
  }
  function showDownload(url, label) {
    if (!url) return;
    pendingDownloadUrl = url;
    if (els.downloadBtn) {
      els.downloadBtn.hidden = false;
      els.downloadBtn.disabled = false;
      els.downloadBtn.textContent = "Download annotated video";
    }
    if (els.downloadHint) {
      els.downloadHint.hidden = false;
      els.downloadHint.textContent = label || "Ready to download";
    }
  }

  function updateProgress(progress, processed, expected) {
    const p = Math.max(0, Math.min(Number(progress) || 0, 1));
    // Never paint a full bar unless truly complete (progress === 1)
    const pct = p >= 0.999 && p < 1 ? 99 : Math.round(p * 100);
    els.progressBar.hidden = false;
    els.progressFill.style.width = `${pct}%`;
    const exp = expected || state.expectedFrames || 0;
    const proc = processed || state.processed || 0;
    if (exp > 0) {
      els.progressLabel.textContent = `${pct}% · ${proc} / ${exp} frames`;
    } else {
      els.progressLabel.textContent = `${pct}% · frame ${proc}`;
    }
  }

  async function downloadAnnotatedFile() {
    // If still processing, ask server to export current frames first.
    if (
      state.mode === "upload" &&
      state.ws &&
      state.ws.readyState === WebSocket.OPEN &&
      (state.running || state.paused) &&
      state.processed > 0
    ) {
      els.statusText.textContent = "Exporting annotated video so far…";
      if (els.downloadBtn) {
        els.downloadBtn.disabled = true;
        els.downloadBtn.textContent = "Exporting…";
      }
      try {
        state.ws.send(JSON.stringify({ type: "export" }));
      } catch (_) {}
      // export_ready handler will call performDownload
      state._downloadAfterExport = true;
      return;
    }
    await performDownload();
  }

  async function performDownload() {
    if (!pendingDownloadUrl) {
      alert("No annotated video is ready yet. Process at least one frame first.");
      return;
    }
    const btn = els.downloadBtn;
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Preparing download…";
    }
    try {
      const res = await fetch(pendingDownloadUrl);
      if (!res.ok) {
        let msg = `Download failed (HTTP ${res.status})`;
        try {
          const j = await res.json();
          if (j.error) msg = j.error;
        } catch (_) {}
        throw new Error(msg);
      }
      const blob = await res.blob();
      if (!blob || blob.size < 100) throw new Error("Downloaded file is empty");
      const cd = res.headers.get("content-disposition") || "";
      let filename = "yolo_annotated.mp4";
      const m = /filename="?([^";]+)"?/i.exec(cd);
      if (m) filename = m[1];
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 2000);
      els.statusText.textContent = `Downloaded ${filename} (${(blob.size / (1024 * 1024)).toFixed(1)} MB)`;
      if (els.downloadHint) {
        els.downloadHint.hidden = false;
        els.downloadHint.textContent = `Saved ${filename}`;
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Download failed");
      els.statusText.textContent = err.message || "Download failed";
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Download annotated video";
      }
      state._downloadAfterExport = false;
    }
  }

  if (els.downloadBtn) {
    els.downloadBtn.addEventListener("click", () => downloadAnnotatedFile());
  }

  function setRecordingUi(on, frames) {
    if (!els.recordPill) return;
    els.recordPill.hidden = !on;
    if (els.recFrames) els.recFrames.textContent = String(frames || 0);
  }
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function renderResults(data) {
    if (typeof data.fps === "number") els.fpsValue.textContent = data.fps.toFixed(1);
    if (typeof data.inference_ms === "number")
      els.infValue.textContent = data.inference_ms.toFixed(0);
    if (typeof data.count === "number") els.objValue.textContent = String(data.count);
    if (data.device) {
      const d = String(data.device).toLowerCase();
      els.deviceLabel.textContent = d === "coreml" ? "CoreML · M4" : d.toUpperCase();
    }
    if (typeof data.recorded_frames === "number") {
      setRecordingUi(!!data.recording, data.recorded_frames);
    }
    const summary = data.summary || [];
    const detections = data.detections || [];
    els.labelCount.textContent = String(summary.reduce((a, s) => a + s.count, 0));
    if (!summary.length) {
      els.summaryList.innerHTML = `<div class="empty-state"><p>No objects in frame</p><span>Try lowering confidence</span></div>`;
    } else {
      els.summaryList.innerHTML = summary
        .map(
          (s) => `
        <div class="label-card">
          <span class="swatch" style="background:${s.color}"></span>
          <div class="label-meta">
            <div class="label-name">${escapeHtml(s.label)}</div>
            <div class="label-sub">avg ${(s.avg_confidence * 100).toFixed(0)}% conf</div>
          </div>
          <span class="count-pill">×${s.count}</span>
        </div>`
        )
        .join("");
    }
    els.instanceList.innerHTML = detections
      .slice(0, 40)
      .map(
        (d) => `
      <div class="instance-row">
        <span class="name"><span class="swatch" style="background:${d.color}"></span>${escapeHtml(d.label)}</span>
        <span class="conf">${(d.confidence * 100).toFixed(0)}%</span>
      </div>`
      )
      .join("");
  }
  function clearLabels() {
    els.summaryList.innerHTML = `<div class="empty-state"><p>No objects yet</p><span>Start detection to see labels here</span></div>`;
    els.instanceList.innerHTML = "";
    els.labelCount.textContent = "0";
    els.objValue.textContent = "0";
    els.fpsValue.textContent = "0";
    els.infValue.textContent = "—";
  }

  async function startLive() {
    try {
      closeSocket();
      resetLiveClientState();
      clearPreviewVideo();
      hideDownload();
      setRecordingUi(false, 0);
      els.progressBar.hidden = true;
      els.statusText.textContent = "Requesting camera…";
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      els.webcam.srcObject = state.stream;
      await els.webcam.play();
      const proto = location.protocol === "https:" ? "wss" : "ws";
      state.ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
      state.ws.onopen = () => {
        state.ws.send(
          JSON.stringify({ type: "config", confidence: Number(els.confSlider.value) / 100 })
        );
        setRunningFlags({ running: true });
        // Drop first frames so the export does not begin on a black camera warm-up.
        state.liveWarmupLeft = 4;
        state.inFlight = 0;
        els.statusText.textContent = "Live detection · starting pipeline…";
        showPlaceholder(false);
        setRecordingUi(true, 0);
        pumpLiveFrames();
      };
      state.ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === "result") {
          state.busyFrame = false;
          state.inFlight = 0;
          showFrame(data.frame);
          renderResults(data);
          const fps = Number(data.fps || 0);
          els.statusText.textContent = `Live · ${fps.toFixed(1)} FPS · ${data.count || 0} objects · MT`;
        } else if (data.type === "recording_started") {
          setRecordingUi(true, 0);
          els.statusText.textContent = "Live · multi-thread pipeline";
        } else if (data.type === "recording_ready") {
          setRecordingUi(false, data.frames || 0);
          if (data.output) {
            showDownload(data.output, `${data.frames || 0} live frames ready`);
            els.statusText.textContent = `Saved · ${data.frames || 0} frames — click Download`;
          } else {
            els.statusText.textContent = "Stopped (no frames to save)";
          }
          finishLiveCleanup();
        } else if (data.type === "error") {
          els.statusText.textContent = data.message || "Error";
        }
      };
      state.ws.onerror = () => {
        els.statusText.textContent = "WebSocket error";
      };
      state.ws.onclose = () => {
        if (state.running && !state.stopping) {
          finishLiveCleanup();
          els.statusText.textContent = "Connection closed";
        }
      };
    } catch (err) {
      console.error(err);
      els.statusText.textContent = "Camera permission denied";
      alert("Could not access the camera.");
      stopAll(true);
    }
  }

  function pumpLiveFrames() {
    // Low-latency live capture:
    // - Send frames at ~30 FPS without waiting for server results
    // - Server keeps only the latest frame (no queue lag)
    // - Display every annotated result as soon as it arrives
    const TARGET_INTERVAL_MS = 33; // ~30 FPS capture rate
    let lastSend = 0;

    const tick = () => {
      if (!state.running || state.stopping || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
      }
      const video = els.webcam;
      const now = performance.now();
      if (
        video.readyState >= 2 &&
        video.videoWidth >= 16 &&
        video.videoHeight >= 16 &&
        !video.paused &&
        now - lastSend >= TARGET_INTERVAL_MS
      ) {
        if (state.liveWarmupLeft > 0) {
          state.liveWarmupLeft -= 1;
          // Always show raw camera during warm-up so UI is never black/stuck.
          try {
            const canvas = els.captureCanvas;
            const maxW = 640;
            const scale = Math.min(1, maxW / video.videoWidth);
            canvas.width = Math.round(video.videoWidth * scale);
            canvas.height = Math.round(video.videoHeight * scale);
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            // Prefer showing live camera until detections stream in.
            if (els.outputFrame.hidden || !els.outputFrame.src) {
              showFrame(canvas.toDataURL("image/jpeg", 0.65));
            }
          } catch (_) {}
          if (state.liveWarmupLeft === 0) {
            els.statusText.textContent = "Live detection · multi-thread pipeline";
          }
        } else {
          try {
            const canvas = els.captureCanvas;
            // 640px wide keeps encode + inference fast for 20+ FPS on M4.
            const maxW = 640;
            const scale = Math.min(1, maxW / video.videoWidth);
            canvas.width = Math.round(video.videoWidth * scale) || 640;
            canvas.height = Math.round(video.videoHeight * scale) || 480;
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            // If the socket is congested, skip this capture so we never queue lag.
            if (state.ws.bufferedAmount > 350000) {
              lastSend = now;
            } else {
              const dataUrl = canvas.toDataURL("image/jpeg", 0.62);
              // Fire-and-forget: server overwrites with latest frame only.
              state.ws.send(JSON.stringify({ type: "frame", frame: dataUrl }));
              lastSend = now;
            }
          } catch (e) {
            console.warn("live capture send failed", e);
          }
        }
      }
      state.loopId = requestAnimationFrame(tick);
    };
    state.loopId = requestAnimationFrame(tick);
  }

  async function startUpload() {
    if (!state.selectedFile) {
      alert("Please choose a video file first.");
      return;
    }
    try {
      closeSocket();
      resetLiveClientState();
      setRunningFlags({ running: true });
      hideDownload();
      setRecordingUi(false, 0);
      state.expectedFrames = 0;
      state.processed = 0;
      state._downloadAfterExport = false;
      // Keep the selected video visible on stage while uploading / until first annotated frame.
      if (state.selectedFile && els.previewVideo && !state.previewObjectUrl) {
        showSelectedVideoPreview(state.selectedFile);
      }
      els.statusText.textContent = "Uploading video…";
      els.progressBar.hidden = false;
      els.progressFill.style.width = "0%";
      els.progressLabel.textContent = "Uploading…";
      const form = new FormData();
      form.append("file", state.selectedFile);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      let meta;
      try {
        meta = await res.json();
      } catch (_) {
        throw new Error(
          `Upload failed (HTTP ${res.status}). Is the server running the latest build?`
        );
      }
      if (!res.ok) throw new Error(meta.error || `Upload failed (HTTP ${res.status})`);
      state.jobId = meta.job_id;
      els.statusText.textContent = `Processing ${meta.filename}…`;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      state.ws = new WebSocket(`${proto}://${location.host}/ws/video/${state.jobId}`);
      state.ws.onopen = () => {
        state.ws.send(
          JSON.stringify({ type: "config", confidence: Number(els.confSlider.value) / 100 })
        );
      };
      state.ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === "start") {
          state.expectedFrames = data.expected_frames || 0;
          state.processed = data.processed || 0;
          updateProgress(data.progress || 0, state.processed, state.expectedFrames);
          els.statusText.textContent = `Detecting · 0 / ${state.expectedFrames} frames`;
        } else if (data.type === "result") {
          state.processed = data.processed || state.processed;
          if (data.expected_frames) state.expectedFrames = data.expected_frames;
          showFrame(data.frame);
          renderResults(data);
          updateProgress(data.progress, state.processed, state.expectedFrames);
          // Offer download of whatever we have so far
          if (state.processed > 0 && els.downloadBtn) {
            els.downloadBtn.hidden = false;
            els.downloadBtn.textContent = "Download so far";
            if (els.downloadHint) {
              els.downloadHint.hidden = false;
              els.downloadHint.textContent = `${state.processed} frames processed (partial OK)`;
            }
          }
          els.statusText.textContent = `Detecting · ${state.processed} / ${state.expectedFrames || "?"} frames`;
        } else if (data.type === "paused") {
          setRunningFlags({ running: false, paused: true });
          state.processed = data.processed || state.processed;
          updateProgress(data.progress, state.processed, data.expected_frames || state.expectedFrames);
          if (data.output) {
            showDownload(
              data.output,
              `Paused at ${data.processed} frames — partial download ready`
            );
            els.statusText.textContent = `Paused · ${data.processed} / ${data.expected_frames || state.expectedFrames} — download or resume`;
          } else {
            els.statusText.textContent = `Paused · ${data.processed} frames${data.error ? " (" + data.error + ")" : ""}`;
          }
        } else if (data.type === "resumed") {
          setRunningFlags({ running: true, paused: false });
          els.statusText.textContent = `Resumed · ${data.processed} / ${data.expected_frames || state.expectedFrames}`;
          updateProgress(data.progress, data.processed, data.expected_frames);
        } else if (data.type === "export_ready") {
          if (data.output) {
            showDownload(
              data.output,
              data.partial
                ? `Partial · ${data.processed} frames`
                : `${data.processed} frames ready`
            );
            els.statusText.textContent = data.partial
              ? `Partial export ready (${data.processed} frames)`
              : `Export ready (${data.processed} frames)`;
            if (state._downloadAfterExport) {
              performDownload();
            }
          }
        } else if (data.type === "done") {
          state.processed = data.processed || state.processed;
          updateProgress(1, state.processed, data.expected_frames || state.expectedFrames);
          els.progressFill.style.width = "100%";
          els.progressLabel.textContent = `100% · ${state.processed} / ${data.expected_frames || state.expectedFrames || state.processed} frames`;
          if (data.output) {
            showDownload(
              data.output,
              data.complete
                ? `${data.processed} frames (complete)`
                : `${data.processed} frames (stopped early)`
            );
            els.statusText.textContent = data.complete
              ? `Done · ${data.processed} frames — click Download`
              : `Stopped · ${data.processed} frames saved — click Download`;
          } else {
            const err = data.error || "Could not build annotated video";
            els.statusText.textContent = `Finished but no download: ${err}`;
            alert(`Processing finished but annotated video was not created.\n${err}`);
          }
          setRunningFlags({ running: false, paused: false });
          closeSocket();
        } else if (data.type === "error") {
          els.statusText.textContent = data.message || "Error";
          if (!state.paused) stopAll(true);
        }
      };
      state.ws.onerror = () => {
        els.statusText.textContent = "WebSocket error";
      };
      state.ws.onclose = () => {
        // Ignore late close events after we already cleared the socket.
        if (state.ws && state.running && !state.stopping && !state.paused) {
          setRunningFlags({ running: false, paused: false });
          els.statusText.textContent = "Connection closed";
        }
      };
    } catch (err) {
      console.error(err);
      els.statusText.textContent = err.message || "Upload failed";
      alert(err.message || "Upload failed");
      stopAll(true);
    }
  }

  function finishLiveCleanup() {
    state.running = false;
    state.stopping = false;
    state.busyFrame = false;
    state.inFlight = 0;
    if (state.loopId) {
      cancelAnimationFrame(state.loopId);
      state.loopId = null;
    }
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    els.webcam.srcObject = null;
    closeSocket();
    resetLiveClientState();
    setRunningFlags({ running: false, paused: false });
    setRecordingUi(false, Number(els.recFrames?.textContent || 0));
  }

  function stopAll(force) {
    if (state.mode === "live" && state.ws && state.ws.readyState === WebSocket.OPEN && state.running) {
      state.stopping = true;
      state.running = false;
      if (state.loopId) {
        cancelAnimationFrame(state.loopId);
        state.loopId = null;
      }
      els.statusText.textContent = "Saving annotated video…";
      els.btnStop.disabled = true;
      try {
        state.ws.send(JSON.stringify({ type: "stop" }));
      } catch (_) {
        finishLiveCleanup();
      }
      setTimeout(() => {
        if (state.ws) {
          els.statusText.textContent = "Save timed out — try again";
          finishLiveCleanup();
        }
      }, 60000);
      return;
    }

    if (
      state.mode === "upload" &&
      state.ws &&
      state.ws.readyState === WebSocket.OPEN &&
      (state.running || state.paused) &&
      !force
    ) {
      // Stop & save: finalize partial/full and enable download
      state.stopping = true;
      els.statusText.textContent = "Saving annotated video…";
      els.btnStop.disabled = true;
      els.btnPause && (els.btnPause.disabled = true);
      els.btnResume && (els.btnResume.disabled = true);
      try {
        state.ws.send(JSON.stringify({ type: "stop" }));
      } catch (_) {
        setRunningFlags({ running: false, paused: false });
        state.ws = null;
      }
      return;
    }

    state.running = false;
    state.paused = false;
    state.stopping = false;
    state.busyFrame = false;
    if (state.loopId) {
      cancelAnimationFrame(state.loopId);
      state.loopId = null;
    }
    closeSocket();
    resetLiveClientState();
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    els.webcam.srcObject = null;
    setRunningFlags({ running: false, paused: false });
    setRecordingUi(false, 0);
    if (!pendingDownloadUrl) els.statusText.textContent = "Stopped";
  }

  if (els.btnBenchmark) {
    els.btnBenchmark.addEventListener("click", async () => {
      els.btnBenchmark.disabled = true;
      els.btnBenchmark.textContent = "Running benchmark…";
      if (els.benchResult) els.benchResult.textContent = "Measuring single-thread vs 4-thread pipeline…";
      els.statusText.textContent = "FPS benchmark running (may take ~1 min)…";
      try {
        const res = await fetch("/api/benchmark?frames=48");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Benchmark failed");
        const b = data.before;
        const a = data.after;
        const line = `Before ${b.avg_fps} FPS (1 thread) → After ${a.avg_fps} FPS (4 threads) · speedup ${data.speedup_x}× · device ${data.device}`;
        if (els.benchResult) els.benchResult.textContent = line;
        els.statusText.textContent = line;
        console.log("benchmark", data);
      } catch (e) {
        console.error(e);
        if (els.benchResult) els.benchResult.textContent = e.message || "Benchmark failed";
        els.statusText.textContent = e.message || "Benchmark failed";
      } finally {
        els.btnBenchmark.disabled = false;
        els.btnBenchmark.textContent = "Compare FPS (before/after MT)";
      }
    });
  }

  setMode("live");
  clearLabels();
})();
