(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    btnLive: $("btnLive"), btnUpload: $("btnUpload"), btnStart: $("btnStart"), btnStop: $("btnStop"),
    confSlider: $("confSlider"), confValue: $("confValue"), webcam: $("webcam"),
    captureCanvas: $("captureCanvas"), outputFrame: $("outputFrame"), placeholder: $("placeholder"),
    placeholderTitle: $("placeholderTitle"), placeholderText: $("placeholderText"),
    uploadZone: $("uploadZone"), fileInput: $("fileInput"), btnPickFile: $("btnPickFile"),
    fileName: $("fileName"), downloadLink: $("downloadLink"), summaryList: $("summaryList"),
    instanceList: $("instanceList"), labelCount: $("labelCount"), fpsValue: $("fpsValue"),
    infValue: $("infValue"), objValue: $("objValue"), deviceLabel: $("deviceLabel"),
    deviceChip: $("deviceChip"), statusText: $("statusText"), progressBar: $("progressBar"),
    progressFill: $("progressFill"), progressLabel: $("progressLabel"),
  };
  const state = { mode: "live", running: false, ws: null, stream: null, loopId: null, jobId: null, selectedFile: null, busyFrame: false };

  fetch("/api/health").then(r => r.json()).then(d => {
    const dev = (d.device || "cpu").toLowerCase();
    els.deviceLabel.textContent = dev === "coreml" ? "CoreML · M4" : dev.toUpperCase();
    if (dev === "coreml" || dev === "mps") els.deviceChip.classList.add("live");
    els.statusText.textContent = `Rust · ${d.model} ready`;
  }).catch(() => { els.deviceLabel.textContent = "offline"; els.statusText.textContent = "Server unreachable"; });

  function setMode(mode) {
    if (state.running) stopAll();
    state.mode = mode;
    els.btnLive.classList.toggle("active", mode === "live");
    els.btnUpload.classList.toggle("active", mode === "upload");
    els.uploadZone.hidden = mode !== "upload";
    els.progressBar.hidden = true;
    els.downloadLink.hidden = true;
    if (mode === "live") {
      els.placeholderTitle.textContent = "Live camera mode";
      els.placeholderText.innerHTML = "Click <strong>Start</strong> to open your webcam. Inference runs in <strong>Rust + CoreML</strong>.";
      els.btnStart.textContent = "Start camera";
    } else {
      els.placeholderTitle.textContent = "Upload a video";
      els.placeholderText.innerHTML = "Choose a video file, then click <strong>Start</strong> to run YOLO frame by frame.";
      els.btnStart.textContent = "Start detection";
    }
    showPlaceholder(true);
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
    els.downloadLink.hidden = true;
  });
  els.btnStart.addEventListener("click", () => state.mode === "live" ? startLive() : startUpload());
  els.btnStop.addEventListener("click", () => stopAll());

  function setRunning(on) {
    state.running = on;
    els.btnStart.disabled = on;
    els.btnStop.disabled = !on;
    els.btnLive.disabled = on;
    els.btnUpload.disabled = on;
    if (on) els.deviceChip.classList.add("live");
  }
  function showPlaceholder(show) {
    els.placeholder.hidden = !show;
    if (show) { els.outputFrame.hidden = true; els.outputFrame.removeAttribute("src"); }
  }
  function showFrame(dataUrl) {
    els.placeholder.hidden = true;
    els.outputFrame.hidden = false;
    els.outputFrame.src = dataUrl;
  }
  function escapeHtml(s) {
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }
  function renderResults(data) {
    if (typeof data.fps === "number") els.fpsValue.textContent = data.fps.toFixed(1);
    if (typeof data.inference_ms === "number") els.infValue.textContent = data.inference_ms.toFixed(0);
    if (typeof data.count === "number") els.objValue.textContent = String(data.count);
    if (data.device) {
      const d = String(data.device).toLowerCase();
      els.deviceLabel.textContent = d === "coreml" ? "CoreML · M4" : d.toUpperCase();
    }
    const summary = data.summary || [];
    const detections = data.detections || [];
    els.labelCount.textContent = String(summary.reduce((a, s) => a + s.count, 0));
    if (!summary.length) {
      els.summaryList.innerHTML = `<div class="empty-state"><p>No objects in frame</p><span>Try lowering confidence</span></div>`;
    } else {
      els.summaryList.innerHTML = summary.map(s => `
        <div class="label-card">
          <span class="swatch" style="background:${s.color}"></span>
          <div class="label-meta">
            <div class="label-name">${escapeHtml(s.label)}</div>
            <div class="label-sub">avg ${(s.avg_confidence * 100).toFixed(0)}% conf</div>
          </div>
          <span class="count-pill">×${s.count}</span>
        </div>`).join("");
    }
    els.instanceList.innerHTML = detections.slice(0, 40).map(d => `
      <div class="instance-row">
        <span class="name"><span class="swatch" style="background:${d.color}"></span>${escapeHtml(d.label)}</span>
        <span class="conf">${(d.confidence * 100).toFixed(0)}%</span>
      </div>`).join("");
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
      els.statusText.textContent = "Requesting camera…";
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      els.webcam.srcObject = state.stream;
      await els.webcam.play();
      const proto = location.protocol === "https:" ? "wss" : "ws";
      state.ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
      state.ws.onopen = () => {
        state.ws.send(JSON.stringify({ type: "config", confidence: Number(els.confSlider.value) / 100 }));
        setRunning(true);
        els.statusText.textContent = "Live detection (Rust)";
        showPlaceholder(false);
        pumpLiveFrames();
      };
      state.ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === "result") { state.busyFrame = false; showFrame(data.frame); renderResults(data); }
        else if (data.type === "error") els.statusText.textContent = data.message || "Error";
      };
      state.ws.onerror = () => { els.statusText.textContent = "WebSocket error"; };
      state.ws.onclose = () => { if (state.running) stopAll(); };
    } catch (err) {
      console.error(err);
      els.statusText.textContent = "Camera permission denied";
      alert("Could not access the camera.");
      stopAll();
    }
  }

  function pumpLiveFrames() {
    const tick = () => {
      if (!state.running || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
      if (!state.busyFrame) {
        const video = els.webcam;
        if (video.readyState >= 2) {
          const canvas = els.captureCanvas;
          const maxW = 960;
          const scale = Math.min(1, maxW / (video.videoWidth || maxW));
          canvas.width = Math.round((video.videoWidth || 640) * scale);
          canvas.height = Math.round((video.videoHeight || 480) * scale);
          canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
          state.busyFrame = true;
          state.ws.send(JSON.stringify({ type: "frame", frame: canvas.toDataURL("image/jpeg", 0.72) }));
        }
      }
      state.loopId = requestAnimationFrame(tick);
    };
    state.loopId = requestAnimationFrame(tick);
  }

  async function startUpload() {
    if (!state.selectedFile) { alert("Please choose a video file first."); return; }
    try {
      setRunning(true);
      els.statusText.textContent = "Uploading video…";
      els.progressBar.hidden = false;
      els.progressFill.style.width = "0%";
      els.progressLabel.textContent = "Uploading…";
      els.downloadLink.hidden = true;
      const form = new FormData();
      form.append("file", state.selectedFile);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      let meta;
      try {
        meta = await res.json();
      } catch (_) {
        throw new Error(`Upload failed (HTTP ${res.status}). Is the server running the latest build?`);
      }
      if (!res.ok) throw new Error(meta.error || `Upload failed (HTTP ${res.status})`);
      state.jobId = meta.job_id;
      els.statusText.textContent = `Processing ${meta.filename}…`;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      state.ws = new WebSocket(`${proto}://${location.host}/ws/video/${state.jobId}`);
      state.ws.onopen = () => {
        state.ws.send(JSON.stringify({ type: "config", confidence: Number(els.confSlider.value) / 100 }));
      };
      state.ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === "result") {
          showFrame(data.frame);
          renderResults(data);
          const pct = Math.round((data.progress || 0) * 100);
          els.progressFill.style.width = `${pct}%`;
          els.progressLabel.textContent = `${pct}% · frame ${data.frame_index}`;
        } else if (data.type === "done") {
          els.statusText.textContent = `Done · ${data.processed} frames`;
          els.progressFill.style.width = "100%";
          els.progressLabel.textContent = "100%";
          if (data.output) { els.downloadLink.hidden = false; els.downloadLink.href = data.output; }
          setRunning(false);
          state.ws = null;
        } else if (data.type === "error") {
          els.statusText.textContent = data.message || "Error";
          stopAll();
        }
      };
    } catch (err) {
      console.error(err);
      els.statusText.textContent = err.message || "Upload failed";
      alert(err.message || "Upload failed");
      stopAll();
    }
  }

  function stopAll() {
    state.running = false;
    state.busyFrame = false;
    if (state.loopId) { cancelAnimationFrame(state.loopId); state.loopId = null; }
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      try { state.ws.send(JSON.stringify({ type: "stop" })); state.ws.close(); } catch (_) {}
    }
    state.ws = null;
    if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
    els.webcam.srcObject = null;
    setRunning(false);
    els.statusText.textContent = "Stopped";
  }

  setMode("live");
  clearLabels();
})();
