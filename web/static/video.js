"use strict";
(() => {
  const byId = (id) => document.getElementById(id);
  const videoInput = byId("video-input"), button = byId("video-detect");
  let selected = null, running = false, serviceReady = false, job = null;
  const savedJobKey = "fall-video-job";
  function storageGet() { try { return sessionStorage.getItem(savedJobKey); } catch { return null; } }
  function storageSet(value) { try { if (value) sessionStorage.setItem(savedJobKey, value); else sessionStorage.removeItem(savedJobKey); } catch { /* Storage is optional. */ } }
  function error(message) { byId("video-error").textContent = message; byId("video-error").hidden = !message; }
  function controls() { button.disabled = running || !selected || !serviceReady; videoInput.disabled = running; }
  window.addEventListener("fall-health", (event) => { serviceReady = event.detail.ready; controls(); });
  function tab(mode) {
    for (const name of ["image", "video"]) {
      const active = name === mode;
      byId(`${name}-workspace`).hidden = !active;
      byId(`${name}-tab`).classList.toggle("active", active);
      byId(`${name}-tab`).setAttribute("aria-selected", String(active));
      byId(`${name}-tab`).tabIndex = active ? 0 : -1;
    }
  }
  for (const name of ["image", "video"]) {
    byId(`${name}-tab`).addEventListener("click", () => tab(name));
    byId(`${name}-tab`).addEventListener("keydown", (event) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const other = event.key === "Home" ? "image" : event.key === "End" ? "video" : name === "video" ? "image" : "video";
        tab(other); byId(`${other}-tab`).focus();
      }
    });
  }
  function choose(list) {
    if (running) return;
    error("");
    if (list.length !== 1) return error("请每次选择一个视频。");
    const file = list[0];
    if (!/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name) || !file.size || file.size > 100 * 1024 * 1024) return error("请选择不超过 100 MB 的有效视频文件。");
    selected = file;
    byId("video-selection").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
    controls();
  }
  videoInput.addEventListener("change", () => { choose(Array.from(videoInput.files)); videoInput.value = ""; });
  const drop = byId("video-dropzone");
  drop.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); if (!running) videoInput.click(); } });
  for (const name of ["dragenter", "dragover"]) drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("dragover"); });
  for (const name of ["dragleave", "drop"]) drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("dragover"); });
  drop.addEventListener("drop", (event) => choose(Array.from(event.dataTransfer.files)));
  function render(result) {
    byId("video-empty").hidden = true; byId("video-result").hidden = false;
    byId("video-result-status").textContent = "检测完成";
    byId("video-summary").textContent = `${result.filename} · ${result.state_label} · 共 ${result.total_frames} 帧 · ${result.frames_with_classification} 帧有分类结果 · 用时 ${result.elapsed_seconds.toFixed(1)} 秒`;
    byId("video-player").src = result.video_url;
    for (const [id, key] of [["video-download", "video_url"], ["video-csv", "csv_url"], ["video-json", "json_url"], ["video-jsonl", "predictions_url"], ["video-zip", "zip_url"]]) byId(id).href = `${result[key]}?download=1`;
  }
  function finish() { running = false; job = null; storageSet(null); byId("video-cancel").hidden = true; controls(); }
  async function poll() {
    if (!job) return;
    try {
      const response = await fetch(`/api/videos/${job}`, {signal: AbortSignal.timeout(10000)});
      const status = await response.json();
      if (!response.ok) { finish(); throw new Error(status.error || "任务不存在，请重新上传。"); }
      error("");
      byId("video-progress").value = status.progress || 0;
      byId("video-activity").textContent = `已处理 ${status.processed_frames} / ${status.total_frames} 帧 · ${status.progress}%`;
      if (status.status === "completed") { render(status.result); finish(); return; }
      if (["failed", "cancelled"].includes(status.status)) { byId("video-result-status").textContent = status.status === "cancelled" ? "已取消" : "检测失败"; finish(); throw new Error(status.error); }
    } catch (failure) { error(job ? "连接暂时中断，正在重连；视频会继续处理。" : failure.message); }
    if (job) setTimeout(poll, 1000);
  }
  button.addEventListener("click", () => {
    if (!selected || running) return;
    running = true; error(""); controls();
    byId("video-result").hidden = true; byId("video-empty").hidden = false;
    byId("video-player").removeAttribute("src"); byId("video-player").load();
    byId("video-progress-area").hidden = false; byId("video-progress").value = 0;
    byId("video-cancel").hidden = true; byId("video-cancel").disabled = false;
    byId("video-result-status").textContent = "正在处理";
    const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/videos"); xhr.responseType = "json";
    xhr.upload.onprogress = (event) => { byId("video-activity").textContent = event.lengthComputable ? `正在上传视频 ${Math.round(event.loaded / event.total * 100)}%，请勿关闭页面…` : "正在上传视频…"; };
    xhr.onload = () => {
      if (xhr.status !== 202) { error(xhr.response?.error || "上传失败，请重试。"); finish(); return; }
      job = xhr.response.job_id; storageSet(job); byId("video-cancel").hidden = false; poll();
    };
    xhr.onerror = () => { error("视频上传连接中断，请重新上传。"); finish(); };
    const data = new FormData(); data.append("video", selected); xhr.send(data);
  });
  byId("video-cancel").addEventListener("click", async () => {
    if (!job) return;
    byId("video-cancel").disabled = true;
    try {
      const response = await fetch(`/api/videos/${job}/cancel`, {method: "POST"});
      if (!response.ok) throw new Error();
      byId("video-activity").textContent = "正在取消，请稍候…";
    } catch { error("取消请求失败，请重试。"); byId("video-cancel").disabled = false; }
  });
  const previous = storageGet();
  if (previous && /^[a-f0-9]{32}$/.test(previous)) {
    job = previous; running = true; tab("video"); controls();
    byId("video-progress-area").hidden = false; byId("video-cancel").hidden = false; poll();
  }
})();
