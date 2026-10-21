"use strict";
const $ = (id) => document.getElementById(id);
const input = $("file-input"), dropzone = $("dropzone"), detectButton = $("detect-button");
const supported = /\.(jpe?g|png|bmp|tiff?|webp)$/i;
let files = [], previewUrls = [], ready = false, busy = false, healthPending = false;
const sizeText = (bytes) => bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function showError(message) {
  $("error-message").textContent = message || "";
  $("error-message").hidden = !message;
}
function updateControls() {
  detectButton.disabled = busy || !ready || !files.length;
  $("clear-button").disabled = busy || !files.length;
  input.disabled = busy;
  $("detect-label").textContent = busy ? "正在检测，请稍候…" : files.length > 1 ? `开始检测 ${files.length} 张图片` : "开始检测";
  $("activity").classList.toggle("busy", busy);
  if (!busy) $("activity").textContent = !ready ? "请先确保检测服务已连接" : files.length ? "准备就绪，点击开始检测" : "请选择需要检测的图片";
}
function renderSelection() {
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = [];
  $("selection").replaceChildren();
  files.forEach((file, index) => {
    const row = element("div", "selection-item");
    const image = element("img");
    image.src = URL.createObjectURL(file); previewUrls.push(image.src); image.alt = file.name;
    const meta = element("div", "file-meta");
    meta.append(element("strong", "", file.name), element("small", "", sizeText(file.size)));
    const remove = element("button", "remove-file", "×");
    remove.setAttribute("aria-label", `移除 ${file.name}`);
    remove.addEventListener("click", () => { if (!busy) { files.splice(index, 1); renderSelection(); } });
    row.append(image, meta, remove); $("selection").append(row);
  });
  $("file-count").textContent = files.length ? `${files.length} 张图片` : "尚未选择";
  $("total-size").textContent = files.length ? `合计 ${sizeText(files.reduce((total, file) => total + file.size, 0))}` : "";
  updateControls();
}
function addFiles(selected) {
  if (busy) return;
  showError("");
  const candidates = [...files, ...Array.from(selected)];
  if (candidates.length > 50) return showError("每次最多选择 50 张图片，请分批检测。");
  const bad = candidates.find((file) => !supported.test(file.name) || file.size === 0 || file.size > 16 * 1024 * 1024);
  if (bad) return showError(`「${bad.name}」格式不受支持、文件为空或超过 16 MB，请重新选择。`);
  if (candidates.reduce((total, file) => total + file.size, 0) > 100 * 1024 * 1024) return showError("本次图片总大小超过 100 MB，请分批检测。");
  files = candidates; renderSelection();
}
async function checkHealth() {
  if (healthPending) return;
  healthPending = true;
  try {
    const response = await fetch("/api/health", {signal: AbortSignal.timeout(7000)});
    const data = await response.json(); ready = response.ok && data.ready;
  } catch { ready = false; }
  finally { healthPending = false; }
  $("service-dot").className = `status-dot ${ready ? "ready" : "offline"}`;
  $("service-status").textContent = ready ? "检测服务已连接" : "检测服务未就绪";
  window.dispatchEvent(new CustomEvent("fall-health", {detail: {ready}}));
  updateControls();
}
function downloadLink(text, url) {
  const link = element("a", "", text); link.href = `${url}?download=1`; return link;
}
function renderResults(data) {
  $("empty-state").hidden = true;
  $("results").replaceChildren();
  $("result-count").textContent = `${data.total} 张图片`;
  $("results-summary").hidden = false;
  $("results-summary").textContent = `检测完成 · 成功 ${data.succeeded} 张${data.failed ? ` · 失败 ${data.failed} 张` : ""} · 用时 ${data.elapsed_seconds.toFixed(2)} 秒`;
  data.results.forEach((result) => {
    const card = element("article", "result-card");
    if (result.image_url) {
      const imageButton = element("button", "result-image-button");
      imageButton.setAttribute("aria-label", `放大 ${result.filename} 的检测结果`);
      const image = element("img", "result-image"); image.src = result.image_url; image.alt = `${result.filename} 检测标注`; image.loading = "lazy";
      imageButton.append(image);
      imageButton.addEventListener("click", () => { $("large-image").src = result.image_url; $("image-dialog").showModal(); });
      card.append(imageButton);
    }
    const body = element("div", "result-body"), title = element("div", "result-title");
    title.append(element("h3", "", result.filename), element("span", `badge ${result.state}`, result.state_label));
    body.append(title);
    if (result.error) body.append(element("p", "result-error", result.error));
    else {
      body.append(element("p", "muted", `检测到 ${result.persons_detected} 人`));
      (result.persons || []).forEach((person) => {
        const row = element("div", "person-row");
        row.append(element("span", "", `人物 ${person.person_id} · ${person.state_label}`), element("span", "person-score", `模型分数 ${(person.confidence * 100).toFixed(1)}%`));
        body.append(row);
      });
      const links = element("div", "result-links");
      links.append(downloadLink("下载标注图片 ↓", result.image_url), downloadLink("详细结果 JSON ↓", result.json_url)); body.append(links);
    }
    card.append(body); $("results").append(card);
  });
  $("downloads").hidden = false;
  $("download-zip").href = data.zip_url;
  $("download-csv").href = `${data.csv_url}?download=1`;
  $("download-json").href = `${data.json_url}?download=1`;
}
input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
dropzone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!busy) input.click(); } });
for (const name of ["dragenter", "dragover"]) dropzone.addEventListener(name, (event) => { event.preventDefault(); if (!busy) dropzone.classList.add("dragover"); });
for (const name of ["dragleave", "drop"]) dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragover"); });
dropzone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
$("clear-button").addEventListener("click", () => { files = []; showError(""); renderSelection(); });
$("refresh-health").addEventListener("click", checkHealth);
$("close-dialog").addEventListener("click", () => $("image-dialog").close());
$("image-dialog").addEventListener("click", (event) => { if (event.target === $("image-dialog")) $("image-dialog").close(); });
detectButton.addEventListener("click", async () => {
  if (busy || !files.length) return;
  busy = true; showError(""); updateControls();
  $("activity").textContent = `正在处理 ${files.length} 张图片，请等待结果…`;
  const body = new FormData(); files.forEach((file) => body.append("images", file));
  try {
    const response = await fetch("/api/detect", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "检测失败，请重试。");
    renderResults(data);
  } catch (error) { showError(error.message === "Failed to fetch" ? "连接中断，请确认网页服务仍在运行后重试。" : error.message); }
  finally { busy = false; updateControls(); await checkHealth(); }
});
checkHealth();
setInterval(checkHealth, 20000);
