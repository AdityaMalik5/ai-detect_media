const els = {
  statusText: document.getElementById("statusText"),
  fileInput: document.getElementById("fileInput"),
  predictBtn: document.getElementById("predictBtn"),
  previewImg: document.getElementById("previewImg"),
  previewPlaceholder: document.getElementById("previewPlaceholder"),
  uploadHint: document.getElementById("uploadHint"),
  resultStatus: document.getElementById("resultStatus"),
  pristineScore: document.getElementById("pristineScore"),
  deepfakeScore: document.getElementById("deepfakeScore"),
  metaLine: document.getElementById("metaLine"),
  historyList: document.getElementById("historyList"),
  historyEmpty: document.querySelector(".historyEmpty"),
  clearHistory: document.getElementById("clearHistory"),
  readyPct: document.getElementById("readyPct"),
  progressFill: document.getElementById("progressFill"),
  navUpload: document.getElementById("navUpload"),
  navHistory: document.getElementById("navHistory"),
  navSettings: document.getElementById("navSettings"),
  searchInput: document.querySelector(".search"),
  uploadPanel: document.querySelector(".uploadPanel"),
};

let selectedFile = null;
let history = JSON.parse(localStorage.getItem("dfHistory") || "[]");

// ── Helpers ─────────────────────────────────────────────────────────────────

function pct(n) {
  return Math.max(0, Math.min(100, Math.round(n)));
}

function saveHistory() {
  localStorage.setItem("dfHistory", JSON.stringify(history.slice(0, 50)));
}

// ── Nav ─────────────────────────────────────────────────────────────────────

function setActiveNav(clickedEl) {
  document.querySelectorAll(".navItem").forEach(function (n) {
    n.classList.remove("active");
  });
  clickedEl.classList.add("active");
}

document.querySelectorAll(".navItem").forEach(function (item) {
  item.addEventListener("click", function (e) {
    e.preventDefault();
    setActiveNav(item);
  });
});

// Detect nav — scroll to upload panel and focus
if (els.navUpload) {
  els.navUpload.addEventListener("click", function () {
    if (els.uploadPanel) {
      els.uploadPanel.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    els.fileInput.click();
  });
}

// History nav — scroll to history panel
if (els.navHistory) {
  els.navHistory.addEventListener("click", function () {
    els.historyList.scrollIntoView({ behavior: "smooth", block: "center" });
  });
}

// Settings nav — toggle model info
if (els.navSettings) {
  els.navSettings.addEventListener("click", function () {
    loadStatus();
    var msg = "Model status refreshed. Check the status badge in the top bar.";
    els.metaLine.textContent = msg;
  });
}

// ── Search — filter history ─────────────────────────────────────────────────

if (els.searchInput) {
  els.searchInput.disabled = false;
  els.searchInput.placeholder = "Search prediction history...";
  els.searchInput.addEventListener("input", function () {
    var query = els.searchInput.value.toLowerCase().trim();
    var items = els.historyList.querySelectorAll(".historyItem");
    items.forEach(function (item) {
      var text = item.textContent.toLowerCase();
      item.style.display = !query || text.indexOf(query) !== -1 ? "" : "none";
    });
  });
}

// ── History rendering ───────────────────────────────────────────────────────

function renderHistory() {
  if (!history.length) {
    els.historyEmpty.style.display = "block";
    els.historyList.querySelectorAll(".historyItem").forEach(function (n) {
      n.remove();
    });
    return;
  }

  els.historyEmpty.style.display = "none";
  els.historyList.querySelectorAll(".historyItem").forEach(function (n) {
    n.remove();
  });

  history.slice(0, 8).forEach(function (item) {
    var div = document.createElement("div");
    div.className = "historyItem";

    var top = document.createElement("div");
    top.className = "historyItemTop";

    var labelEl = document.createElement("div");
    labelEl.className = "historyLabel";
    labelEl.textContent = item.label;

    var pillEl = document.createElement("div");
    pillEl.className = "pill";
    pillEl.textContent = item.date;

    top.appendChild(labelEl);
    top.appendChild(pillEl);

    var meta = document.createElement("div");
    meta.className = "historyMeta";
    meta.textContent =
      "Pristine: " +
      (item.pristine_prob * 100).toFixed(2) +
      "% | Deepfake: " +
      (item.deepfake_prob * 100).toFixed(2) +
      "%";

    var fname = document.createElement("div");
    fname.className = "historyMeta";
    fname.textContent = item.filename || "";

    div.appendChild(top);
    div.appendChild(meta);
    if (item.filename) div.appendChild(fname);
    els.historyList.appendChild(div);
  });
}

// ── Status ──────────────────────────────────────────────────────────────────

async function loadStatus() {
  try {
    var res = await fetch("/api/status");
    var data = await res.json();
    var ready = data.model_loaded ? "Ready" : "Not ready";
    els.statusText.textContent = ready;
    var w = data.model_loaded ? 100 : 0;
    els.readyPct.textContent = w + "%";
    els.progressFill.style.width = w + "%";

    if (!data.model_loaded) {
      els.predictBtn.disabled = true;
      els.predictBtn.title = data.init_error || "Model not loaded";
    }
  } catch (e) {
    els.statusText.textContent = "Offline";
    els.readyPct.textContent = "0%";
    els.progressFill.style.width = "0%";
  }
}

// ── File selection ──────────────────────────────────────────────────────────

function setUploadingState() {
  els.predictBtn.disabled = !selectedFile;
  if (selectedFile) {
    els.uploadHint.textContent = selectedFile.name;
  } else {
    els.uploadHint.textContent = "Waiting for upload";
  }
}

function resetResults() {
  els.resultStatus.textContent = "\u2014";
  els.pristineScore.textContent = "0.00";
  els.deepfakeScore.textContent = "0.00";
  els.metaLine.textContent = "";
}

els.fileInput.addEventListener("change", function () {
  selectedFile =
    els.fileInput.files && els.fileInput.files[0] ? els.fileInput.files[0] : null;
  if (selectedFile) {
    var url = URL.createObjectURL(selectedFile);
    els.previewImg.src = url;
    els.previewImg.style.display = "block";
    els.previewPlaceholder.style.display = "none";
    resetResults();
  } else {
    els.previewImg.src = "";
    els.previewImg.style.display = "none";
    els.previewPlaceholder.style.display = "block";
  }
  setUploadingState();
});

// ── Drag and drop on the preview box ────────────────────────────────────────

var previewBox = document.querySelector(".previewBox");
if (previewBox) {
  previewBox.addEventListener("dragover", function (e) {
    e.preventDefault();
    previewBox.style.borderColor = "var(--accent)";
  });
  previewBox.addEventListener("dragleave", function () {
    previewBox.style.borderColor = "";
  });
  previewBox.addEventListener("drop", function (e) {
    e.preventDefault();
    previewBox.style.borderColor = "";
    var files = e.dataTransfer.files;
    if (files && files.length > 0) {
      // Update the file input so the rest of the flow works
      var dt = new DataTransfer();
      dt.items.add(files[0]);
      els.fileInput.files = dt.files;
      els.fileInput.dispatchEvent(new Event("change"));
    }
  });

  // Click on preview box also opens file picker
  previewBox.addEventListener("click", function () {
    els.fileInput.click();
  });
  previewBox.style.cursor = "pointer";
}

// ── Predict ─────────────────────────────────────────────────────────────────

els.predictBtn.addEventListener("click", async function () {
  if (!selectedFile) return;

  els.predictBtn.disabled = true;
  els.predictBtn.textContent = "Detecting...";
  els.resultStatus.textContent = "Working...";
  els.metaLine.textContent = "";

  try {
    var form = new FormData();
    form.append("file", selectedFile);

    var res = await fetch("/api/predict", {
      method: "POST",
      body: form,
    });

    var data = await res.json();
    if (!data.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    var label = data.label;
    els.resultStatus.textContent = label;

    var pristinePct = data.pristine_prob * 100;
    var deepfakePct = data.deepfake_prob * 100;

    els.pristineScore.textContent = pristinePct.toFixed(2) + "%";
    els.deepfakeScore.textContent = deepfakePct.toFixed(2) + "%";

    els.metaLine.textContent = "Analysis complete. Input size: " + data.meta.input_size + "px.";

    var now = new Date();
    history.unshift({
      label: label,
      pristine_prob: data.pristine_prob,
      deepfake_prob: data.deepfake_prob,
      filename: selectedFile.name,
      date:
        now.toLocaleDateString() +
        " " +
        now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    });
    saveHistory();
    renderHistory();
  } catch (e) {
    els.resultStatus.textContent = "Error";
    els.metaLine.textContent = String(e.message || e);
    els.pristineScore.textContent = "0.00";
    els.deepfakeScore.textContent = "0.00";
  } finally {
    els.predictBtn.textContent = "Detect";
    setUploadingState();
  }
});

// ── Clear history ───────────────────────────────────────────────────────────

els.clearHistory.addEventListener("click", function () {
  history = [];
  saveHistory();
  renderHistory();
  if (els.searchInput) els.searchInput.value = "";
});

// ── Init ────────────────────────────────────────────────────────────────────

loadStatus();
setUploadingState();
renderHistory();
