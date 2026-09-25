const textInput = document.getElementById("textInput");
const fileInput = document.getElementById("fileInput");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const counter = document.getElementById("counter");
const fileName = document.getElementById("fileName");
const errorBox = document.getElementById("errorBox");
const resultSection = document.getElementById("resultSection");

function countWords(text) {
  const matches = text.trim().match(/[\p{L}\p{N}_'-]+/gu);

  return matches ? matches.length : 0;
}

function setText(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.textContent = value;
  }
}

function setBar(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.style.width = `${Math.max(0, Math.min(100, value))}%`;
  }
}

function formatPercent(value) {
  if (typeof value !== "number") {
    return "-";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function updateCounter() {
  counter.textContent = `${countWords(textInput.value)} kata`;
}

function clearError() {
  errorBox.textContent = "";

  errorBox.classList.add("hidden");
}

function showError(message) {
  errorBox.textContent = message;

  errorBox.classList.remove("hidden");
}

function setLoading(active) {
  analyzeButton.disabled = active;
  clearButton.disabled = active;
  fileInput.disabled = active;

  analyzeButton.textContent = active ? "Memeriksa..." : "Periksa teks";
}

function verdictTitle(key) {
  if (key === "ai") {
    return "Teks menunjukkan pola yang cukup kuat ke arah AI.";
  }

  if (key === "human") {
    return "Teks lebih dekat dengan pola tulisan manusia.";
  }

  return "Hasilnya masih berada di area abu-abu.";
}

function createSectionCard(section) {
  const details = document.createElement("details");

  details.className = `section-card ${section.label_key}`;

  const summary = document.createElement("summary");

  const left = document.createElement("div");

  left.className = "section-summary-left";

  const title = document.createElement("strong");

  title.textContent = `Bagian ${section.index}`;

  const meta = document.createElement("span");

  meta.textContent = `${section.word_count} kata`;

  left.append(title, meta);

  const right = document.createElement("div");

  right.className = "section-summary-right";

  const badge = document.createElement("span");

  badge.className = `mini-badge ${section.label_key}`;

  badge.textContent = section.label;

  const score = document.createElement("strong");

  score.textContent = `${section.score}/100`;

  right.append(badge, score);

  summary.append(left, right);

  const body = document.createElement("div");

  body.className = "section-body";

  const text = document.createElement("p");

  text.textContent = section.text;

  const components = document.createElement("div");

  components.className = "section-components";

  const items = [
    ["Pola kata", section.components.word],
    ["Pola karakter", section.components.char],
    ["Gaya tulis", section.components.style],
  ];

  items.forEach(([label, value]) => {
    const item = document.createElement("span");

    item.textContent = `${label}: ${value}`;

    components.appendChild(item);
  });

  body.append(text, components);

  details.append(summary, body);

  return details;
}

function renderNotes(notes) {
  const panel = document.getElementById("notesPanel");

  const list = document.getElementById("notesList");

  list.innerHTML = "";

  if (!notes || notes.length === 0) {
    panel.classList.add("hidden");

    return;
  }

  notes.forEach((note) => {
    const item = document.createElement("li");

    item.textContent = note;

    list.appendChild(item);
  });

  panel.classList.remove("hidden");
}

function renderModelInfo(model) {
  setText("modelName", model.name || "-");

  setText("modelVersion", model.version || "-");

  setText("selectiveAccuracy", formatPercent(model.selective_accuracy));

  setText("selectiveCoverage", formatPercent(model.selective_coverage));

  setText("falsePositiveRate", formatPercent(model.ai_false_positive_rate));

  setText("testSamples", model.test_samples ?? "-");

  const sources = model.sources || {};

  const sourceEntries = Object.entries(sources);

  const sourceText = sourceEntries.length
    ? sourceEntries.map(([name, count]) => `${name}: ${count}`).join(" · ")
    : "Sumber data tidak tersedia.";

  setText("sourceInfo", `Data pelatihan: ${sourceText}`);
}

function renderResult(data) {
  const badge = document.getElementById("verdictBadge");

  badge.className = `verdict-badge ${data.verdict_key}`;

  badge.textContent = data.verdict;

  setText(
    "confidenceText",
    `Keyakinan ${data.confidence.toLowerCase()} · ${data.confidence_score}/100`,
  );

  setText("resultTitle", verdictTitle(data.verdict_key));

  setText("resultSummary", data.summary);

  setText("aiIndex", data.ai_index);

  setText("modelAgreement", `${data.model_agreement}%`);

  setText("sectionConsistency", `${data.section_consistency}%`);

  setText("wordCount", `${data.word_count} kata`);

  setText("sectionCount", data.section_count);

  const scoreRing = document.getElementById("scoreRing");

  scoreRing.style.setProperty("--score", data.ai_index);

  scoreRing.className = `score-ring ${data.verdict_key}`;

  setText("wordScore", data.components.word);

  setText("charScore", data.components.char);

  setText("styleScore", data.components.style);

  setBar("wordBar", data.components.word);

  setBar("charBar", data.components.char);

  setBar("styleBar", data.components.style);

  setText("aiSectionCount", data.section_summary.ai);

  setText("uncertainSectionCount", data.section_summary.uncertain);

  setText("humanSectionCount", data.section_summary.human);

  setText(
    "thresholdInfo",
    `Ambang manusia ≤ ${data.thresholds.human} · AI ≥ ${data.thresholds.ai}`,
  );

  renderNotes(data.notes);

  renderModelInfo(data.model || {});

  const sectionsList = document.getElementById("sectionsList");

  sectionsList.innerHTML = "";

  data.sections.forEach((section) => {
    sectionsList.appendChild(createSectionCard(section));
  });

  resultSection.classList.remove("hidden");

  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

textInput.addEventListener("input", updateCounter);

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name || "Belum ada file";
});

clearButton.addEventListener("click", () => {
  textInput.value = "";
  fileInput.value = "";

  fileName.textContent = "Belum ada file";

  resultSection.classList.add("hidden");

  clearError();

  updateCounter();

  textInput.focus();
});

analyzeButton.addEventListener("click", async () => {
  clearError();

  const text = textInput.value.trim();

  const selectedFile = fileInput.files[0];

  if (!text && !selectedFile) {
    showError("Masukkan teks atau pilih dokumen terlebih dahulu.");

    return;
  }

  const form = new FormData();

  if (text) {
    form.append("text", text);
  }

  if (selectedFile) {
    form.append("file", selectedFile);
  }

  setLoading(true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
    });

    let payload;

    try {
      payload = await response.json();
    } catch {
      payload = {};
    }

    if (!response.ok) {
      throw new Error(payload.detail || "Pemeriksaan gagal.");
    }

    renderResult(payload);
  } catch (error) {
    showError(error.message || "Terjadi kesalahan saat memeriksa teks.");
  } finally {
    setLoading(false);
  }
});

updateCounter();
