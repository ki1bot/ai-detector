import { elements, formatPercent, setBar, setText } from "./dom.js";

function overallPresentation(data) {
  if (data.verdict_key === "ai") {
    return {
      key: "ai",
      badge: "Warning AI",
      title: "Teks memiliki indikasi AI yang kuat.",
    };
  }

  if (data.verdict_key === "human") {
    return {
      key: "human",
      badge: "Cenderung manusia",
      title: "Teks lebih konsisten dengan pola tulisan manusia.",
    };
  }

  if ((data.section_summary?.ai || 0) > 0) {
    return {
      key: "uncertain",
      badge: "Sebagian terindikasi AI",
      title: "Sebagian teks perlu ditinjau lebih lanjut.",
    };
  }

  return {
    key: "uncertain",
    badge: "Belum pasti",
    title: "Hasil belum cukup konsisten untuk menetapkan satu sisi.",
  };
}

function sectionPresentation(section) {
  if (section.label_key === "ai") {
    return {
      label: "Indikasi AI",
      key: "ai",
    };
  }

  if (section.label_key === "human") {
    return {
      label: "Cenderung manusia",
      key: "human",
    };
  }

  return {
    label: "Campuran / belum pasti",
    key: "uncertain",
  };
}

function createMarkedBlock(section) {
  const presentation = sectionPresentation(section);
  const details = document.createElement("details");

  details.className = `marked-block ${presentation.key}`;
  details.open = section.index <= 2;

  const summary = document.createElement("summary");
  const meta = document.createElement("div");

  meta.className = "marked-meta";

  const badge = document.createElement("span");

  badge.className = `mini-badge ${presentation.key}`;
  badge.textContent = presentation.label;

  const sectionInfo = document.createElement("strong");

  sectionInfo.textContent = `Bagian ${section.index}`;

  const wordCount = document.createElement("small");

  wordCount.textContent = `${section.word_count} kata`;

  meta.append(badge, sectionInfo, wordCount);

  const score = document.createElement("span");

  score.className = "marked-score";
  score.textContent = `Indeks AI ${section.score}/100`;

  summary.append(meta, score);

  const body = document.createElement("div");

  body.className = "marked-body";

  const text = document.createElement("p");

  text.className = "marked-text";
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

    item.textContent = `${label}: ${value}/100`;

    components.appendChild(item);
  });

  body.append(text, components);
  details.append(summary, body);

  return details;
}

function renderMarkedText(sections) {
  const container = document.getElementById("markedTextList");

  container.innerHTML = "";

  sections.forEach((section) => {
    container.appendChild(createMarkedBlock(section));
  });
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
  setText("balancedAccuracy", formatPercent(model.balanced_accuracy));

  setText(
    "rocAuc",
    typeof model.roc_auc === "number" ? model.roc_auc.toFixed(3) : "-",
  );

  setText("testSamples", model.test_samples ?? "-");

  const sources = model.sources || {};
  const sourceEntries = Object.entries(sources);

  const sourceText = sourceEntries.length
    ? sourceEntries.map(([name, count]) => `${name}: ${count}`).join(" · ")
    : "Sumber data tidak tersedia.";

  setText("sourceInfo", `Data pelatihan: ${sourceText}`);
}

export function renderResult(data) {
  const presentation = overallPresentation(data);
  const badge = document.getElementById("verdictBadge");
  const resultMain = document.getElementById("resultMain");
  const scoreBox = document.getElementById("scoreBox");

  badge.className = `verdict-badge ${presentation.key}`;
  badge.textContent = presentation.badge;

  resultMain.className = `card result-main ${presentation.key}`;
  scoreBox.className = `score-box ${presentation.key}`;

  setText(
    "confidenceText",
    `Reliabilitas ${data.confidence.toLowerCase()} · ${data.confidence_score}/100`,
  );

  setText("resultTitle", presentation.title);
  setText("resultSummary", data.summary);
  setText("aiIndex", data.ai_index);
  setText("modelAgreement", `${data.model_agreement}%`);
  setText("sectionConsistency", `${data.section_consistency}%`);
  setText("wordCount", `${data.word_count} kata`);
  setText("sectionCount", data.section_count);

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
    `Manusia ≤ ${data.thresholds.human} · AI ≥ ${data.thresholds.ai}`,
  );

  renderMarkedText(data.sections || []);
  renderNotes(data.notes);
  renderModelInfo(data.model || {});

  elements.resultSection.classList.remove("hidden");

  elements.resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}
