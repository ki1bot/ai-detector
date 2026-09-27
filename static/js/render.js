import { elements, formatPercent, setBar, setText } from "./dom.js";

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
  const badge = document.getElementById("verdictBadge");

  badge.className = `verdict-badge ${data.verdict_key}`;
  badge.textContent = data.verdict;

  setText(
    "confidenceText",
    `Reliabilitas ${data.confidence.toLowerCase()} · ${data.confidence_score}/100`,
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

  elements.resultSection.classList.remove("hidden");

  elements.resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}
