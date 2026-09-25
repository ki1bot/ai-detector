const textInput = document.getElementById("textInput");
const fileInput = document.getElementById("fileInput");
const analyzeButton = document.getElementById("analyzeButton");
const counter = document.getElementById("counter");
const fileName = document.getElementById("fileName");
const errorBox = document.getElementById("errorBox");
const resultSection = document.getElementById("resultSection");

function countWords(text) {
  const matches = text.trim().match(/[\p{L}\p{N}_'-]+/gu);

  return matches ? matches.length : 0;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

textInput.addEventListener("input", () => {
  counter.textContent = `${countWords(textInput.value)} kata`;
});

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0]?.name || "Belum ada file";
});

function setLoading(active) {
  analyzeButton.disabled = active;

  analyzeButton.textContent = active ? "Menganalisis..." : "Analisis teks";
}

function showError(message) {
  errorBox.textContent = message;

  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";

  errorBox.classList.add("hidden");
}

function renderResult(data) {
  document.getElementById("aiScore").textContent = `${data.ai_score}%`;

  document.getElementById("humanScore").textContent = `${data.human_score}%`;

  document.getElementById("verdict").textContent = data.label;

  document.getElementById("reliability").textContent = data.reliability;

  document.getElementById("documentStats").textContent =
    `${data.word_count} kata · ` +
    `${data.character_count} karakter · ` +
    `${data.section_count} bagian`;

  document.getElementById("meterFill").style.width = `${data.ai_score}%`;

  const warningPanel = document.getElementById("warningPanel");

  const warningList = document.getElementById("warningList");

  warningList.innerHTML = "";

  const warnings = [...data.warnings, data.disclaimer];

  if (warnings.length) {
    warnings.forEach((warning) => {
      const item = document.createElement("li");

      item.textContent = warning;

      warningList.appendChild(item);
    });

    warningPanel.classList.remove("hidden");
  } else {
    warningPanel.classList.add("hidden");
  }

  const sectionsList = document.getElementById("sectionsList");

  sectionsList.innerHTML = data.sections
    .map(
      (section) => `
                    <article class="section-card ${section.label_key}">
                        <div class="section-topline">
                            <strong>
                                Bagian ${section.index}
                            </strong>

                            <span>
                                ${section.label}
                                ·
                                ${section.ai_score}% AI
                            </span>
                        </div>

                        <p>
                            ${escapeHtml(section.text)}
                        </p>

                        <small>
                            ${section.word_count} kata
                        </small>
                    </article>
                `,
    )
    .join("");

  const metrics = data.model_metrics || {};

  const metricParts = [];

  if (typeof metrics.accuracy === "number") {
    metricParts.push(
      `Accuracy holdout ${(metrics.accuracy * 100).toFixed(2)}%`,
    );
  }

  if (typeof metrics.f1 === "number") {
    metricParts.push(`F1 ${(metrics.f1 * 100).toFixed(2)}%`);
  }

  if (typeof metrics.roc_auc === "number") {
    metricParts.push(`ROC-AUC ${(metrics.roc_auc * 100).toFixed(2)}%`);
  }

  if (metrics.samples_test) {
    metricParts.push(`${metrics.samples_test} sampel uji`);
  }

  document.getElementById("modelInfo").textContent = metricParts.length
    ? metricParts.join(" · ")
    : "Metadata evaluasi belum tersedia.";

  resultSection.classList.remove("hidden");

  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

analyzeButton.addEventListener("click", async () => {
  clearError();

  const form = new FormData();

  const text = textInput.value.trim();

  if (text) {
    form.append("text", text);
  }

  if (fileInput.files[0]) {
    form.append("file", fileInput.files[0]);
  }

  if (!text && !fileInput.files[0]) {
    showError("Masukkan teks atau pilih dokumen terlebih dahulu.");

    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Analisis gagal.");
    }

    renderResult(payload);
  } catch (error) {
    showError(error.message || "Terjadi kesalahan saat menganalisis teks.");
  } finally {
    setLoading(false);
  }
});
