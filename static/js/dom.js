export const elements = {
  textInput: document.getElementById("textInput"),
  fileInput: document.getElementById("fileInput"),
  analyzeButton: document.getElementById("analyzeButton"),
  clearButton: document.getElementById("clearButton"),
  counter: document.getElementById("counter"),
  fileName: document.getElementById("fileName"),
  errorBox: document.getElementById("errorBox"),
  resultSection: document.getElementById("resultSection"),
};

export function countWords(text) {
  const matches = text.trim().match(/[\p{L}\p{N}_'-]+/gu);

  return matches ? matches.length : 0;
}

export function setText(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.textContent = value;
  }
}

export function setBar(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.style.width = `${Math.max(0, Math.min(100, value))}%`;
  }
}

export function formatPercent(value) {
  if (typeof value !== "number") {
    return "-";
  }

  return `${(value * 100).toFixed(1)}%`;
}

export function updateCounter() {
  elements.counter.textContent = `${countWords(elements.textInput.value)} kata`;
}

export function clearError() {
  elements.errorBox.textContent = "";
  elements.errorBox.classList.add("hidden");
}

export function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.classList.remove("hidden");
}

export function setLoading(active) {
  elements.analyzeButton.disabled = active;
  elements.clearButton.disabled = active;
  elements.fileInput.disabled = active;
  elements.analyzeButton.textContent = active ? "Memeriksa..." : "Periksa teks";
}
