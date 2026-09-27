import { analyzeContent } from "./api.js";
import {
  clearError,
  elements,
  setLoading,
  showError,
  updateCounter,
} from "./dom.js";
import { renderResult } from "./render.js";

elements.textInput.addEventListener("input", updateCounter);

elements.fileInput.addEventListener("change", () => {
  elements.fileName.textContent =
    elements.fileInput.files[0]?.name || "Belum ada file";
});

elements.clearButton.addEventListener("click", () => {
  elements.textInput.value = "";
  elements.fileInput.value = "";
  elements.fileName.textContent = "Belum ada file";
  elements.resultSection.classList.add("hidden");

  clearError();
  updateCounter();

  elements.textInput.focus();
});

elements.analyzeButton.addEventListener("click", async () => {
  clearError();

  const text = elements.textInput.value.trim();
  const selectedFile = elements.fileInput.files[0];

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
    const payload = await analyzeContent(form);

    renderResult(payload);
  } catch (error) {
    showError(error.message || "Terjadi kesalahan saat memeriksa teks.");
  } finally {
    setLoading(false);
  }
});

updateCounter();
