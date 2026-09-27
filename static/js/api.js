export async function analyzeContent(form) {
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

  return payload;
}
