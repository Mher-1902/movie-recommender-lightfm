const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function getRecommendationsByTitle(title, topK = 10) {
  const params = new URLSearchParams({ title, top_k: String(topK) });
  const res = await fetch(`${API_BASE}/recommendations/by-title?${params.toString()}`);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}
