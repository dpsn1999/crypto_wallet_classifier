// api/analyze.js
export default async function handler(req, res) {
  const { address, limit = 10 } = req.query;
  const backend = process.env.BACKEND_URL; // your ngrok URL

  if (!backend) {
    return res.status(500).json({ detail: "BACKEND_URL not set" });
  }

  const url = new URL("/api/analyze", backend);
  url.searchParams.set("address", address);
  url.searchParams.set("limit", limit);

  try {
    const r = await fetch(url, { timeout: 30000 });
    const text = await r.text();
    res.status(r.status).send(text);
  } catch (e) {
    res.status(502).json({ detail: "Proxy error: " + e.message });
  }
}
