async function digest(value) {
  const bytes = new TextEncoder().encode(value);
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

self.onmessage = async ({ data }) => {
  const { id, task, payload = {} } = data;
  try {
    let result;
    if (task === 'process') {
      const text = String(payload.text || '');
      result = { chars: text.length, words: text.trim() ? text.trim().split(/\s+/).length : 0, sha256: await digest(text) };
    } else if (task === 'crawl') {
      const response = await fetch(String(payload.url), { credentials: 'omit', redirect: 'follow' });
      const text = (await response.text()).slice(0, Number(payload.maxChars || 200000));
      result = { url: response.url, status: response.status, contentType: response.headers.get('content-type'), text, sha256: await digest(text), truncated: text.length >= Number(payload.maxChars || 200000) };
    } else if (task === 'agent') {
      const response = await fetch('/v1/respond', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ message: String(payload.objective), execute: false }) });
      result = await response.json();
      if (!response.ok) throw new Error(result?.error?.message || `Agent HTTP ${response.status}`);
    } else {
      throw new Error(`Unknown worker task: ${task}`);
    }
    self.postMessage({ id, ok: true, task, result });
  } catch (error) {
    self.postMessage({ id, ok: false, task, error: String(error?.message || error) });
  }
};
