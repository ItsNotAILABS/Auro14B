import crypto from "node:crypto";

export class SignalLensRelayClient {
  constructor({ baseUrl, apiKey, fetchFn = fetch }) {
    if (!baseUrl || !apiKey) throw new Error("SignalLens Relay baseUrl and apiKey are required");
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.fetchFn = fetchFn;
  }

  async read(url, mode = "auto") {
    const response = await this.fetchFn(`${this.baseUrl}/v1/read`, {
      method: "POST",
      headers: { authorization: `Bearer ${this.apiKey}`, "content-type": "application/json" },
      body: JSON.stringify({ url, mode })
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || `Relay denied source (${response.status})`);
    if (!payload.receipt?.content_sha256 || !payload.metering?.event_id) throw new Error("Relay response omitted canonical receipt");
    return {
      schema: "signallens.perception.v1",
      source: payload.source,
      content: payload.content,
      relay_receipt: payload.receipt,
      relay_metering: payload.metering,
      perception_sha256: crypto.createHash("sha256").update(JSON.stringify({ source: payload.source, content: payload.content, receipt: payload.receipt })).digest("hex")
    };
  }
}

export function signalLensFromEnv(env = process.env) {
  return new SignalLensRelayClient({ baseUrl: env.NEXUS_RELAY_BASE_URL, apiKey: env.NEXUS_RELAY_API_KEY });
}
