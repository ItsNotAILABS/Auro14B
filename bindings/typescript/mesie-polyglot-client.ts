/**
 * MESIE Polyglot TypeScript client — call MESIE edge API or embed locally.
 */

export type AISAction = "validate" | "match" | "embed" | "health";

export type AISVectorMessage = {
  action: AISAction;
  runtime: "typescript" | "http";
  record?: Record<string, unknown>;
  record_a?: Record<string, unknown>;
  record_b?: Record<string, unknown>;
};

export type AISVectorResponse = {
  ok: boolean;
  runtime: string;
  action: string;
  data: Record<string, unknown>;
  vector?: number[];
  latency_ms?: number;
  error?: string;
};

export class MESIEPolyglotClient {
  constructor(private baseUrl: string = "", private apiKey: string = "") {}

  async dispatch(msg: AISVectorMessage): Promise<AISVectorResponse> {
    const t0 = performance.now();
    if (!this.baseUrl) {
      return { ok: false, runtime: "typescript", action: msg.action, data: {}, error: "baseUrl required" };
    }
    const path = msg.action === "match" ? "/v1/match" : "/v1/validate";
    const body =
      msg.action === "match"
        ? { reference: msg.record_a, candidate: msg.record_b }
        : { record: msg.record };
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    const res = await fetch(`${this.baseUrl}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
    const data = await res.json();
    return {
      ok: res.ok,
      runtime: "http",
      action: msg.action,
      data,
      latency_ms: performance.now() - t0,
    };
  }
}