import { matchRecords, validateRecord } from "./spectral";

export interface Env {
  MESIE_API_KEY?: string;
  MESIE_PUBLIC?: string;
  MESIE_DATA?: R2Bucket;
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-MESIE-Key",
};

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

function unauthorized(): Response {
  return json({ error: "Unauthorized. Set Authorization: Bearer <key> or X-MESIE-Key." }, 401);
}

function requireAuth(request: Request, env: Env): boolean {
  if (env.MESIE_PUBLIC === "true") return true;
  const expected = env.MESIE_API_KEY;
  if (!expected) return true;

  const header =
    request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "") ??
    request.headers.get("X-MESIE-Key");
  return header === expected;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    if (path === "/" || path === "/health") {
      return json({
        service: "mesie-api",
        version: "0.2.0",
        status: "ok",
        engine: "cloudflare-workers",
      });
    }

    if (path === "/v1" || path === "/v1/") {
      return json({
        endpoints: [
          "GET /health",
          "GET /v1/datasets",
          "POST /v1/validate",
          "POST /v1/match",
        ],
        python_sdk: "https://github.com/FreddyCreates/Multi-Element-Spectral-Intelligence-Engine-MESIE-",
      });
    }

    if (path === "/v1/datasets") {
      return json({
        references: [
          "earthquake_psd_reference",
          "structural_fas_reference",
          "rotdnn_reference",
          "vibration_monitoring_reference",
        ],
        benchmarks: [
          "embedding_training_data",
          "spectral_classification_benchmark",
        ],
        note: "Full records ship with pip package `mesie` (data/). R2 binding optional for edge-hosted corpora.",
      });
    }

    if (path === "/v1/validate" || path === "/v1/match") {
      if (!requireAuth(request, env)) return unauthorized();
      if (request.method !== "POST") {
        return json({ error: "Method not allowed. Use POST." }, 405);
      }

      let body: unknown;
      try {
        body = await request.json();
      } catch {
        return json({ error: "Invalid JSON body." }, 400);
      }

      if (path === "/v1/validate") {
        return json(validateRecord(body));
      }

      const payload = body as { reference?: unknown; candidate?: unknown };
      if (!payload.reference || !payload.candidate) {
        return json({ error: "Body must include 'reference' and 'candidate' records." }, 400);
      }
      const refCheck = validateRecord(payload.reference);
      const candCheck = validateRecord(payload.candidate);
      if (!refCheck.is_valid || !candCheck.is_valid) {
        return json(
          {
            error: "Invalid record(s).",
            reference: refCheck,
            candidate: candCheck,
          },
          422,
        );
      }
      return json(
        matchRecords(
          payload.reference as Parameters<typeof matchRecords>[0],
          payload.candidate as Parameters<typeof matchRecords>[1],
        ),
      );
    }

    return json({ error: "Not found", path }, 404);
  },
};