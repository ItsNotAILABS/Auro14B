type ModelLane = "hosted-compatibility" | "auro-reference" | "auro-promoted";

type Env = {
  AI: Ai;
  ASSETS: Fetcher;
  MODEL_LANE: ModelLane;
  MODEL_ID: string;
  PRODUCT_VERSION: string;
};

const headers = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer"
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers });
}

function modelStatus(env: Env) {
  const native = env.MODEL_LANE === "auro-promoted";
  return {
    product: "Auro14B",
    product_version: env.PRODUCT_VERSION,
    lane: env.MODEL_LANE,
    provider_model: env.MODEL_ID,
    auro_native: native,
    claim: native
      ? "Promoted Auro checkpoint; verify release receipt and hashes."
      : "Operational compatibility lane; not represented as trained Auro14B weights."
  };
}

function validateChatBody(value: unknown): { messages: Array<{ role: string; content: string }>; maxTokens: number } {
  if (!value || typeof value !== "object") throw new Error("JSON body required");
  const body = value as { messages?: unknown; message?: unknown; max_tokens?: unknown };
  let messages: Array<{ role: string; content: string }> = [];
  if (Array.isArray(body.messages)) {
    messages = body.messages.map((item) => {
      if (!item || typeof item !== "object") throw new Error("invalid message");
      const row = item as { role?: unknown; content?: unknown };
      const content = String(row.content ?? "").trim();
      if (!content) throw new Error("message content required");
      return { role: String(row.role ?? "user"), content };
    });
  } else if (typeof body.message === "string" && body.message.trim()) {
    messages = [{ role: "user", content: body.message.trim() }];
  }
  if (!messages.length) throw new Error("messages or message required");
  const total = messages.reduce((sum, item) => sum + item.content.length, 0);
  if (total > 24000) throw new Error("request content exceeds 24000 characters");
  const requested = Number(body.max_tokens ?? 512);
  const maxTokens = Number.isFinite(requested) ? Math.min(2048, Math.max(1, requested)) : 512;
  return { messages, maxTokens };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      const url = new URL(request.url);
      if (request.method === "GET" && url.pathname === "/api/health") {
        return json({ ok: true, service: "auro14b-app", model: modelStatus(env) });
      }
      if (request.method === "GET" && url.pathname === "/api/model-card") {
        return json({
          ...modelStatus(env),
          trained_checkpoint_verified: env.MODEL_LANE === "auro-promoted",
          tokenizer_verified: env.MODEL_LANE === "auro-promoted",
          official_benchmarks_verified: false,
          unresolved_blockers: env.MODEL_LANE === "auro-promoted" ? ["Official benchmark receipt not attached to runtime"] : ["Auro14B promoted checkpoint is not active"]
        });
      }
      if (request.method === "GET" && url.pathname === "/api/capabilities") {
        return json({
          chat: true,
          streaming: false,
          receipts: "response-local",
          public_site: true,
          github_pages_mirror: true,
          model_lane: env.MODEL_LANE
        });
      }
      if (request.method === "POST" && url.pathname === "/api/chat") {
        const { messages, maxTokens } = validateChatBody(await request.json());
        const output = await env.AI.run(env.MODEL_ID as keyof AiModels, { messages, max_tokens: maxTokens } as never) as { response?: string };
        const content = output.response ?? "";
        const receipt = {
          id: crypto.randomUUID(),
          created_at: new Date().toISOString(),
          model_lane: env.MODEL_LANE,
          provider_model: env.MODEL_ID,
          request_messages: messages.length,
          response_characters: content.length
        };
        return json({ ok: true, answer: content, model: modelStatus(env), receipt });
      }
      if (url.pathname.startsWith("/api/")) return json({ error: "not found" }, 404);
      if (request.method === "GET" || request.method === "HEAD") return env.ASSETS.fetch(request);
      return json({ error: "method not allowed" }, 405);
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "request failed" }, 400);
    }
  }
};
