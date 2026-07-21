import { coordinationId, OperatorAgent, TeamAgent, RepositoryAgent, FleetAgent, ApprovalSession, ChatSession } from "./durable";
import { decide } from "./policy";
import type { NormalizedTask, OperationRequest, RhiManifest } from "./contracts";
import { nativeChat, nativeGet, relay, type NativeEnv } from "./native-client";

export { OperatorAgent, TeamAgent, RepositoryAgent, FleetAgent, ApprovalSession, ChatSession };
export { GovernedOperationWorkflow } from "./workflow";

interface Env extends NativeEnv {
  ASSETS: Fetcher;
  PRODUCT_VERSION: string;
  OPERATOR_AGENT: DurableObjectNamespace<OperatorAgent>;
  TEAM_AGENT: DurableObjectNamespace<TeamAgent>;
  REPOSITORY_AGENT: DurableObjectNamespace<RepositoryAgent>;
  FLEET_AGENT: DurableObjectNamespace<FleetAgent>;
  APPROVAL_SESSION: DurableObjectNamespace<ApprovalSession>;
  CHAT_SESSION: DurableObjectNamespace<ChatSession>;
  RHI_MANIFEST_JSON?: string;
}

const jsonHeaders = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer"
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: jsonHeaders });
}

function validateChatBody(value: unknown): { messages: Array<{ role: string; content: string }>; maxTokens: number } {
  if (!value || typeof value !== "object") throw new Error("JSON body required");
  const body = value as { messages?: unknown; message?: unknown; max_tokens?: unknown };
  const messages = Array.isArray(body.messages)
    ? body.messages.map((item) => {
        if (!item || typeof item !== "object") throw new Error("invalid message");
        const row = item as { role?: unknown; content?: unknown };
        const content = String(row.content ?? "").trim();
        if (!content) throw new Error("message content required");
        return { role: String(row.role ?? "user"), content };
      })
    : typeof body.message === "string" && body.message.trim()
      ? [{ role: "user", content: body.message.trim() }]
      : [];
  if (!messages.length) throw new Error("messages or message required");
  if (messages.reduce((sum, item) => sum + item.content.length, 0) > 100_000) throw new Error("request content exceeds 100000 characters");
  const requested = Number(body.max_tokens ?? 1024);
  return { messages, maxTokens: Number.isFinite(requested) ? Math.min(8192, Math.max(1, requested)) : 1024 };
}

function normalizeTask(value: unknown): NormalizedTask {
  const body = (value && typeof value === "object" ? value : {}) as Partial<NormalizedTask>;
  if (!body.operator_id || !body.organization_id || !body.requested_outcome) throw new Error("operator_id, organization_id, and requested_outcome are required");
  return {
    task_id: body.task_id || crypto.randomUUID(),
    operator_id: body.operator_id,
    organization_id: body.organization_id,
    requested_outcome: body.requested_outcome,
    target_resources: body.target_resources || [],
    constraints: body.constraints || [],
    prohibited_actions: body.prohibited_actions || [],
    required_evidence: body.required_evidence || [],
    deadline: body.deadline || null,
    budget: body.budget || { currency: "USD", maximum: 5 }
  };
}

function rhiStatus(env: Env): { status: string; manifest?: RhiManifest; executable: boolean } {
  if (!env.RHI_MANIFEST_JSON) return { status: "DISCOVERED_NAME_UNBOUND", executable: false };
  const manifest = JSON.parse(env.RHI_MANIFEST_JSON) as RhiManifest;
  if (!manifest.native || manifest.externalModelFallback !== false) throw new Error("RHI manifest violates native boundary");
  return { status: "BOUND", executable: true, manifest };
}

function namespaceFor(env: Env, kind: string): DurableObjectNamespace {
  const map: Record<string, DurableObjectNamespace> = {
    operator: env.OPERATOR_AGENT,
    team: env.TEAM_AGENT,
    repository: env.REPOSITORY_AGENT,
    fleet: env.FLEET_AGENT,
    approval: env.APPROVAL_SESSION,
    chat: env.CHAT_SESSION
  };
  const namespace = map[kind];
  if (!namespace) throw new Error("unknown coordination kind");
  return namespace;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      const url = new URL(request.url);
      if (request.method === "GET" && url.pathname === "/api/health") {
        let nativeReady = false;
        try { nativeReady = (await nativeGet(env, "/v1/health/ready")).ok; } catch { nativeReady = false; }
        return json({ ok: true, service: "auro14b-commercial-gateway", native: true, native_ready: nativeReady, product_version: env.PRODUCT_VERSION });
      }
      if (request.method === "GET" && url.pathname === "/api/models") return relay(await nativeGet(env, "/v1/models"));
      if (request.method === "GET" && url.pathname === "/api/capabilities") return relay(await nativeGet(env, "/v1/capabilities"));
      if (request.method === "GET" && url.pathname === "/api/model-card") {
        return json({ product: "Auro14B · HIM", owner: "ItsNotAILABS", compute_plane: "MESIE + native Auro model fleet", external_model_fallback: false, native_endpoint_configured: Boolean(env.AURO_NATIVE_BASE_URL), rhi: rhiStatus(env) });
      }
      if (request.method === "GET" && url.pathname === "/api/rhi") return json(rhiStatus(env));
      if (request.method === "POST" && url.pathname === "/api/chat") {
        const { messages, maxTokens } = validateChatBody(await request.json());
        return relay(await nativeChat(env, messages, maxTokens));
      }
      if (request.method === "POST" && url.pathname === "/api/tasks/normalize") return json(normalizeTask(await request.json()));
      if (request.method === "POST" && url.pathname === "/api/policy/evaluate") {
        const body = await request.json() as { operation: OperationRequest; budgetUsd?: number; maximumScopes?: string[] };
        return json(decide(body.operation, Number(body.budgetUsd ?? 5), body.maximumScopes ?? []));
      }
      if (request.method === "POST" && url.pathname === "/api/coordination/event") {
        const body = await request.json() as { kind: string; key: string; event: { id?: string; kind: string; payload: unknown; idempotencyKey?: string } };
        const namespace = namespaceFor(env, body.kind);
        const stub = namespace.get(namespace.idFromName(body.key)) as unknown as { append(event: unknown): Promise<unknown> };
        return json(await stub.append({ ...body.event, id: body.event.id || crypto.randomUUID(), createdAt: new Date().toISOString() }));
      }
      if (request.method === "GET" && url.pathname === "/api/coordination/ids") {
        return json({ patterns: { operator: coordinationId.operator("{operator_id}"), team: coordinationId.team("{team_id}"), repository: coordinationId.repository("{provider}", "{owner}", "{repository}"), fleet: coordinationId.fleet("{organization_id}"), approval: coordinationId.approval("{task_id}"), chat: coordinationId.chat("{organization_id}", "{session_id}") } });
      }
      if (url.pathname.startsWith("/api/")) return json({ error: "not found" }, 404);
      if (request.method === "GET" || request.method === "HEAD") return env.ASSETS.fetch(request);
      return json({ error: "method not allowed" }, 405);
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "request failed" }, 400);
    }
  }
};
