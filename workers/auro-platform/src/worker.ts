import { Think } from '@cloudflare/think';
import { createExecuteTool } from '@cloudflare/think/tools/execute';
import { routeAgentRequest } from 'agents';
import { createWorkersAI } from 'workers-ai-provider';
export { CodemodeRuntime } from '@cloudflare/codemode';

export interface Env {
  AI: Ai;
  ASSETS: Fetcher;
  AuroPlatform: DurableObjectNamespace<AuroPlatform>;
  BROWSER: Fetcher;
  LOADER: unknown;
  WORKERS_AI_MODEL: string;
  CLOUDFLARE_MCP_URL: string;
  CLOUDFLARE_API_TOKEN?: string;
  ALLOW_CLOUDFLARE_MUTATIONS: string;
}

export class AuroPlatform extends Think<Env> {
  workspaceBash = false;
  extensionLoader = this.env.LOADER as never;

  getModel() {
    return createWorkersAI({ binding: this.env.AI })(this.env.WORKERS_AI_MODEL);
  }

  getSystemPrompt() {
    const mutations = this.env.ALLOW_CLOUDFLARE_MUTATIONS === 'true';
    return `You are Auro Sovereign Platform, a durable Cloudflare-native operator using MESIE reasoning.
Plan before acting. Use Cloudflare API MCP search before execute. Never invent endpoint names.
Cloudflare mutation policy: ${mutations ? 'enabled only when the latest user message contains OPERATOR_APPROVED and the exact resource/action is restated' : 'disabled; search, inspect, explain, and draft only'}.
Never expose tokens, OAuth grants, headers, secrets, or hidden reasoning. For destructive or billing-impacting work, stop and request explicit approval even when mutations are enabled.
Use the durable workspace for plans and receipts. Browser login, MFA, CAPTCHA, payment, domain transfer, deletion, and Zero Trust policy changes require human control.`;
  }

  async onStart() {
    const headers = this.env.CLOUDFLARE_API_TOKEN
      ? { Authorization: `Bearer ${this.env.CLOUDFLARE_API_TOKEN}` }
      : undefined;
    await this.addMcpServer('cloudflare-api', this.env.CLOUDFLARE_MCP_URL, headers ? { transport: { headers } } : undefined);
  }

  getTools() {
    return { execute: createExecuteTool(this) };
  }
}

function json(value: unknown, status = 200) {
  return Response.json(value, { status, headers: { 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' } });
}

export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);
    if (url.pathname === '/health') {
      return json({
        ok: true,
        service: 'auro-sovereign-platform',
        model: env.WORKERS_AI_MODEL,
        cloudflare_api_mcp: env.CLOUDFLARE_MCP_URL,
        mutations_enabled: env.ALLOW_CLOUDFLARE_MUTATIONS === 'true',
        planes: ['workers-ai', 'think-agent', 'durable-object-sqlite', 'api-mcp', 'dynamic-workers', 'browser-run', 'workspace'],
      });
    }
    if (url.pathname === '/api/platform') {
      return json({
        schema: 'auro.cloudflare.platform.v1',
        agent: 'AuroPlatform',
        mcp_tools: ['search', 'execute'],
        mutation_gate: 'ALLOW_CLOUDFLARE_MUTATIONS plus OPERATOR_APPROVED',
        observability: { logs: true, traces: true, trace_sampling: 0.05 },
        claim_boundary: 'Sandbox SDK containers require the separate paid-plan binding; Code Mode and Browser Run use Dynamic Worker isolation in this deployment.',
      });
    }
    const agent = await routeAgentRequest(request, env);
    if (agent) return agent;
    return env.ASSETS.fetch(request);
  },
} satisfies ExportedHandler<Env>;
