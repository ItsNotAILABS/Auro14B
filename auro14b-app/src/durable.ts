import { DurableObject } from "cloudflare:workers";

export type AgentEnv = { PRODUCT_VERSION: string };

export type CoordinationEvent = {
  id: string;
  kind: string;
  payload: unknown;
  createdAt: string;
  idempotencyKey?: string;
};

abstract class CoordinationUnit extends DurableObject<AgentEnv> {
  constructor(ctx: DurableObjectState, env: AgentEnv) {
    super(ctx, env);
    ctx.storage.sql.exec(`
      CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        idempotency_key TEXT UNIQUE
      );
      CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
    `);
  }

  async append(event: CoordinationEvent): Promise<{ accepted: boolean; event: CoordinationEvent }> {
    if (event.idempotencyKey) {
      const existing = this.ctx.storage.sql
        .exec("SELECT id, kind, payload, created_at, idempotency_key FROM events WHERE idempotency_key = ? LIMIT 1", event.idempotencyKey)
        .toArray()[0] as Record<string, unknown> | undefined;
      if (existing) return { accepted: false, event: this.row(existing) };
    }

    this.ctx.storage.sql.exec(
      "INSERT INTO events (id, kind, payload, created_at, idempotency_key) VALUES (?, ?, ?, ?, ?)",
      event.id,
      event.kind,
      JSON.stringify(event.payload ?? null),
      event.createdAt,
      event.idempotencyKey ?? null
    );
    return { accepted: true, event };
  }

  async recent(limit = 50): Promise<CoordinationEvent[]> {
    const bounded = Math.max(1, Math.min(200, Math.trunc(limit)));
    return this.ctx.storage.sql
      .exec("SELECT id, kind, payload, created_at, idempotency_key FROM events ORDER BY created_at DESC LIMIT ?", bounded)
      .toArray()
      .map((row) => this.row(row as Record<string, unknown>));
  }

  async snapshot(): Promise<{ object: string; version: string; events: CoordinationEvent[] }> {
    return { object: this.constructor.name, version: this.env.PRODUCT_VERSION, events: await this.recent(20) };
  }

  async alarm(): Promise<void> {
    const cutoff = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    this.ctx.storage.sql.exec("DELETE FROM events WHERE created_at < ?", cutoff);
  }

  private row(row: Record<string, unknown>): CoordinationEvent {
    return {
      id: String(row.id),
      kind: String(row.kind),
      payload: JSON.parse(String(row.payload)),
      createdAt: String(row.created_at),
      idempotencyKey: row.idempotency_key ? String(row.idempotency_key) : undefined
    };
  }
}

export class OperatorAgent extends CoordinationUnit {}
export class TeamAgent extends CoordinationUnit {}
export class RepositoryAgent extends CoordinationUnit {}
export class FleetAgent extends CoordinationUnit {}
export class ApprovalSession extends CoordinationUnit {}
export class ChatSession extends CoordinationUnit {}

export const coordinationId = {
  operator: (operatorId: string) => `operator:${operatorId}`,
  team: (teamId: string) => `team:${teamId}`,
  repository: (provider: string, owner: string, repository: string) => `repo:${provider}:${owner}:${repository}`,
  fleet: (organizationId: string) => `fleet:${organizationId}`,
  approval: (taskId: string) => `approval:${taskId}`,
  chat: (organizationId: string, sessionId: string) => `chat:${organizationId}:${sessionId}`
} as const;
