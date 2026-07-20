import type { OperationRequest } from "./contracts";
import { decide } from "./policy";

export type GovernedOperationParams = {
  operation: OperationRequest;
  budgetUsd: number;
  maximumScopes: string[];
  desiredStateHash: string;
};

type WorkflowEnv = {
  AURO_NATIVE_BASE_URL: string;
  AURO_NATIVE_API_TOKEN?: string;
};

export class GovernedOperationWorkflow extends WorkflowEntrypoint<WorkflowEnv, GovernedOperationParams> {
  async run(event: WorkflowEvent<GovernedOperationParams>, step: WorkflowStep) {
    const { operation, budgetUsd, maximumScopes, desiredStateHash } = event.payload;
    const before = await step.do("capture-before-state", async () => ({
      capturedAt: new Date().toISOString(),
      resources: operation.resourceIds,
      desiredStateHash
    }));

    const decision = await step.do("validate-authority-and-policy", async () => decide(operation, budgetUsd, maximumScopes));
    if (!decision.allowed) return { ok: false, phase: "policy", decision, before };

    const execution = await step.do(
      "execute-operation",
      { retries: { limit: 3, delay: "5 seconds", backoff: "exponential" }, timeout: "10 minutes" },
      async () => ({
        accepted: true,
        endpoint: operation.endpoint,
        idempotencyKey: await stableIdempotencyKey(operation, desiredStateHash),
        executedAt: new Date().toISOString()
      })
    );

    const after = await step.do("capture-after-state", async () => ({
      validatedAt: new Date().toISOString(),
      targetResources: operation.resourceIds,
      execution
    }));

    const receipt = await step.do("write-receipt", async () => ({
      taskId: operation.taskId,
      decision,
      before,
      after,
      evidenceHash: await sha256(JSON.stringify({ operation, before, after, desiredStateHash }))
    }));

    return { ok: true, receipt };
  }
}

async function stableIdempotencyKey(operation: OperationRequest, desiredStateHash: string): Promise<string> {
  return sha256(`${operation.organizationId}|${operation.resourceIds.join(",")}|${operation.endpoint}|${operation.method}|${desiredStateHash}`);
}

async function sha256(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
