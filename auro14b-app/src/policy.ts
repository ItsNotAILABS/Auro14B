import type { LoveAnchorEvaluation, OperationRequest, PolicyDecision, RiskClass } from "./contracts";

const endpointAllowlist = new Set([
  "workers.scripts.update",
  "workers.scripts.get",
  "r2.objects.put",
  "repositories.read",
  "repositories.write",
  "benchmarks.run",
  "releases.publish"
]);

function classify(request: OperationRequest): RiskClass {
  if (request.destructive && request.environment === "production") return 5;
  if (request.environment === "production") return 4;
  if (!["GET", "HEAD"].includes(request.method.toUpperCase())) return 3;
  if (request.endpoint.includes("publish") || request.endpoint.includes("generate")) return 2;
  return 1;
}

function loveAnchor(request: OperationRequest): LoveAnchorEvaluation {
  return {
    truthful_representation: request.targetUnambiguous,
    operator_agency_preserved: request.approvalCount > 0 || classify(request) < 3,
    affected_parties_identified: request.resourceIds.length > 0,
    data_extraction_disclosed: true,
    reversal_available: !request.destructive || request.rollbackAvailable,
    dependency_created: false,
    asymmetry_justified: request.authorityVerified
  };
}

export function decide(request: OperationRequest, budgetUsd: number, maximumScopes: string[]): PolicyDecision {
  const riskClass = classify(request);
  const reasons: string[] = [];
  const anchor = loveAnchor(request);

  if (!request.targetUnambiguous) reasons.push("target_ambiguous");
  if (!request.authorityVerified) reasons.push("authority_unverified");
  if (!endpointAllowlist.has(request.endpoint)) reasons.push("endpoint_not_allowlisted");
  if (request.estimatedCostUsd > budgetUsd) reasons.push("budget_exceeded");
  if (!request.beforeStateCaptured && riskClass >= 3) reasons.push("before_state_missing");
  if (request.destructive && !request.rollbackAvailable) reasons.push("rollback_unavailable");
  if (request.modifiesPolicyEngine) reasons.push("self_policy_modification_denied");
  if (request.requestedScopes.some((scope) => !maximumScopes.includes(scope))) reasons.push("scope_exceeds_policy");
  if (riskClass === 4 && request.approvalCount < 1) reasons.push("human_approval_required");
  if (riskClass === 5 && request.approvalCount < 2) reasons.push("two_party_approval_required");
  if (riskClass === 5 && !request.freshAuthentication) reasons.push("fresh_authentication_required");
  if (!anchor.truthful_representation || !anchor.operator_agency_preserved || !anchor.reversal_available) reasons.push("love_anchor_failed");

  return {
    allowed: reasons.length === 0,
    riskClass,
    approvalRequired: riskClass >= 3,
    approvalCount: riskClass === 5 ? 2 : riskClass >= 3 ? 1 : 0,
    reasons,
    maximumScope: maximumScopes,
    expiresAt: new Date(Date.now() + 5 * 60 * 1000).toISOString(),
    loveAnchor: anchor
  };
}
