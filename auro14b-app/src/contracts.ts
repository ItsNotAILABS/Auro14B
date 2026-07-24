export type RiskClass = 0 | 1 | 2 | 3 | 4 | 5;

export interface NormalizedTask {
  task_id: string;
  operator_id: string;
  organization_id: string;
  requested_outcome: string;
  target_resources: string[];
  constraints: string[];
  prohibited_actions: string[];
  required_evidence: string[];
  deadline: string | null;
  budget: { currency: "USD"; maximum: number };
}

export interface OperationRequest {
  taskId: string;
  actorId: string;
  organizationId: string;
  endpoint: string;
  method: string;
  resourceIds: string[];
  environment: "development" | "staging" | "production";
  estimatedCostUsd: number;
  requestedScopes: string[];
  destructive: boolean;
  targetUnambiguous: boolean;
  authorityVerified: boolean;
  beforeStateCaptured: boolean;
  rollbackAvailable: boolean;
  modifiesPolicyEngine: boolean;
  approvalCount: number;
  freshAuthentication: boolean;
}

export interface LoveAnchorEvaluation {
  truthful_representation: boolean;
  operator_agency_preserved: boolean;
  affected_parties_identified: boolean;
  data_extraction_disclosed: boolean;
  reversal_available: boolean;
  dependency_created: boolean;
  asymmetry_justified: boolean;
}

export interface PolicyDecision {
  allowed: boolean;
  riskClass: RiskClass;
  approvalRequired: boolean;
  approvalCount: number;
  reasons: string[];
  maximumScope: string[];
  expiresAt: string;
  loveAnchor: LoveAnchorEvaluation;
}

export interface RhiManifest {
  id: string;
  version: string;
  repository: { provider: "github"; owner: string; name: string; commit: string };
  role: "model" | "reasoning-organ" | "human-interface" | "memory" | "orchestrator" | "governance" | "multimodal-organ";
  capabilities: string[];
  requiredBindings: string[];
  requiredScopes: string[];
  modelFamilies: string[];
  modalities: Array<"text" | "code" | "image" | "audio" | "video" | "document">;
  native: true;
  externalModelFallback: false;
}
