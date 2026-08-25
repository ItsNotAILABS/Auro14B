"""
Julia mirror of the AURO family contract.
Contract version 2.1.0. Architecture and composition only; no checkpoint claim.
"""
module AuroFamily

export CONTRACT_VERSION, FAMILY_ID, FAMILY_MODEL_IDS, SPECIALIST_TRIAD
export ModelTier, SubAgentRole, ModelLane
export model_id_for_tier, tier_for_model_id, parameter_target
export can_host, resolve_child_model_id, family_model_ids, builtin_lanes

const CONTRACT_VERSION = "2.1.0"
const FAMILY_ID = "Auro"

@enum ModelTier begin
    ATOMIC
    EDGE
    SPECIALIST
    GENERAL
    ORCHESTRATOR
    FRONTIER
end

@enum SubAgentRole begin
    ROUTING_SEED
    CLASSIFIER
    JSON_REPAIR
    TOOL_SELECTION
    INTENT_EXTRACT
    RETRIEVAL_FILTER
    STRUCTURED_TRANSFORM
    CODE_TRIAGE
    MEMORY_CONSOLIDATION
    SEMANTIC_OUTLINE
    TOOL_EXECUTION_PLAN
    CODE_PATCH
    EVIDENCE_REVIEW
    LOCAL_WORKER
    EXPERT_CONSENSUS
    TEXT_EXPANSION
    ROUTER
    TOOL_CALL
    EMBED_FAST
    SPECTRAL_TRIAGE
    CODE_EDIT
    SPECTRAL_MATCH
    JSON_STRUCT
    TOOL_PLAN
    REASON
    PLAN
    CRITIQUE
    SPECTRAL_EXPLAIN
    ORCHESTRATOR_ROLE
    COUNCIL_CHAIR
    INSTRUCT_DEV
    MULTI_AGENT_ROUTER
    FRONTIER_RESEARCH
    LONG_HORIZON
    SAFETY_REVIEW
    DEEP_COUNCIL
end

struct ModelLane
    model_id::String
    parameter_target::Int
    tier::ModelTier
    can_embed_subagents::Bool
    embeddable_tiers::Vector{ModelTier}
    roles::Vector{SubAgentRole}
end

const TIER_RANK = Dict(ATOMIC=>0, EDGE=>1, SPECIALIST=>2, GENERAL=>3, ORCHESTRATOR=>4, FRONTIER=>5)
const TIER_TO_MODEL = Dict(ATOMIC=>"Auro-500M", EDGE=>"Auro-2B", SPECIALIST=>"Auro-4B", GENERAL=>"Auro-8B", ORCHESTRATOR=>"Auro-14B", FRONTIER=>"Auro-100B")
const FAMILY_MODEL_IDS = ["Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"]
const MODEL_TO_TIER = Dict("Auro-156K"=>ATOMIC, "Auro-250M"=>ATOMIC, "Auro-500M"=>ATOMIC, "Auro-2B"=>EDGE, "Auro-4B"=>SPECIALIST, "Auro-8B"=>GENERAL, "Auro-14B"=>ORCHESTRATOR, "Auro-100B"=>FRONTIER)
const PARAM_TARGETS = Dict("Auro-156K"=>156_000, "Auro-250M"=>250_000_000, "Auro-500M"=>500_000_000, "Auro-2B"=>2_000_000_000, "Auro-4B"=>4_000_000_000, "Auro-8B"=>8_000_000_000, "Auro-14B"=>14_000_000_000, "Auro-100B"=>100_000_000_000)
const SPECIALIST_TRIAD = ["Auro-500M-SENSUS", "Auro-500M-PRAXIS", "Auro-500M-VERBUM"]

const ROLE_DEFAULT_MODEL = Dict(
    ROUTING_SEED=>"Auro-156K", CLASSIFIER=>"Auro-156K", JSON_REPAIR=>"Auro-156K", TOOL_SELECTION=>"Auro-156K",
    INTENT_EXTRACT=>"Auro-250M", RETRIEVAL_FILTER=>"Auro-250M", STRUCTURED_TRANSFORM=>"Auro-250M", CODE_TRIAGE=>"Auro-250M", MEMORY_CONSOLIDATION=>"Auro-250M", SEMANTIC_OUTLINE=>"Auro-250M",
    TOOL_EXECUTION_PLAN=>"Auro-500M", CODE_PATCH=>"Auro-500M", EVIDENCE_REVIEW=>"Auro-500M", LOCAL_WORKER=>"Auro-500M", EXPERT_CONSENSUS=>"Auro-500M", TEXT_EXPANSION=>"Auro-500M",
    ROUTER=>"Auro-2B", TOOL_CALL=>"Auro-2B", EMBED_FAST=>"Auro-2B", SPECTRAL_TRIAGE=>"Auro-2B",
    CODE_EDIT=>"Auro-4B", SPECTRAL_MATCH=>"Auro-4B", JSON_STRUCT=>"Auro-4B", TOOL_PLAN=>"Auro-4B",
    REASON=>"Auro-8B", PLAN=>"Auro-8B", CRITIQUE=>"Auro-8B", SPECTRAL_EXPLAIN=>"Auro-8B",
    ORCHESTRATOR_ROLE=>"Auro-14B", COUNCIL_CHAIR=>"Auro-14B", INSTRUCT_DEV=>"Auro-14B", MULTI_AGENT_ROUTER=>"Auro-14B",
    FRONTIER_RESEARCH=>"Auro-100B", LONG_HORIZON=>"Auro-100B", SAFETY_REVIEW=>"Auro-100B", DEEP_COUNCIL=>"Auro-100B"
)

model_id_for_tier(t::ModelTier) = TIER_TO_MODEL[t]
tier_for_model_id(id::AbstractString) = MODEL_TO_TIER[String(id)]
parameter_target(id::AbstractString) = PARAM_TARGETS[String(id)]
family_model_ids() = copy(FAMILY_MODEL_IDS)
can_host(parent_tier::ModelTier, child_tier::ModelTier) = TIER_RANK[parent_tier] > TIER_RANK[child_tier]

function resolve_child_model_id(parent_model_id::AbstractString, role::SubAgentRole)::String
    child = ROLE_DEFAULT_MODEL[role]
    can_host(tier_for_model_id(parent_model_id), tier_for_model_id(child)) || error("incompatible parent and child lane")
    return child
end

function builtin_lanes()::Vector{ModelLane}
    return [
        ModelLane("Auro-156K", PARAM_TARGETS["Auro-156K"], ATOMIC, false, ModelTier[], [ROUTING_SEED, CLASSIFIER, JSON_REPAIR, TOOL_SELECTION]),
        ModelLane("Auro-250M", PARAM_TARGETS["Auro-250M"], ATOMIC, false, ModelTier[], [INTENT_EXTRACT, RETRIEVAL_FILTER, STRUCTURED_TRANSFORM, CODE_TRIAGE, MEMORY_CONSOLIDATION, SEMANTIC_OUTLINE]),
        ModelLane("Auro-500M", PARAM_TARGETS["Auro-500M"], ATOMIC, false, ModelTier[], [TOOL_EXECUTION_PLAN, CODE_PATCH, EVIDENCE_REVIEW, LOCAL_WORKER, EXPERT_CONSENSUS, TEXT_EXPANSION]),
        ModelLane("Auro-2B", PARAM_TARGETS["Auro-2B"], EDGE, true, [ATOMIC], [ROUTER, TOOL_CALL, EMBED_FAST, SPECTRAL_TRIAGE]),
        ModelLane("Auro-4B", PARAM_TARGETS["Auro-4B"], SPECIALIST, true, [ATOMIC, EDGE], [CODE_EDIT, SPECTRAL_MATCH, JSON_STRUCT, TOOL_PLAN]),
        ModelLane("Auro-8B", PARAM_TARGETS["Auro-8B"], GENERAL, true, [ATOMIC, EDGE, SPECIALIST], [REASON, PLAN, CRITIQUE, SPECTRAL_EXPLAIN]),
        ModelLane("Auro-14B", PARAM_TARGETS["Auro-14B"], ORCHESTRATOR, true, [ATOMIC, EDGE, SPECIALIST, GENERAL], [ORCHESTRATOR_ROLE, COUNCIL_CHAIR, INSTRUCT_DEV, MULTI_AGENT_ROUTER]),
        ModelLane("Auro-100B", PARAM_TARGETS["Auro-100B"], FRONTIER, true, [ATOMIC, EDGE, SPECIALIST, GENERAL, ORCHESTRATOR], [FRONTIER_RESEARCH, LONG_HORIZON, SAFETY_REVIEW, DEEP_COUNCIL])
    ]
end

end
