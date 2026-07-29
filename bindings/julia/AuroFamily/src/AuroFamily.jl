"""
AuroFamily — Julia contract mirroring `auro_native_llm.types`.

Contract version: 2.0.0
Family: 156K / 250M / 500M / 2B / 4B / 8B / 14B / 100B.
Architecture and routing contracts do not imply trained checkpoint evidence.
"""
module AuroFamily

export CONTRACT_VERSION, FAMILY_ID
export ModelTier, SubAgentRole
export ArchitectureSpec, ModelLane, SubAgentSpec, SubAgentDispatch
export model_id_for_tier, tier_for_model_id, parameter_target
export can_host, can_host_models, resolve_child_model_id, family_model_ids, builtin_lanes

const CONTRACT_VERSION = "2.0.0"
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
    STYLE_GUARD
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
    CREATIVE_BRANCH
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

struct ArchitectureSpec
    hidden_size::Int
    layers::Int
    attention_heads::Int
    kv_heads::Int
    intermediate_size::Int
    context_window_tokens_target::Int
    vocab_size_target::Int
end

struct ModelLane
    model_id::String
    parameter_target::Int
    tier::ModelTier
    can_embed_subagents::Bool
    embeddable_tiers::Vector{ModelTier}
    roles::Vector{SubAgentRole}
end

struct SubAgentSpec
    agent_id::String
    role::SubAgentRole
    child_model_id::String
    parent_model_id::String
    task_id::String
    intent::String
end

struct SubAgentDispatch
    ok::Bool
    parent_model_id::String
    child_model_id::String
    role::SubAgentRole
    agent_id::String
    task_id::String
    message::String
    error::Union{String,Nothing}
end

const TIER_RANK = Dict(ATOMIC => 0, EDGE => 1, SPECIALIST => 2, GENERAL => 3, ORCHESTRATOR => 4, FRONTIER => 5)
const TIER_TO_MODEL = Dict(ATOMIC => "Auro-500M", EDGE => "Auro-2B", SPECIALIST => "Auro-4B", GENERAL => "Auro-8B", ORCHESTRATOR => "Auro-14B", FRONTIER => "Auro-100B")
const MODEL_TO_TIER = Dict(
    "Auro-156K" => ATOMIC,
    "Auro-250M" => ATOMIC,
    "Auro-500M" => ATOMIC,
    "Auro-500M-SENSUS" => ATOMIC,
    "Auro-500M-PRAXIS" => ATOMIC,
    "Auro-500M-VERBUM" => ATOMIC,
    "Auro-2B" => EDGE,
    "Auro-4B" => SPECIALIST,
    "Auro-8B" => GENERAL,
    "Auro-14B" => ORCHESTRATOR,
    "Auro-100B" => FRONTIER,
)
const PARAM_TARGETS = Dict(
    "Auro-156K" => 156_000,
    "Auro-250M" => 250_000_000,
    "Auro-500M" => 500_000_000,
    "Auro-500M-SENSUS" => 500_000_000,
    "Auro-500M-PRAXIS" => 500_000_000,
    "Auro-500M-VERBUM" => 500_000_000,
    "Auro-2B" => 2_000_000_000,
    "Auro-4B" => 4_000_000_000,
    "Auro-8B" => 8_000_000_000,
    "Auro-14B" => 14_000_000_000,
    "Auro-100B" => 100_000_000_000,
)

const ATOMIC_ROLE_MODEL = Dict(
    ROUTING_SEED => "Auro-156K", CLASSIFIER => "Auro-156K", JSON_REPAIR => "Auro-156K",
    TOOL_SELECTION => "Auro-156K", STYLE_GUARD => "Auro-156K",
    INTENT_EXTRACT => "Auro-250M", RETRIEVAL_FILTER => "Auro-250M",
    STRUCTURED_TRANSFORM => "Auro-250M", CODE_TRIAGE => "Auro-250M",
    MEMORY_CONSOLIDATION => "Auro-250M", SEMANTIC_OUTLINE => "Auro-250M",
    TOOL_EXECUTION_PLAN => "Auro-500M", CODE_PATCH => "Auro-500M",
    EVIDENCE_REVIEW => "Auro-500M", LOCAL_WORKER => "Auro-500M",
    EXPERT_CONSENSUS => "Auro-500M", TEXT_EXPANSION => "Auro-500M", CREATIVE_BRANCH => "Auro-500M",
)

const ROLE_DEFAULT_TIER = Dict(
    (role => ATOMIC for role in keys(ATOMIC_ROLE_MODEL))...,
    ROUTER => EDGE, TOOL_CALL => EDGE, EMBED_FAST => EDGE, SPECTRAL_TRIAGE => EDGE,
    CODE_EDIT => SPECIALIST, SPECTRAL_MATCH => SPECIALIST, JSON_STRUCT => SPECIALIST, TOOL_PLAN => SPECIALIST,
    REASON => GENERAL, PLAN => GENERAL, CRITIQUE => GENERAL, SPECTRAL_EXPLAIN => GENERAL,
    ORCHESTRATOR_ROLE => ORCHESTRATOR, COUNCIL_CHAIR => ORCHESTRATOR,
    INSTRUCT_DEV => ORCHESTRATOR, MULTI_AGENT_ROUTER => ORCHESTRATOR,
    FRONTIER_RESEARCH => FRONTIER, LONG_HORIZON => FRONTIER, SAFETY_REVIEW => FRONTIER, DEEP_COUNCIL => FRONTIER,
)

model_id_for_tier(t::ModelTier) = TIER_TO_MODEL[t]
tier_for_model_id(id::AbstractString) = MODEL_TO_TIER[String(id)]
parameter_target(id::AbstractString) = PARAM_TARGETS[String(id)]
family_model_ids() = collect(keys(PARAM_TARGETS))
can_host(parent_tier::ModelTier, child_tier::ModelTier)::Bool = TIER_RANK[parent_tier] > TIER_RANK[child_tier]
can_host_models(parent_model_id::AbstractString, child_model_id::AbstractString)::Bool = parameter_target(parent_model_id) > parameter_target(child_model_id)

function resolve_child_model_id(parent_model_id::AbstractString, role::SubAgentRole)::String
    parent = String(parent_model_id)
    if haskey(ATOMIC_ROLE_MODEL, role)
        child = ATOMIC_ROLE_MODEL[role]
        can_host_models(parent, child) || error("parent=$parent cannot host child=$child")
        return child
    end
    parent_tier = tier_for_model_id(parent)
    preferred = get(ROLE_DEFAULT_TIER, role, SPECIALIST)
    ordered = sort(collect(keys(TIER_RANK)); by = tier -> TIER_RANK[tier])
    for tier in ordered
        child = model_id_for_tier(tier)
        if TIER_RANK[tier] >= TIER_RANK[preferred] && can_host_models(parent, child)
            return child
        end
    end
    role in builtin_lane(parent).roles && return parent
    error("no embeddable lane for role under parent=$parent")
end

function builtin_lane(model_id::AbstractString)::ModelLane
    lane = findfirst(item -> item.model_id == model_id, builtin_lanes())
    lane === nothing && error("unknown model lane: $model_id")
    return builtin_lanes()[lane]
end

function builtin_lanes()::Vector{ModelLane}
    return [
        ModelLane("Auro-156K", 156_000, ATOMIC, false, ModelTier[], [ROUTING_SEED, CLASSIFIER, JSON_REPAIR, TOOL_SELECTION, STYLE_GUARD]),
        ModelLane("Auro-250M", 250_000_000, ATOMIC, true, [ATOMIC], [INTENT_EXTRACT, RETRIEVAL_FILTER, STRUCTURED_TRANSFORM, CODE_TRIAGE, MEMORY_CONSOLIDATION, SEMANTIC_OUTLINE]),
        ModelLane("Auro-500M", 500_000_000, ATOMIC, true, [ATOMIC], [TOOL_EXECUTION_PLAN, CODE_PATCH, EVIDENCE_REVIEW, LOCAL_WORKER, EXPERT_CONSENSUS, TEXT_EXPANSION, CREATIVE_BRANCH]),
        ModelLane("Auro-2B", 2_000_000_000, EDGE, true, [ATOMIC], [ROUTER, TOOL_CALL, EMBED_FAST, SPECTRAL_TRIAGE]),
        ModelLane("Auro-4B", 4_000_000_000, SPECIALIST, true, [ATOMIC, EDGE], [CODE_EDIT, SPECTRAL_MATCH, JSON_STRUCT, TOOL_PLAN]),
        ModelLane("Auro-8B", 8_000_000_000, GENERAL, true, [ATOMIC, EDGE, SPECIALIST], [REASON, PLAN, CRITIQUE, SPECTRAL_EXPLAIN]),
        ModelLane("Auro-14B", 14_000_000_000, ORCHESTRATOR, true, [ATOMIC, EDGE, SPECIALIST, GENERAL], [ORCHESTRATOR_ROLE, COUNCIL_CHAIR, INSTRUCT_DEV, MULTI_AGENT_ROUTER]),
        ModelLane("Auro-100B", 100_000_000_000, FRONTIER, true, [ATOMIC, EDGE, SPECIALIST, GENERAL, ORCHESTRATOR], [FRONTIER_RESEARCH, LONG_HORIZON, SAFETY_REVIEW, DEEP_COUNCIL]),
    ]
end

end # module
