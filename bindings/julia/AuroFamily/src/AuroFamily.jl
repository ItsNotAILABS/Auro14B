"""
AuroFamily — Julia types mirroring auro_native_llm.types (Python) and AuroFamily.hs (Haskell).

Contract version: 1.0.0
Family: Auro — 2B / 4B / 8B / 14B / 100B multi-embedded sub-agents.
Scaffold only — no trained checkpoint claims.
"""
module AuroFamily

export CONTRACT_VERSION, FAMILY_ID
export ModelTier, SubAgentRole
export ArchitectureSpec, ModelLane, SubAgentSpec, SubAgentDispatch
export model_id_for_tier, tier_for_model_id, parameter_target
export can_host, resolve_child_model_id, family_model_ids

const CONTRACT_VERSION = "1.0.0"
const FAMILY_ID = "Auro"

@enum ModelTier begin
    EDGE          # Auro-2B
    SPECIALIST    # Auro-4B
    GENERAL       # Auro-8B
    ORCHESTRATOR  # Auro-14B
    FRONTIER      # Auro-100B
end

@enum SubAgentRole begin
    # Edge
    ROUTER
    TOOL_CALL
    EMBED_FAST
    SPECTRAL_TRIAGE
    # Specialist
    CODE_EDIT
    SPECTRAL_MATCH
    JSON_STRUCT
    TOOL_PLAN
    # General
    REASON
    PLAN
    CRITIQUE
    SPECTRAL_EXPLAIN
    # Orchestrator
    ORCHESTRATOR_ROLE
    COUNCIL_CHAIR
    INSTRUCT_DEV
    MULTI_AGENT_ROUTER
    # Frontier
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

const TIER_RANK = Dict(
    EDGE => 0,
    SPECIALIST => 1,
    GENERAL => 2,
    ORCHESTRATOR => 3,
    FRONTIER => 4,
)

const TIER_TO_MODEL = Dict(
    EDGE => "Auro-2B",
    SPECIALIST => "Auro-4B",
    GENERAL => "Auro-8B",
    ORCHESTRATOR => "Auro-14B",
    FRONTIER => "Auro-100B",
)

const MODEL_TO_TIER = Dict(v => k for (k, v) in TIER_TO_MODEL)

const PARAM_TARGETS = Dict(
    "Auro-2B" => 2_000_000_000,
    "Auro-4B" => 4_000_000_000,
    "Auro-8B" => 8_000_000_000,
    "Auro-14B" => 14_000_000_000,
    "Auro-100B" => 100_000_000_000,
)

const ROLE_DEFAULT_TIER = Dict(
    ROUTER => EDGE,
    TOOL_CALL => EDGE,
    EMBED_FAST => EDGE,
    SPECTRAL_TRIAGE => EDGE,
    CODE_EDIT => SPECIALIST,
    SPECTRAL_MATCH => SPECIALIST,
    JSON_STRUCT => SPECIALIST,
    TOOL_PLAN => SPECIALIST,
    REASON => GENERAL,
    PLAN => GENERAL,
    CRITIQUE => GENERAL,
    SPECTRAL_EXPLAIN => GENERAL,
    ORCHESTRATOR_ROLE => ORCHESTRATOR,
    COUNCIL_CHAIR => ORCHESTRATOR,
    INSTRUCT_DEV => ORCHESTRATOR,
    MULTI_AGENT_ROUTER => ORCHESTRATOR,
    FRONTIER_RESEARCH => FRONTIER,
    LONG_HORIZON => FRONTIER,
    SAFETY_REVIEW => FRONTIER,
    DEEP_COUNCIL => FRONTIER,
)

model_id_for_tier(t::ModelTier) = TIER_TO_MODEL[t]
tier_for_model_id(id::AbstractString) = MODEL_TO_TIER[String(id)]
parameter_target(id::AbstractString) = PARAM_TARGETS[String(id)]
family_model_ids() = collect(keys(PARAM_TARGETS))

function can_host(parent_tier::ModelTier, child_tier::ModelTier)::Bool
    return TIER_RANK[parent_tier] > TIER_RANK[child_tier]
end

function resolve_child_model_id(parent_model_id::AbstractString, role::SubAgentRole)::String
    parent_tier = tier_for_model_id(parent_model_id)
    preferred = get(ROLE_DEFAULT_TIER, role, SPECIALIST)
    ordered = sort(collect(keys(TIER_RANK)); by = t -> TIER_RANK[t])
    for t in ordered
        if TIER_RANK[t] >= TIER_RANK[preferred] && can_host(parent_tier, t)
            return model_id_for_tier(t)
        end
    end
    error("no embeddable lane for role under parent=$parent_model_id")
end

function builtin_lanes()::Vector{ModelLane}
    return [
        ModelLane("Auro-2B", PARAM_TARGETS["Auro-2B"], EDGE, false, ModelTier[],
            [ROUTER, TOOL_CALL, EMBED_FAST, SPECTRAL_TRIAGE]),
        ModelLane("Auro-4B", PARAM_TARGETS["Auro-4B"], SPECIALIST, true, [EDGE],
            [CODE_EDIT, SPECTRAL_MATCH, JSON_STRUCT, TOOL_PLAN]),
        ModelLane("Auro-8B", PARAM_TARGETS["Auro-8B"], GENERAL, true, [EDGE, SPECIALIST],
            [REASON, PLAN, CRITIQUE, SPECTRAL_EXPLAIN]),
        ModelLane("Auro-14B", PARAM_TARGETS["Auro-14B"], ORCHESTRATOR, true,
            [EDGE, SPECIALIST, GENERAL],
            [ORCHESTRATOR_ROLE, COUNCIL_CHAIR, INSTRUCT_DEV, MULTI_AGENT_ROUTER]),
        ModelLane("Auro-100B", PARAM_TARGETS["Auro-100B"], FRONTIER, true,
            [EDGE, SPECIALIST, GENERAL, ORCHESTRATOR],
            [FRONTIER_RESEARCH, LONG_HORIZON, SAFETY_REVIEW, DEEP_COUNCIL]),
    ]
end

end # module
