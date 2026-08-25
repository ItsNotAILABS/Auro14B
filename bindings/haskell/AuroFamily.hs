{- |
Module      : AuroFamily
Description : AURO atomic-to-frontier family and embedded-council contract.
Copyright   : (c) Alfredo Medina / ItsNotAILABS, 2026
License     : Apache-2.0
Stability   : Experimental architecture contract; no trained checkpoint claim.

Contract version: 2.1.0
-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE StrictData #-}

module AuroFamily
  ( contractVersion
  , familyId
  , specialistTriad
  , ModelTier(..)
  , SubAgentRole(..)
  , tierRank
  , modelIdForTier
  , tierForModelId
  , parameterTarget
  , roleDefaultModel
  , ArchitectureSpec(..)
  , ModelLane(..)
  , SubAgentSpec(..)
  , SubAgentDispatch(..)
  , familyModelIds
  , canHost
  , resolveChildModelId
  , builtinLanes
  ) where

import GHC.Generics (Generic)

contractVersion :: String
contractVersion = "2.1.0"

familyId :: String
familyId = "Auro"

specialistTriad :: [String]
specialistTriad = ["Auro-500M-SENSUS", "Auro-500M-PRAXIS", "Auro-500M-VERBUM"]

data ModelTier = Atomic | Edge | Specialist | General | Orchestrator | Frontier
  deriving (Eq, Ord, Show, Read, Generic, Enum, Bounded)

tierRank :: ModelTier -> Int
tierRank Atomic       = 0
tierRank Edge         = 1
tierRank Specialist   = 2
tierRank General      = 3
tierRank Orchestrator = 4
tierRank Frontier     = 5

modelIdForTier :: ModelTier -> String
modelIdForTier Atomic       = "Auro-500M"
modelIdForTier Edge         = "Auro-2B"
modelIdForTier Specialist   = "Auro-4B"
modelIdForTier General      = "Auro-8B"
modelIdForTier Orchestrator = "Auro-14B"
modelIdForTier Frontier     = "Auro-100B"

tierForModelId :: String -> Maybe ModelTier
tierForModelId "Auro-156K" = Just Atomic
tierForModelId "Auro-250M" = Just Atomic
tierForModelId "Auro-500M" = Just Atomic
tierForModelId "Auro-2B"   = Just Edge
tierForModelId "Auro-4B"   = Just Specialist
tierForModelId "Auro-8B"   = Just General
tierForModelId "Auro-14B"  = Just Orchestrator
tierForModelId "Auro-100B" = Just Frontier
tierForModelId _            = Nothing

parameterTarget :: String -> Maybe Integer
parameterTarget "Auro-156K" = Just 156000
parameterTarget "Auro-250M" = Just 250000000
parameterTarget "Auro-500M" = Just 500000000
parameterTarget "Auro-2B"   = Just 2000000000
parameterTarget "Auro-4B"   = Just 4000000000
parameterTarget "Auro-8B"   = Just 8000000000
parameterTarget "Auro-14B"  = Just 14000000000
parameterTarget "Auro-100B" = Just 100000000000
parameterTarget _            = Nothing

familyModelIds :: [String]
familyModelIds = ["Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"]

data SubAgentRole
  = RoutingSeed | Classifier | JsonRepair | ToolSelection
  | IntentExtract | RetrievalFilter | StructuredTransform | CodeTriage | MemoryConsolidation | SemanticOutline
  | ToolExecutionPlan | CodePatch | EvidenceReview | LocalWorker | ExpertConsensus | TextExpansion
  | Router | ToolCall | EmbedFast | SpectralTriage
  | CodeEdit | SpectralMatch | JsonStruct | ToolPlan
  | Reason | Plan | Critique | SpectralExplain
  | OrchestratorRole | CouncilChair | InstructDev | MultiAgentRouter
  | FrontierResearch | LongHorizon | SafetyReview | DeepCouncil
  deriving (Eq, Ord, Show, Read, Generic, Enum, Bounded)

roleDefaultModel :: SubAgentRole -> String
roleDefaultModel RoutingSeed      = "Auro-156K"
roleDefaultModel Classifier       = "Auro-156K"
roleDefaultModel JsonRepair       = "Auro-156K"
roleDefaultModel ToolSelection    = "Auro-156K"
roleDefaultModel IntentExtract    = "Auro-250M"
roleDefaultModel RetrievalFilter  = "Auro-250M"
roleDefaultModel StructuredTransform = "Auro-250M"
roleDefaultModel CodeTriage       = "Auro-250M"
roleDefaultModel MemoryConsolidation = "Auro-250M"
roleDefaultModel SemanticOutline  = "Auro-250M"
roleDefaultModel ToolExecutionPlan = "Auro-500M"
roleDefaultModel CodePatch        = "Auro-500M"
roleDefaultModel EvidenceReview   = "Auro-500M"
roleDefaultModel LocalWorker      = "Auro-500M"
roleDefaultModel ExpertConsensus  = "Auro-500M"
roleDefaultModel TextExpansion    = "Auro-500M"
roleDefaultModel Router           = "Auro-2B"
roleDefaultModel ToolCall         = "Auro-2B"
roleDefaultModel EmbedFast        = "Auro-2B"
roleDefaultModel SpectralTriage   = "Auro-2B"
roleDefaultModel CodeEdit         = "Auro-4B"
roleDefaultModel SpectralMatch    = "Auro-4B"
roleDefaultModel JsonStruct       = "Auro-4B"
roleDefaultModel ToolPlan         = "Auro-4B"
roleDefaultModel Reason           = "Auro-8B"
roleDefaultModel Plan             = "Auro-8B"
roleDefaultModel Critique         = "Auro-8B"
roleDefaultModel SpectralExplain  = "Auro-8B"
roleDefaultModel OrchestratorRole = "Auro-14B"
roleDefaultModel CouncilChair     = "Auro-14B"
roleDefaultModel InstructDev      = "Auro-14B"
roleDefaultModel MultiAgentRouter = "Auro-14B"
roleDefaultModel FrontierResearch = "Auro-100B"
roleDefaultModel LongHorizon      = "Auro-100B"
roleDefaultModel SafetyReview     = "Auro-100B"
roleDefaultModel DeepCouncil      = "Auro-100B"

data ArchitectureSpec = ArchitectureSpec
  { archHiddenSize                :: !Int
  , archLayers                    :: !Int
  , archAttentionHeads            :: !Int
  , archKvHeads                   :: !Int
  , archIntermediateSize          :: !Int
  , archContextWindowTokensTarget :: !Int
  , archVocabSizeTarget           :: !Int
  } deriving (Eq, Show, Generic)

data ModelLane = ModelLane
  { laneModelId           :: !String
  , laneParameterTarget   :: !Integer
  , laneTier              :: !ModelTier
  , laneCanEmbedSubagents :: !Bool
  , laneEmbeddableTiers   :: ![ModelTier]
  , laneRoles             :: ![SubAgentRole]
  } deriving (Eq, Show, Generic)

data SubAgentSpec = SubAgentSpec
  { specAgentId       :: !String
  , specRole          :: !SubAgentRole
  , specChildModelId  :: !String
  , specParentModelId :: !String
  , specTaskId        :: !String
  , specIntent        :: !String
  , specEvidenceRefs  :: ![String]
  } deriving (Eq, Show, Generic)

data SubAgentDispatch = SubAgentDispatch
  { dispatchOk             :: !Bool
  , dispatchParentModelId  :: !String
  , dispatchChildModelId   :: !String
  , dispatchRole           :: !SubAgentRole
  , dispatchAgentId        :: !String
  , dispatchTaskId         :: !String
  , dispatchMessage        :: !String
  , dispatchError          :: !(Maybe String)
  } deriving (Eq, Show, Generic)

canHost :: ModelTier -> ModelTier -> Bool
canHost parent child = tierRank parent > tierRank child

resolveChildModelId :: String -> SubAgentRole -> Either String String
resolveChildModelId parentModelId role = do
  parentTier <- maybe (Left $ "unknown parent model_id: " ++ parentModelId) Right (tierForModelId parentModelId)
  let child = roleDefaultModel role
  childTier <- maybe (Left $ "unknown child model_id: " ++ child) Right (tierForModelId child)
  if canHost parentTier childTier
    then Right child
    else Left "incompatible parent and child lane"

builtinLanes :: [ModelLane]
builtinLanes =
  [ ModelLane "Auro-156K" 156000 Atomic False [] [RoutingSeed, Classifier, JsonRepair, ToolSelection]
  , ModelLane "Auro-250M" 250000000 Atomic False [] [IntentExtract, RetrievalFilter, StructuredTransform, CodeTriage, MemoryConsolidation, SemanticOutline]
  , ModelLane "Auro-500M" 500000000 Atomic False [] [ToolExecutionPlan, CodePatch, EvidenceReview, LocalWorker, ExpertConsensus, TextExpansion]
  , ModelLane "Auro-2B" 2000000000 Edge True [Atomic] [Router, ToolCall, EmbedFast, SpectralTriage]
  , ModelLane "Auro-4B" 4000000000 Specialist True [Atomic, Edge] [CodeEdit, SpectralMatch, JsonStruct, ToolPlan]
  , ModelLane "Auro-8B" 8000000000 General True [Atomic, Edge, Specialist] [Reason, Plan, Critique, SpectralExplain]
  , ModelLane "Auro-14B" 14000000000 Orchestrator True [Atomic, Edge, Specialist, General] [OrchestratorRole, CouncilChair, InstructDev, MultiAgentRouter]
  , ModelLane "Auro-100B" 100000000000 Frontier True [Atomic, Edge, Specialist, General, Orchestrator] [FrontierResearch, LongHorizon, SafetyReview, DeepCouncil]
  ]
