{- |
Module      : AuroFamily
Description : AURO family types from 156K through 100B.

Mirrors Python `auro_native_llm.types` and Julia `AuroFamily`.
Architecture/routing contracts do not claim trained checkpoint evidence.
Contract version: 2.0.0
-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE StrictData #-}

module AuroFamily
  ( contractVersion
  , familyId
  , ModelTier(..)
  , SubAgentRole(..)
  , tierRank
  , modelIdForTier
  , tierForModelId
  , parameterTarget
  , roleDefaultTier
  , rolePreferredModel
  , ArchitectureSpec(..)
  , ModelLane(..)
  , SubAgentSpec(..)
  , SubAgentDispatch(..)
  , familyModelIds
  , canHost
  , canHostModels
  , resolveChildModelId
  , builtinLanes
  ) where

import Data.List (find, sortOn)
import GHC.Generics (Generic)

contractVersion :: String
contractVersion = "2.0.0"

familyId :: String
familyId = "Auro"

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
tierForModelId "Auro-156K"        = Just Atomic
tierForModelId "Auro-250M"        = Just Atomic
tierForModelId "Auro-500M"        = Just Atomic
tierForModelId "Auro-500M-SENSUS" = Just Atomic
tierForModelId "Auro-500M-PRAXIS" = Just Atomic
tierForModelId "Auro-500M-VERBUM" = Just Atomic
tierForModelId "Auro-2B"          = Just Edge
tierForModelId "Auro-4B"          = Just Specialist
tierForModelId "Auro-8B"          = Just General
tierForModelId "Auro-14B"         = Just Orchestrator
tierForModelId "Auro-100B"        = Just Frontier
tierForModelId _                   = Nothing

parameterTarget :: String -> Maybe Integer
parameterTarget "Auro-156K"        = Just 156000
parameterTarget "Auro-250M"        = Just 250000000
parameterTarget "Auro-500M"        = Just 500000000
parameterTarget "Auro-500M-SENSUS" = Just 500000000
parameterTarget "Auro-500M-PRAXIS" = Just 500000000
parameterTarget "Auro-500M-VERBUM" = Just 500000000
parameterTarget "Auro-2B"          = Just 2000000000
parameterTarget "Auro-4B"          = Just 4000000000
parameterTarget "Auro-8B"          = Just 8000000000
parameterTarget "Auro-14B"         = Just 14000000000
parameterTarget "Auro-100B"        = Just 100000000000
parameterTarget _                   = Nothing

familyModelIds :: [String]
familyModelIds = ["Auro-156K", "Auro-250M", "Auro-500M", "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"]

data SubAgentRole
  = RoutingSeed | Classifier | JsonRepair | ToolSelection | StyleGuard
  | IntentExtract | RetrievalFilter | StructuredTransform | CodeTriage | MemoryConsolidation | SemanticOutline
  | ToolExecutionPlan | CodePatch | EvidenceReview | LocalWorker | ExpertConsensus | TextExpansion | CreativeBranch
  | Router | ToolCall | EmbedFast | SpectralTriage
  | CodeEdit | SpectralMatch | JsonStruct | ToolPlan
  | Reason | Plan | Critique | SpectralExplain
  | OrchestratorRole | CouncilChair | InstructDev | MultiAgentRouter
  | FrontierResearch | LongHorizon | SafetyReview | DeepCouncil
  deriving (Eq, Ord, Show, Read, Generic, Enum, Bounded)

roleDefaultTier :: SubAgentRole -> ModelTier
roleDefaultTier role
  | role `elem` atomicRoles = Atomic
roleDefaultTier Router           = Edge
roleDefaultTier ToolCall         = Edge
roleDefaultTier EmbedFast        = Edge
roleDefaultTier SpectralTriage   = Edge
roleDefaultTier CodeEdit         = Specialist
roleDefaultTier SpectralMatch    = Specialist
roleDefaultTier JsonStruct       = Specialist
roleDefaultTier ToolPlan         = Specialist
roleDefaultTier Reason           = General
roleDefaultTier Plan             = General
roleDefaultTier Critique         = General
roleDefaultTier SpectralExplain  = General
roleDefaultTier OrchestratorRole = Orchestrator
roleDefaultTier CouncilChair     = Orchestrator
roleDefaultTier InstructDev      = Orchestrator
roleDefaultTier MultiAgentRouter = Orchestrator
roleDefaultTier FrontierResearch = Frontier
roleDefaultTier LongHorizon      = Frontier
roleDefaultTier SafetyReview     = Frontier
roleDefaultTier DeepCouncil      = Frontier

atomicRoles :: [SubAgentRole]
atomicRoles =
  [ RoutingSeed, Classifier, JsonRepair, ToolSelection, StyleGuard
  , IntentExtract, RetrievalFilter, StructuredTransform, CodeTriage, MemoryConsolidation, SemanticOutline
  , ToolExecutionPlan, CodePatch, EvidenceReview, LocalWorker, ExpertConsensus, TextExpansion, CreativeBranch
  ]

rolePreferredModel :: SubAgentRole -> Maybe String
rolePreferredModel role
  | role `elem` [RoutingSeed, Classifier, JsonRepair, ToolSelection, StyleGuard] = Just "Auro-156K"
  | role `elem` [IntentExtract, RetrievalFilter, StructuredTransform, CodeTriage, MemoryConsolidation, SemanticOutline] = Just "Auro-250M"
  | role `elem` [ToolExecutionPlan, CodePatch, EvidenceReview, LocalWorker, ExpertConsensus, TextExpansion, CreativeBranch] = Just "Auro-500M"
  | otherwise = Nothing

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

canHostModels :: String -> String -> Bool
canHostModels parent child =
  case (parameterTarget parent, parameterTarget child) of
    (Just p, Just c) -> p > c
    _                -> False

resolveChildModelId :: String -> SubAgentRole -> Either String String
resolveChildModelId parentModelId role =
  case rolePreferredModel role of
    Just child -> if canHostModels parentModelId child
                    then Right child
                    else Left $ "parent=" ++ parentModelId ++ " cannot host child=" ++ child
    Nothing ->
      case tierForModelId parentModelId of
        Nothing -> Left $ "unknown parent model_id: " ++ parentModelId
        Just parentTier ->
          let preferred = roleDefaultTier role
              ordered = sortOn tierRank [minBound .. maxBound :: ModelTier]
              candidates = [modelIdForTier t | t <- ordered, tierRank t >= tierRank preferred, canHostModels parentModelId (modelIdForTier t)]
          in case candidates of
               (child:_) -> Right child
               [] -> case find ((== parentModelId) . laneModelId) builtinLanes of
                       Just lane | role `elem` laneRoles lane -> Right parentModelId
                       _ -> Left $ "no embeddable lane for role under parent=" ++ parentModelId

builtinLanes :: [ModelLane]
builtinLanes =
  [ ModelLane "Auro-156K" 156000 Atomic False [] [RoutingSeed, Classifier, JsonRepair, ToolSelection, StyleGuard]
  , ModelLane "Auro-250M" 250000000 Atomic True [Atomic] [IntentExtract, RetrievalFilter, StructuredTransform, CodeTriage, MemoryConsolidation, SemanticOutline]
  , ModelLane "Auro-500M" 500000000 Atomic True [Atomic] [ToolExecutionPlan, CodePatch, EvidenceReview, LocalWorker, ExpertConsensus, TextExpansion, CreativeBranch]
  , ModelLane "Auro-2B" 2000000000 Edge True [Atomic] [Router, ToolCall, EmbedFast, SpectralTriage]
  , ModelLane "Auro-4B" 4000000000 Specialist True [Atomic, Edge] [CodeEdit, SpectralMatch, JsonStruct, ToolPlan]
  , ModelLane "Auro-8B" 8000000000 General True [Atomic, Edge, Specialist] [Reason, Plan, Critique, SpectralExplain]
  , ModelLane "Auro-14B" 14000000000 Orchestrator True [Atomic, Edge, Specialist, General] [OrchestratorRole, CouncilChair, InstructDev, MultiAgentRouter]
  , ModelLane "Auro-100B" 100000000000 Frontier True [Atomic, Edge, Specialist, General, Orchestrator] [FrontierResearch, LongHorizon, SafetyReview, DeepCouncil]
  ]
