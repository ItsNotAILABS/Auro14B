{- |
Module      : AuroFamily
Description : Auro native LLM family types — 2B / 4B / 8B / 14B / 100B
              multi-embedded sub-agents. Mirrors Python auro_native_llm.types
              and Julia AuroFamily.
Copyright   : (c) Alfredo Medina / ItsNotAILABS, 2026
License     : Apache-2.0
Stability   : Experimental (scaffold — no trained checkpoint claims)

Contract version: 1.0.0
-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE StrictData #-}

module AuroFamily
  ( -- * Contract
    contractVersion
  , familyId
    -- * Tiers & roles
  , ModelTier(..)
  , SubAgentRole(..)
  , tierRank
  , modelIdForTier
  , tierForModelId
  , parameterTarget
  , roleDefaultTier
    -- * Architecture & lanes
  , ArchitectureSpec(..)
  , ModelLane(..)
  , SubAgentSpec(..)
  , SubAgentDispatch(..)
  , familyModelIds
  , canHost
  , resolveChildModelId
  , builtinLanes
  ) where

import Data.List (sortOn)
import GHC.Generics (Generic)

-- ---------------------------------------------------------------------------
-- Contract constants
-- ---------------------------------------------------------------------------

contractVersion :: String
contractVersion = "1.0.0"

familyId :: String
familyId = "Auro"

-- ---------------------------------------------------------------------------
-- Tiers
-- ---------------------------------------------------------------------------

data ModelTier
  = Edge          -- ^ Auro-2B
  | Specialist    -- ^ Auro-4B
  | General       -- ^ Auro-8B
  | Orchestrator  -- ^ Auro-14B
  | Frontier      -- ^ Auro-100B
  deriving (Eq, Ord, Show, Read, Generic, Enum, Bounded)

tierRank :: ModelTier -> Int
tierRank Edge         = 0
tierRank Specialist   = 1
tierRank General      = 2
tierRank Orchestrator = 3
tierRank Frontier     = 4

modelIdForTier :: ModelTier -> String
modelIdForTier Edge         = "Auro-2B"
modelIdForTier Specialist   = "Auro-4B"
modelIdForTier General      = "Auro-8B"
modelIdForTier Orchestrator = "Auro-14B"
modelIdForTier Frontier     = "Auro-100B"

tierForModelId :: String -> Maybe ModelTier
tierForModelId "Auro-2B"   = Just Edge
tierForModelId "Auro-4B"   = Just Specialist
tierForModelId "Auro-8B"   = Just General
tierForModelId "Auro-14B"  = Just Orchestrator
tierForModelId "Auro-100B" = Just Frontier
tierForModelId _           = Nothing

parameterTarget :: String -> Maybe Integer
parameterTarget "Auro-2B"   = Just 2000000000
parameterTarget "Auro-4B"   = Just 4000000000
parameterTarget "Auro-8B"   = Just 8000000000
parameterTarget "Auro-14B"  = Just 14000000000
parameterTarget "Auro-100B" = Just 100000000000
parameterTarget _           = Nothing

familyModelIds :: [String]
familyModelIds = ["Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B"]

-- ---------------------------------------------------------------------------
-- Roles
-- ---------------------------------------------------------------------------

data SubAgentRole
  = -- Edge
    Router | ToolCall | EmbedFast | SpectralTriage
  | -- Specialist
    CodeEdit | SpectralMatch | JsonStruct | ToolPlan
  | -- General
    Reason | Plan | Critique | SpectralExplain
  | -- Orchestrator
    OrchestratorRole | CouncilChair | InstructDev | MultiAgentRouter
  | -- Frontier
    FrontierResearch | LongHorizon | SafetyReview | DeepCouncil
  deriving (Eq, Ord, Show, Read, Generic, Enum, Bounded)

roleDefaultTier :: SubAgentRole -> ModelTier
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

-- ---------------------------------------------------------------------------
-- Specs (unique record field names across the module)
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- Multi-embedded hosting rules
-- ---------------------------------------------------------------------------

canHost :: ModelTier -> ModelTier -> Bool
canHost parent child = tierRank parent > tierRank child

resolveChildModelId :: String -> SubAgentRole -> Either String String
resolveChildModelId parentModelId role =
  case tierForModelId parentModelId of
    Nothing -> Left $ "unknown parent model_id: " ++ parentModelId
    Just parentTier ->
      let preferred = roleDefaultTier role
          ordered   = sortOn tierRank [minBound .. maxBound :: ModelTier]
          candidates =
            [ t
            | t <- ordered
            , tierRank t >= tierRank preferred
            , canHost parentTier t
            ]
      in case candidates of
           (t:_) -> Right (modelIdForTier t)
           []    ->
             if roleDefaultTier role == parentTier
               then Right parentModelId
               else Left $ "no embeddable lane for role under parent=" ++ parentModelId

-- ---------------------------------------------------------------------------
-- Builtin family table
-- ---------------------------------------------------------------------------

builtinLanes :: [ModelLane]
builtinLanes =
  [ ModelLane "Auro-2B"   2000000000   Edge         False []
      [Router, ToolCall, EmbedFast, SpectralTriage]
  , ModelLane "Auro-4B"   4000000000   Specialist   True  [Edge]
      [CodeEdit, SpectralMatch, JsonStruct, ToolPlan]
  , ModelLane "Auro-8B"   8000000000   General      True  [Edge, Specialist]
      [Reason, Plan, Critique, SpectralExplain]
  , ModelLane "Auro-14B"  14000000000  Orchestrator True  [Edge, Specialist, General]
      [OrchestratorRole, CouncilChair, InstructDev, MultiAgentRouter]
  , ModelLane "Auro-100B" 100000000000 Frontier     True  [Edge, Specialist, General, Orchestrator]
      [FrontierResearch, LongHorizon, SafetyReview, DeepCouncil]
  ]
