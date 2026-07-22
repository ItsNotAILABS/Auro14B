"""Constitutional substrate for AURO checkpoint identity and promotion governance."""
from .checkpoint_constitution import (
    AIOPS_DOMAINS,
    PROTOCOLS,
    ConstitutionalCheckpoint,
    ConstitutionalGateError,
    ProtocolAssertion,
    build_constitutional_checkpoint,
    load_and_verify_constitutional_manifest,
    require_promotable,
    validate_inventory,
    write_constitutional_manifest,
)

__all__ = [
    "AIOPS_DOMAINS",
    "PROTOCOLS",
    "ConstitutionalCheckpoint",
    "ConstitutionalGateError",
    "ProtocolAssertion",
    "build_constitutional_checkpoint",
    "load_and_verify_constitutional_manifest",
    "require_promotable",
    "validate_inventory",
    "write_constitutional_manifest",
]
