"""Cross-domain spectral transfer learning.

Provides domain adaptation techniques (CORAL, MMD, domain-invariant normalization)
for transferring spectral knowledge across heterogeneous domains.
"""

from mesie.transfer.alignment import (
    CORAL,
    MMD,
    DomainInvariantNormalizer,
)
from mesie.transfer.cross_domain import (
    CrossDomainTransferEngine,
    TransferResult,
)

__all__ = [
    "CORAL",
    "CrossDomainTransferEngine",
    "DomainInvariantNormalizer",
    "MMD",
    "TransferResult",
]
