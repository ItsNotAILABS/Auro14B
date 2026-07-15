"""MESIE validation exceptions."""

from __future__ import annotations


class MESIEValidationError(Exception):
    """Raised when spectral record validation fails critically."""

    def __init__(self, message: str, errors: list = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class MESIEConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


class MESIEIOError(Exception):
    """Raised when input/output operations fail."""
    pass
