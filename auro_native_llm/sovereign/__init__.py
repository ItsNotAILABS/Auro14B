"""Versioned bridge to the FreddyCreates/sovereign training contract."""

from .contract import SovereignBinding, bind_sovereign, discover_sovereign_root

__all__ = ["SovereignBinding", "bind_sovereign", "discover_sovereign_root"]
