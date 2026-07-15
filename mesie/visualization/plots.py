"""Plotting functions for spectral records."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.io.loaders import RecordInput, load_record


def plot_spectrum(
    record: RecordInput,
    title: Optional[str] = None,
    log_scale: bool = True,
    show: bool = True,
) -> None:
    """Plot spectral components of a record.

    Requires matplotlib to be installed.

    Args:
        record: Input spectral record.
        title: Plot title.
        log_scale: Whether to use log scale on x-axis.
        show: Whether to call plt.show().

    Raises:
        ImportError: If matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    rec = load_record(record)

    fig, ax = plt.subplots(figsize=(10, 6))
    for comp in rec.components:
        ax.plot(comp.frequency, comp.amplitude, label=comp.name)

    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Spectral Record: {rec.record_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()


def plot_comparison(
    reference: RecordInput,
    candidate: RecordInput,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot comparison between reference and candidate records.

    Args:
        reference: Reference record.
        candidate: Candidate record.
        title: Plot title.
        show: Whether to call plt.show().

    Raises:
        ImportError: If matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    ref = load_record(reference)
    cand = load_record(candidate)

    fig, ax = plt.subplots(figsize=(10, 6))
    for comp in ref.components:
        ax.plot(comp.frequency, comp.amplitude, label=f"Ref: {comp.name}", linestyle="-")
    for comp in cand.components:
        ax.plot(comp.frequency, comp.amplitude, label=f"Cand: {comp.name}", linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Spectral Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()
