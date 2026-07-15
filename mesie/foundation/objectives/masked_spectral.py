"""Masked spectral modeling objectives.

Implements BERT-style masked prediction adapted for spectral data,
including band masking, structured masking, and hierarchical masking
strategies for self-supervised spectral pretraining.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MaskedSpectralModeling:
    """Core masked spectral modeling objective.

    Masks portions of spectral input and trains the model to
    reconstruct the masked regions from context. This is the
    spectral equivalent of masked language modeling.

    Supports:
    - Random token masking (BERT-style)
    - Frequency band masking
    - Temporal span masking
    - Structured/causal masking
    - Progressive masking ratio

    Attributes:
        mask_ratio: Fraction of tokens to mask.
        mask_token_value: Value used for masked positions.
        strategy: Masking strategy.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        mask_token_value: float = 0.0,
        strategy: str = "random",
        replace_with_mask: float = 0.8,
        replace_with_random: float = 0.1,
        keep_original: float = 0.1,
        min_mask_span: int = 1,
        max_mask_span: int = 10,
        predict_original: bool = True,
    ):
        """Initialize masked spectral modeling.

        Args:
            mask_ratio: Fraction of positions to mask.
            mask_token_value: Value for masked positions.
            strategy: Masking strategy ('random', 'span', 'block').
            replace_with_mask: Probability of replacing with mask token.
            replace_with_random: Probability of random replacement.
            keep_original: Probability of keeping original.
            min_mask_span: Minimum span length for span masking.
            max_mask_span: Maximum span length for span masking.
            predict_original: Whether targets are original values.
        """
        self.mask_ratio = mask_ratio
        self.mask_token_value = mask_token_value
        self.strategy = strategy
        self.replace_with_mask = replace_with_mask
        self.replace_with_random = replace_with_random
        self.keep_original = keep_original
        self.min_mask_span = min_mask_span
        self.max_mask_span = max_mask_span
        self.predict_original = predict_original

        assert abs(replace_with_mask + replace_with_random + keep_original - 1.0) < 1e-6

    def create_mask(
        self,
        shape: Tuple[int, ...],
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Create masking pattern.

        Args:
            shape: Input shape (batch_size, seq_len) or (batch_size, seq_len, dim).
            rng: Random state.

        Returns:
            Boolean mask (True = masked position).
        """
        if rng is None:
            rng = np.random.RandomState()

        batch_size = shape[0]
        seq_len = shape[1]

        if self.strategy == "random":
            mask = self._random_mask(batch_size, seq_len, rng)
        elif self.strategy == "span":
            mask = self._span_mask(batch_size, seq_len, rng)
        elif self.strategy == "block":
            mask = self._block_mask(batch_size, seq_len, rng)
        else:
            mask = self._random_mask(batch_size, seq_len, rng)

        return mask

    def _random_mask(
        self, batch_size: int, seq_len: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Create random token mask."""
        mask = rng.random((batch_size, seq_len)) < self.mask_ratio
        return mask

    def _span_mask(
        self, batch_size: int, seq_len: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Create span masking (contiguous regions)."""
        mask = np.zeros((batch_size, seq_len), dtype=bool)

        for b in range(batch_size):
            num_masked = 0
            target_masked = int(seq_len * self.mask_ratio)

            while num_masked < target_masked:
                span_len = rng.randint(self.min_mask_span, self.max_mask_span + 1)
                start = rng.randint(0, max(1, seq_len - span_len))
                end = min(start + span_len, seq_len)
                mask[b, start:end] = True
                num_masked = np.sum(mask[b])

        return mask

    def _block_mask(
        self, batch_size: int, seq_len: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Create block masking (2D blocks for spectrogram-like inputs)."""
        mask = np.zeros((batch_size, seq_len), dtype=bool)

        num_blocks = max(1, int(seq_len * self.mask_ratio / self.max_mask_span))

        for b in range(batch_size):
            for _ in range(num_blocks):
                block_size = rng.randint(self.min_mask_span, self.max_mask_span + 1)
                start = rng.randint(0, max(1, seq_len - block_size))
                mask[b, start:start + block_size] = True

        return mask

    def apply_mask(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mask to input with replacement strategies.

        Args:
            x: Input tensor [B, T, D] or [B, T].
            mask: Boolean mask [B, T].
            rng: Random state.

        Returns:
            Tuple of (masked_input, targets).
        """
        if rng is None:
            rng = np.random.RandomState()

        masked_x = x.copy()
        targets = x.copy()

        # For each masked position, decide replacement strategy
        for b in range(mask.shape[0]):
            for t in range(mask.shape[1]):
                if not mask[b, t]:
                    continue

                rand_val = rng.random()
                if rand_val < self.replace_with_mask:
                    # Replace with mask token
                    if x.ndim == 3:
                        masked_x[b, t, :] = self.mask_token_value
                    else:
                        masked_x[b, t] = self.mask_token_value
                elif rand_val < self.replace_with_mask + self.replace_with_random:
                    # Replace with random value
                    if x.ndim == 3:
                        masked_x[b, t, :] = rng.randn(x.shape[2]) * 0.1
                    else:
                        masked_x[b, t] = rng.randn() * 0.1
                # else: keep original

        return masked_x, targets

    def compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute masked prediction loss.

        Only computes loss at masked positions.

        Args:
            predictions: Model predictions [B, T, D].
            targets: Original values [B, T, D].
            mask: Boolean mask [B, T].

        Returns:
            Loss and metrics.
        """
        # Expand mask to match predictions shape
        if predictions.ndim == 3 and mask.ndim == 2:
            mask_expanded = mask[:, :, np.newaxis]
        else:
            mask_expanded = mask

        # Compute error only at masked positions
        error = (predictions - targets) ** 2
        masked_error = error * mask_expanded

        num_masked = np.sum(mask) + 1e-10

        # MSE at masked positions
        mse = float(np.sum(masked_error) / (num_masked * predictions.shape[-1]
                                             if predictions.ndim == 3 else num_masked))

        # Also compute accuracy-like metrics
        # (how well we reconstruct relative to random)
        signal_power = float(np.mean(targets ** 2) + 1e-10)
        snr = 10 * np.log10(signal_power / (mse + 1e-10))

        metrics = {
            "masked_mse": mse,
            "reconstruction_snr_db": float(snr),
            "num_masked_tokens": int(np.sum(mask)),
            "mask_ratio_actual": float(np.mean(mask)),
        }

        return mse, metrics


class BandMasking:
    """Frequency band masking strategy.

    Masks entire frequency bands, forcing the model to
    predict missing spectral content from surrounding bands.
    This teaches frequency-domain relationships.

    Attributes:
        num_bands: Number of frequency bands.
        mask_bands_ratio: Fraction of bands to mask.
    """

    def __init__(
        self,
        num_bands: int = 32,
        mask_bands_ratio: float = 0.25,
        sample_rate: float = 1.0,
        band_scale: str = "mel",
        min_masked_bands: int = 1,
        max_masked_bands: int = 8,
    ):
        """Initialize band masking.

        Args:
            num_bands: Total number of bands.
            mask_bands_ratio: Fraction to mask.
            sample_rate: Sampling rate.
            band_scale: Frequency scale.
            min_masked_bands: Minimum bands to mask.
            max_masked_bands: Maximum bands to mask.
        """
        self.num_bands = num_bands
        self.mask_bands_ratio = mask_bands_ratio
        self.sample_rate = sample_rate
        self.band_scale = band_scale
        self.min_masked_bands = min_masked_bands
        self.max_masked_bands = max_masked_bands

        self.band_edges = self._compute_band_edges()

    def _compute_band_edges(self) -> np.ndarray:
        """Compute band edge frequencies."""
        nyquist = self.sample_rate / 2

        if self.band_scale == "mel":
            mel_low = 0
            mel_high = 2595 * np.log10(1 + nyquist / 700)
            mel_edges = np.linspace(mel_low, mel_high, self.num_bands + 1)
            return 700 * (10 ** (mel_edges / 2595) - 1)
        elif self.band_scale == "log":
            return np.logspace(
                np.log10(max(1.0, nyquist / 1000)),
                np.log10(nyquist),
                self.num_bands + 1
            )
        else:
            return np.linspace(0, nyquist, self.num_bands + 1)

    def create_band_mask(
        self,
        n_freqs: int,
        batch_size: int = 1,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Create frequency band mask.

        Args:
            n_freqs: Number of frequency bins.
            batch_size: Batch size.
            rng: Random state.

        Returns:
            Frequency mask [B, n_freqs] (True = masked).
        """
        if rng is None:
            rng = np.random.RandomState()

        mask = np.zeros((batch_size, n_freqs), dtype=bool)
        freqs = np.linspace(0, self.sample_rate / 2, n_freqs)

        for b in range(batch_size):
            # Select bands to mask
            num_to_mask = rng.randint(
                self.min_masked_bands,
                self.max_masked_bands + 1
            )
            num_to_mask = min(num_to_mask, self.num_bands)
            masked_bands = rng.choice(self.num_bands, num_to_mask, replace=False)

            for band_idx in masked_bands:
                low = self.band_edges[band_idx]
                high = self.band_edges[band_idx + 1]
                band_mask = (freqs >= low) & (freqs < high)
                mask[b] |= band_mask

        return mask

    def apply_to_spectrum(
        self,
        spectrum: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply band mask to spectrum.

        Args:
            spectrum: Input spectrum [B, T, F] or [B, F].
            mask: Frequency mask [B, F].

        Returns:
            Masked spectrum and targets.
        """
        targets = spectrum.copy()
        masked_spectrum = spectrum.copy()

        if spectrum.ndim == 3:
            # [B, T, F] - mask frequency dimension
            mask_expanded = mask[:, np.newaxis, :]
            masked_spectrum = masked_spectrum * (~mask_expanded)
        else:
            masked_spectrum = masked_spectrum * (~mask)

        return masked_spectrum, targets


class StructuredMasking:
    """Structured masking for time-frequency representations.

    Creates structured masking patterns that respect the
    time-frequency structure of spectrograms:
    - Horizontal stripes (frequency masking)
    - Vertical stripes (time masking)
    - Rectangular blocks
    - Diagonal patterns (for time-frequency coupling)

    Attributes:
        pattern_type: Type of structured pattern.
        mask_ratio: Target mask ratio.
    """

    def __init__(
        self,
        pattern_type: str = "mixed",
        mask_ratio: float = 0.3,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        max_time_width: int = 20,
        max_freq_width: int = 16,
    ):
        """Initialize structured masking.

        Args:
            pattern_type: Pattern type ('time', 'freq', 'block', 'mixed').
            mask_ratio: Target masking ratio.
            num_time_masks: Number of time masks.
            num_freq_masks: Number of frequency masks.
            max_time_width: Max time mask width.
            max_freq_width: Max frequency mask width.
        """
        self.pattern_type = pattern_type
        self.mask_ratio = mask_ratio
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.max_time_width = max_time_width
        self.max_freq_width = max_freq_width

    def create_mask(
        self,
        time_steps: int,
        freq_bins: int,
        batch_size: int = 1,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Create structured 2D mask.

        Args:
            time_steps: Number of time steps.
            freq_bins: Number of frequency bins.
            batch_size: Batch size.
            rng: Random state.

        Returns:
            2D mask [B, T, F] (True = masked).
        """
        if rng is None:
            rng = np.random.RandomState()

        mask = np.zeros((batch_size, time_steps, freq_bins), dtype=bool)

        for b in range(batch_size):
            if self.pattern_type == "time":
                mask[b] = self._time_mask(time_steps, freq_bins, rng)
            elif self.pattern_type == "freq":
                mask[b] = self._freq_mask(time_steps, freq_bins, rng)
            elif self.pattern_type == "block":
                mask[b] = self._block_mask(time_steps, freq_bins, rng)
            elif self.pattern_type == "mixed":
                # Randomly choose pattern type
                pattern = rng.choice(["time", "freq", "block"])
                if pattern == "time":
                    mask[b] = self._time_mask(time_steps, freq_bins, rng)
                elif pattern == "freq":
                    mask[b] = self._freq_mask(time_steps, freq_bins, rng)
                else:
                    mask[b] = self._block_mask(time_steps, freq_bins, rng)

        return mask

    def _time_mask(
        self, T: int, F: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Create time-masking pattern (vertical stripes)."""
        mask = np.zeros((T, F), dtype=bool)
        for _ in range(self.num_time_masks):
            width = rng.randint(1, min(self.max_time_width, T) + 1)
            start = rng.randint(0, max(1, T - width))
            mask[start:start + width, :] = True
        return mask

    def _freq_mask(
        self, T: int, F: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Create frequency masking (horizontal stripes)."""
        mask = np.zeros((T, F), dtype=bool)
        for _ in range(self.num_freq_masks):
            width = rng.randint(1, min(self.max_freq_width, F) + 1)
            start = rng.randint(0, max(1, F - width))
            mask[:, start:start + width] = True
        return mask

    def _block_mask(
        self, T: int, F: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Create block masking (rectangles)."""
        mask = np.zeros((T, F), dtype=bool)
        target_area = int(T * F * self.mask_ratio)
        masked_area = 0

        while masked_area < target_area:
            t_width = rng.randint(1, min(self.max_time_width, T) + 1)
            f_width = rng.randint(1, min(self.max_freq_width, F) + 1)
            t_start = rng.randint(0, max(1, T - t_width))
            f_start = rng.randint(0, max(1, F - f_width))
            mask[t_start:t_start + t_width, f_start:f_start + f_width] = True
            masked_area = np.sum(mask)

        return mask


class HierarchicalMasking:
    """Hierarchical masking at multiple granularities.

    Masks at different levels of the spectral hierarchy:
    - Fine: individual frequency bins
    - Medium: frequency sub-bands
    - Coarse: octave bands
    - Global: entire spectral regions

    Forces the model to learn representations at multiple scales.

    Attributes:
        levels: Number of hierarchy levels.
        level_ratios: Mask ratio per level.
    """

    def __init__(
        self,
        levels: int = 4,
        level_ratios: Optional[List[float]] = None,
        level_names: Optional[List[str]] = None,
    ):
        """Initialize hierarchical masking.

        Args:
            levels: Number of hierarchy levels.
            level_ratios: Mask ratio per level.
            level_names: Names for each level.
        """
        self.levels = levels
        self.level_ratios = level_ratios or [0.3, 0.2, 0.1, 0.05]
        self.level_names = level_names or ["fine", "medium", "coarse", "global"]

    def create_hierarchical_mask(
        self,
        seq_len: int,
        num_freqs: int,
        batch_size: int = 1,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """Create masks at all hierarchy levels.

        Args:
            seq_len: Sequence length.
            num_freqs: Number of frequency bins.
            batch_size: Batch size.
            rng: Random state.

        Returns:
            Dictionary of masks per level.
        """
        if rng is None:
            rng = np.random.RandomState()

        masks = {}
        granularities = [1, 4, 16, num_freqs // 4]

        for level in range(self.levels):
            gran = granularities[min(level, len(granularities) - 1)]
            ratio = self.level_ratios[min(level, len(self.level_ratios) - 1)]
            name = self.level_names[min(level, len(self.level_names) - 1)]

            mask = np.zeros((batch_size, seq_len, num_freqs), dtype=bool)

            for b in range(batch_size):
                # Mask at this granularity
                n_groups = max(1, num_freqs // gran)
                n_to_mask = max(1, int(n_groups * ratio))
                masked_groups = rng.choice(n_groups, n_to_mask, replace=False)

                for g in masked_groups:
                    start = g * gran
                    end = min(start + gran, num_freqs)
                    # Also select time range
                    t_start = 0 if level >= 2 else rng.randint(0, max(1, seq_len // 2))
                    t_end = seq_len if level >= 2 else rng.randint(seq_len // 2, seq_len)
                    mask[b, t_start:t_end, start:end] = True

            masks[name] = mask

        return masks

    def compute_hierarchical_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        masks: Dict[str, np.ndarray],
        level_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute loss at each hierarchy level.

        Args:
            predictions: Predicted values [B, T, F].
            targets: Target values [B, T, F].
            masks: Per-level masks.
            level_weights: Per-level loss weights.

        Returns:
            Total loss and per-level losses.
        """
        if level_weights is None:
            level_weights = {name: 1.0 / self.levels for name in self.level_names}

        total_loss = 0.0
        level_losses: Dict[str, float] = {}

        for name, mask in masks.items():
            error = (predictions - targets) ** 2
            masked_error = error * mask
            num_masked = np.sum(mask) + 1e-10

            level_loss = float(np.sum(masked_error) / num_masked)
            level_losses[name] = level_loss

            weight = level_weights.get(name, 1.0 / self.levels)
            total_loss += weight * level_loss

        level_losses["total"] = total_loss
        return total_loss, level_losses
