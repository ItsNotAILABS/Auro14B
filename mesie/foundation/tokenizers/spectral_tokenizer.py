"""Spectral tokenizer implementations.

This module provides the complete tokenizer pipeline for converting
raw spectral data from various modalities into token sequences
suitable for transformer processing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesie.foundation.tokenizers.codebook import (
    SpectralCodebook,
    ResidualQuantizer,
    ProductQuantizer,
)
from mesie.foundation.tokenizers.augmentation import SpectralAugmentation


class VQVAETokenizer:
    """VQ-VAE based spectral tokenizer.

    Uses a Vector Quantized Variational Autoencoder to convert
    continuous spectral patches into discrete token sequences.
    Supports residual quantization for higher fidelity.

    Attributes:
        input_dim: Input spectral dimension.
        codebook_size: Number of discrete codes.
        codebook_dim: Code vector dimension.
        num_codebooks: Number of residual codebook stages.
        encoder_dims: Encoder layer dimensions.
        decoder_dims: Decoder layer dimensions.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        codebook_size: int = 8192,
        codebook_dim: int = 256,
        num_codebooks: int = 4,
        encoder_dims: Optional[List[int]] = None,
        decoder_dims: Optional[List[int]] = None,
        commitment_weight: float = 0.25,
        use_residual: bool = True,
    ):
        """Initialize VQ-VAE tokenizer.

        Args:
            input_dim: Input spectral dimension.
            codebook_size: Number of codes.
            codebook_dim: Code dimension.
            num_codebooks: Number of residual stages.
            encoder_dims: Encoder hidden dimensions.
            decoder_dims: Decoder hidden dimensions.
            commitment_weight: Commitment loss weight.
            use_residual: Whether to use residual quantization.
        """
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.commitment_weight = commitment_weight
        self.use_residual = use_residual

        encoder_dims = encoder_dims or [512, 384, codebook_dim]
        decoder_dims = decoder_dims or [codebook_dim, 384, 512, input_dim]

        # Encoder layers
        self.encoder_layers = []
        dims = [input_dim] + encoder_dims
        for i in range(len(dims) - 1):
            self.encoder_layers.append({
                "weight": np.random.randn(dims[i], dims[i + 1]) * 0.02,
                "bias": np.zeros(dims[i + 1]),
            })

        # Decoder layers
        self.decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            self.decoder_layers.append({
                "weight": np.random.randn(decoder_dims[i], decoder_dims[i + 1]) * 0.02,
                "bias": np.zeros(decoder_dims[i + 1]),
            })

        # Quantizer
        if use_residual:
            self.quantizer = ResidualQuantizer(
                num_codebooks=num_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
        else:
            self.quantizer = SpectralCodebook(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        ))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode spectral input to latent representation.

        Args:
            x: Input [..., input_dim].

        Returns:
            Latent representation [..., codebook_dim].
        """
        hidden = x
        for i, layer in enumerate(self.encoder_layers):
            hidden = np.einsum("...d,do->...o", hidden, layer["weight"]) + layer["bias"]
            if i < len(self.encoder_layers) - 1:
                hidden = self._gelu(hidden)
        return hidden

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation to spectral output.

        Args:
            z: Latent representation [..., codebook_dim].

        Returns:
            Reconstructed output [..., input_dim].
        """
        hidden = z
        for i, layer in enumerate(self.decoder_layers):
            hidden = np.einsum("...d,do->...o", hidden, layer["weight"]) + layer["bias"]
            if i < len(self.decoder_layers) - 1:
                hidden = self._gelu(hidden)
        return hidden

    def tokenize(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert spectral input to discrete tokens.

        Args:
            x: Input [..., input_dim].

        Returns:
            Tuple of (token_ids, info).
        """
        # Encode
        latent = self.encode(x)

        # Quantize
        if self.use_residual:
            indices, quantized, info = self.quantizer.encode(latent)
        else:
            indices, quantized, info = self.quantizer.encode(latent)

        # Compute commitment loss
        commitment_loss = float(np.mean((latent - quantized) ** 2))
        info["commitment_loss"] = commitment_loss * self.commitment_weight

        return indices, info

    def detokenize(self, token_ids: np.ndarray) -> np.ndarray:
        """Convert discrete tokens back to spectral representation.

        Args:
            token_ids: Token indices.

        Returns:
            Reconstructed spectral output [..., input_dim].
        """
        if self.use_residual:
            quantized = self.quantizer.decode(token_ids)
        else:
            quantized = self.quantizer.decode(token_ids)

        return self.decode(quantized)


class PatchTokenizer:
    """Patch-based spectral tokenizer.

    Splits spectral data into patches (similar to ViT) and projects
    each patch into a token embedding. Supports both 1D (frequency-only)
    and 2D (time-frequency) patches.

    Attributes:
        patch_size: Patch size in frequency bins.
        patch_stride: Stride between patches.
        input_dim: Total input dimension.
        output_dim: Output token dimension.
        use_cls_token: Whether to prepend CLS token.
        num_register_tokens: Number of register tokens.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
        patch_size: int = 16,
        patch_stride: int = 16,
        num_channels: int = 1,
        use_cls_token: bool = True,
        num_register_tokens: int = 4,
        use_position_embedding: bool = True,
    ):
        """Initialize patch tokenizer.

        Args:
            input_dim: Input spectral dimension.
            output_dim: Output token dimension.
            patch_size: Size of each patch.
            patch_stride: Stride between patches.
            num_channels: Number of input channels.
            use_cls_token: Whether to use CLS token.
            num_register_tokens: Number of register tokens.
            use_position_embedding: Whether to add position embeddings.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.use_cls_token = use_cls_token
        self.num_register_tokens = num_register_tokens
        self.use_position_embedding = use_position_embedding

        # Number of patches
        self.num_patches = (input_dim - patch_size) // patch_stride + 1

        # Patch projection
        patch_input_dim = patch_size * num_channels
        self.patch_proj = np.random.randn(patch_input_dim, output_dim) * 0.02
        self.patch_bias = np.zeros(output_dim)

        # Special tokens
        if use_cls_token:
            self.cls_token = np.random.randn(1, output_dim) * 0.02
        if num_register_tokens > 0:
            self.register_tokens = np.random.randn(num_register_tokens, output_dim) * 0.02

        # Position embeddings
        total_tokens = self.num_patches + (1 if use_cls_token else 0) + num_register_tokens
        if use_position_embedding:
            self.position_embeddings = np.random.randn(total_tokens, output_dim) * 0.02

    def patchify(self, x: np.ndarray) -> np.ndarray:
        """Split input into patches.

        Args:
            x: Input [..., (channels,) input_dim].

        Returns:
            Patches [..., num_patches, patch_size * channels].
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # Handle multi-channel
        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] != self.num_channels):
            # Single channel
            flat = x.reshape(-1, self.input_dim)
        else:
            flat = x.reshape(-1, self.num_channels, self.input_dim)

        patches = []
        for i in range(self.num_patches):
            start = i * self.patch_stride
            end = start + self.patch_size
            if flat.ndim == 2:
                patch = flat[:, start:end]
            else:
                patch = flat[:, :, start:end].reshape(flat.shape[0], -1)
            patches.append(patch)

        return np.stack(patches, axis=-2)

    def tokenize(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert spectral input to patch tokens.

        Args:
            x: Input spectral data.

        Returns:
            Tuple of (token_embeddings, info).
        """
        # Extract patches
        patches = self.patchify(x)
        batch_size = patches.shape[0]

        # Project patches
        tokens = np.einsum("...d,do->...o", patches, self.patch_proj) + self.patch_bias

        # Prepend special tokens
        prefix_tokens = []
        if self.use_cls_token:
            cls = np.repeat(self.cls_token[np.newaxis, :], batch_size, axis=0)
            prefix_tokens.append(cls)
        if self.num_register_tokens > 0:
            reg = np.repeat(self.register_tokens[np.newaxis, :], batch_size, axis=0)
            prefix_tokens.append(reg)

        if prefix_tokens:
            prefix = np.concatenate(prefix_tokens, axis=-2)
            tokens = np.concatenate([prefix, tokens], axis=-2)

        # Add position embeddings
        if self.use_position_embedding:
            seq_len = tokens.shape[-2]
            tokens = tokens + self.position_embeddings[:seq_len]

        info: Dict[str, Any] = {
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "total_tokens": tokens.shape[-2],
            "has_cls": self.use_cls_token,
            "num_registers": self.num_register_tokens,
        }

        return tokens, info

    def unpatchify(self, tokens: np.ndarray) -> np.ndarray:
        """Reconstruct spectral data from patch tokens.

        Args:
            tokens: Token embeddings [..., num_tokens, output_dim].

        Returns:
            Reconstructed spectral data [..., input_dim].
        """
        # Remove special tokens
        offset = (1 if self.use_cls_token else 0) + self.num_register_tokens
        patch_tokens = tokens[..., offset:, :]

        # Inverse projection (pseudo-inverse)
        patch_proj_pinv = np.linalg.pinv(self.patch_proj)
        patches = np.einsum("...d,do->...o", patch_tokens, patch_proj_pinv)

        # Reconstruct from patches (simple: just concatenate non-overlapping)
        if self.patch_stride == self.patch_size:
            # Non-overlapping patches
            reconstructed = patches.reshape(*patches.shape[:-2], -1)
        else:
            # Overlapping: average in overlap regions
            batch_shape = patches.shape[:-2]
            output_len = self.num_patches * self.patch_stride + self.patch_size - self.patch_stride
            reconstructed = np.zeros((*batch_shape, output_len))
            counts = np.zeros(output_len)

            for i in range(self.num_patches):
                start = i * self.patch_stride
                end = start + self.patch_size
                reconstructed[..., start:end] += patches[..., i, :self.patch_size]
                counts[start:end] += 1

            reconstructed = reconstructed / (counts + 1e-10)

        # Crop to original size
        return reconstructed[..., :self.input_dim]


class ContinuousTokenizer:
    """Continuous (non-quantized) spectral tokenizer.

    Projects spectral patches into continuous token embeddings without
    discretization. Used when the model operates on continuous representations.

    Attributes:
        input_dim: Input spectral dimension.
        output_dim: Output embedding dimension.
        window_size: Window size for tokenization.
        hop_size: Hop size between windows.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
        window_size: int = 64,
        hop_size: int = 32,
        use_learned_window: bool = True,
        normalize: bool = True,
    ):
        """Initialize continuous tokenizer.

        Args:
            input_dim: Input dimension.
            output_dim: Output embedding dimension.
            window_size: Window size.
            hop_size: Hop size.
            use_learned_window: Whether to use learned windowing.
            normalize: Whether to normalize output embeddings.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.hop_size = hop_size
        self.normalize = normalize

        # Projection
        self.proj = np.random.randn(window_size, output_dim) * 0.02
        self.bias = np.zeros(output_dim)

        # Learned window function
        if use_learned_window:
            # Initialize with Hann window
            self.window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / window_size))
        else:
            self.window = np.ones(window_size)

        # Number of tokens
        self.num_tokens = (input_dim - window_size) // hop_size + 1

    def tokenize(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert spectral input to continuous tokens.

        Args:
            x: Input [..., input_dim].

        Returns:
            Tuple of (token_embeddings, info).
        """
        # Extract windowed frames
        frames = []
        for i in range(self.num_tokens):
            start = i * self.hop_size
            end = start + self.window_size
            frame = x[..., start:end] * self.window
            frames.append(frame)

        frames_array = np.stack(frames, axis=-2)  # [..., num_tokens, window_size]

        # Project
        tokens = np.einsum("...w,wo->...o", frames_array, self.proj) + self.bias

        # Normalize
        if self.normalize:
            norm = np.sqrt(np.sum(tokens ** 2, axis=-1, keepdims=True) + 1e-10)
            tokens = tokens / norm

        info: Dict[str, Any] = {
            "num_tokens": self.num_tokens,
            "window_size": self.window_size,
            "hop_size": self.hop_size,
            "continuous": True,
        }

        return tokens, info

    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Reconstruct from continuous tokens.

        Args:
            tokens: Token embeddings [..., num_tokens, output_dim].

        Returns:
            Reconstructed signal [..., input_dim].
        """
        # Inverse projection
        proj_pinv = np.linalg.pinv(self.proj)
        frames = np.einsum("...d,dw->...w", tokens, proj_pinv)

        # Overlap-add reconstruction
        batch_shape = frames.shape[:-2]
        output_len = (self.num_tokens - 1) * self.hop_size + self.window_size
        reconstructed = np.zeros((*batch_shape, output_len))
        window_sum = np.zeros(output_len)

        num_frames = frames.shape[-2]
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.window_size
            reconstructed[..., start:end] += frames[..., i, :] * self.window
            window_sum[start:end] += self.window ** 2

        # Normalize by window sum
        window_sum = np.maximum(window_sum, 1e-10)
        reconstructed = reconstructed / window_sum

        return reconstructed[..., :self.input_dim]


class HybridTokenizer:
    """Hybrid tokenizer combining discrete and continuous representations.

    Uses VQ-VAE for coarse discrete tokens and continuous embeddings
    for fine-grained details, providing both discrete structure
    (for prediction) and continuous fidelity (for reconstruction).

    Attributes:
        discrete_tokenizer: VQ-VAE tokenizer for discrete codes.
        continuous_tokenizer: Continuous tokenizer for embeddings.
        alpha: Blending factor between discrete and continuous.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        discrete_codebook_size: int = 8192,
        discrete_dim: int = 256,
        continuous_dim: int = 256,
        num_codebooks: int = 4,
        patch_size: int = 16,
        alpha: float = 0.7,
    ):
        """Initialize hybrid tokenizer.

        Args:
            input_dim: Input spectral dimension.
            discrete_codebook_size: Size of discrete codebook.
            discrete_dim: Discrete code dimension.
            continuous_dim: Continuous embedding dimension.
            num_codebooks: Number of VQ codebooks.
            patch_size: Patch size for continuous tokenizer.
            alpha: Weight for discrete vs continuous (1=all discrete).
        """
        self.input_dim = input_dim
        self.alpha = alpha
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        # Discrete tokenizer
        self.discrete_tokenizer = VQVAETokenizer(
            input_dim=input_dim,
            codebook_size=discrete_codebook_size,
            codebook_dim=discrete_dim,
            num_codebooks=num_codebooks,
        )

        # Continuous tokenizer
        self.continuous_tokenizer = ContinuousTokenizer(
            input_dim=input_dim,
            output_dim=continuous_dim,
            window_size=patch_size,
            hop_size=patch_size // 2,
        )

        # Fusion layer
        total_dim = discrete_dim + continuous_dim
        self.fusion_proj = np.random.randn(total_dim, discrete_dim) * 0.02
        self.fusion_bias = np.zeros(discrete_dim)

    def tokenize(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Tokenize with both discrete and continuous representations.

        Args:
            x: Input spectral data [..., input_dim].

        Returns:
            Tuple of (discrete_indices, continuous_tokens, info).
        """
        # Discrete tokenization
        discrete_indices, discrete_info = self.discrete_tokenizer.tokenize(x)

        # Continuous tokenization
        continuous_tokens, continuous_info = self.continuous_tokenizer.tokenize(x)

        info: Dict[str, Any] = {
            "discrete_info": discrete_info,
            "continuous_info": continuous_info,
            "alpha": self.alpha,
        }

        return discrete_indices, continuous_tokens, info

    def detokenize(
        self,
        discrete_indices: Optional[np.ndarray] = None,
        continuous_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct from hybrid tokens.

        Args:
            discrete_indices: Optional discrete token IDs.
            continuous_tokens: Optional continuous embeddings.

        Returns:
            Reconstructed spectral data.
        """
        reconstructions = []

        if discrete_indices is not None:
            discrete_recon = self.discrete_tokenizer.detokenize(discrete_indices)
            reconstructions.append(discrete_recon * self.alpha)

        if continuous_tokens is not None:
            continuous_recon = self.continuous_tokenizer.detokenize(continuous_tokens)
            reconstructions.append(continuous_recon * (1 - self.alpha))

        if len(reconstructions) == 2:
            return reconstructions[0] + reconstructions[1]
        elif len(reconstructions) == 1:
            return reconstructions[0]
        else:
            raise ValueError("At least one of discrete or continuous must be provided")


class SpectralTokenizer:
    """Universal spectral tokenizer with multi-modal support.

    Provides a unified tokenization interface for all spectral modalities,
    handling modality-specific preprocessing, tokenization strategy selection,
    and special token management.

    Attributes:
        tokenizer_type: Active tokenizer type.
        modality_tokenizers: Per-modality tokenizer instances.
        special_tokens: Special token definitions.
        max_tokens: Maximum token sequence length.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
        tokenizer_type: str = "hybrid",
        codebook_size: int = 8192,
        num_codebooks: int = 4,
        patch_size: int = 16,
        max_tokens: int = 4096,
        special_tokens: Optional[Dict[str, int]] = None,
        modality_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize universal spectral tokenizer.

        Args:
            input_dim: Input spectral dimension.
            output_dim: Output token dimension.
            tokenizer_type: Default tokenizer type.
            codebook_size: Codebook size for VQ.
            num_codebooks: Number of VQ codebooks.
            patch_size: Patch size.
            max_tokens: Maximum sequence length.
            special_tokens: Special token mapping.
            modality_configs: Per-modality configuration overrides.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tokenizer_type = tokenizer_type
        self.max_tokens = max_tokens

        # Special tokens
        self.special_tokens = special_tokens or {
            "PAD": 0, "BOS": 1, "EOS": 2, "MASK": 3,
            "SEP": 4, "UNK": 5,
            "SEISMIC": 6, "VIBRATION": 7, "EEG": 8,
            "ECG": 9, "AUDIO": 10, "RF": 11, "SYNTHETIC": 12,
        }
        self.num_special_tokens = len(self.special_tokens)

        # Create tokenizers
        if tokenizer_type == "vqvae":
            self.tokenizer = VQVAETokenizer(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=output_dim,
                num_codebooks=num_codebooks,
            )
        elif tokenizer_type == "patch":
            self.tokenizer = PatchTokenizer(
                input_dim=input_dim,
                output_dim=output_dim,
                patch_size=patch_size,
            )
        elif tokenizer_type == "continuous":
            self.tokenizer = ContinuousTokenizer(
                input_dim=input_dim,
                output_dim=output_dim,
                window_size=patch_size,
            )
        elif tokenizer_type == "hybrid":
            self.tokenizer = HybridTokenizer(
                input_dim=input_dim,
                discrete_codebook_size=codebook_size,
                discrete_dim=output_dim,
                continuous_dim=output_dim,
                num_codebooks=num_codebooks,
                patch_size=patch_size,
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

        # Augmentation pipeline
        self.augmentation = SpectralAugmentation()

        # Modality-specific preprocessing
        self.modality_configs = modality_configs or {}

    def tokenize(
        self,
        x: np.ndarray,
        modality: Optional[str] = None,
        add_special_tokens: bool = True,
        augment: bool = False,
    ) -> Dict[str, Any]:
        """Tokenize spectral input.

        Args:
            x: Input spectral data.
            modality: Optional modality identifier.
            add_special_tokens: Whether to add BOS/EOS tokens.
            augment: Whether to apply augmentation.

        Returns:
            Dictionary with tokenization results.
        """
        # Apply augmentation if requested
        if augment:
            x = self.augmentation(x)

        # Tokenize based on type
        if self.tokenizer_type == "hybrid":
            discrete_ids, continuous_tokens, info = self.tokenizer.tokenize(x)
            result: Dict[str, Any] = {
                "discrete_ids": discrete_ids,
                "continuous_tokens": continuous_tokens,
                "info": info,
            }
        elif self.tokenizer_type == "vqvae":
            token_ids, info = self.tokenizer.tokenize(x)
            result = {"token_ids": token_ids, "info": info}
        elif self.tokenizer_type == "patch":
            tokens, info = self.tokenizer.tokenize(x)
            result = {"embeddings": tokens, "info": info}
        elif self.tokenizer_type == "continuous":
            tokens, info = self.tokenizer.tokenize(x)
            result = {"embeddings": tokens, "info": info}
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

        # Add modality info
        if modality is not None:
            modality_upper = modality.upper()
            if modality_upper in self.special_tokens:
                result["modality_token"] = self.special_tokens[modality_upper]
            result["modality"] = modality

        return result

    def detokenize(self, tokens: Dict[str, Any]) -> np.ndarray:
        """Reconstruct spectral data from tokens.

        Args:
            tokens: Tokenization output dictionary.

        Returns:
            Reconstructed spectral data.
        """
        if self.tokenizer_type == "hybrid":
            return self.tokenizer.detokenize(
                discrete_indices=tokens.get("discrete_ids"),
                continuous_tokens=tokens.get("continuous_tokens"),
            )
        elif self.tokenizer_type == "vqvae":
            return self.tokenizer.detokenize(tokens["token_ids"])
        elif self.tokenizer_type in ("patch", "continuous"):
            return self.tokenizer.detokenize(tokens["embeddings"])
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    def get_vocab_size(self) -> int:
        """Get total vocabulary size including special tokens."""
        if self.tokenizer_type in ("vqvae", "hybrid"):
            return self.num_special_tokens + self.tokenizer.discrete_tokenizer.codebook_size \
                if self.tokenizer_type == "hybrid" else \
                self.num_special_tokens + self.tokenizer.codebook_size
        return self.num_special_tokens

    def create_contrastive_pair(
        self, x: np.ndarray, modality: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Create augmented pair for contrastive pretraining.

        Args:
            x: Input spectral data.
            modality: Optional modality.

        Returns:
            Tuple of (view1_tokens, view2_tokens, augmentation_info).
        """
        view1, view2, aug_info = self.augmentation.create_positive_pair(x)

        tokens1 = self.tokenize(view1, modality=modality, augment=False)
        tokens2 = self.tokenize(view2, modality=modality, augment=False)

        return tokens1, tokens2, aug_info
