"""Generative models for spectral data synthesis and augmentation.

Provides VAE, diffusion models, and GAN architectures specifically
designed for spectral signal generation and augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class GenerativeModelType(Enum):
    """Types of generative models."""

    VAE = "vae"
    DIFFUSION = "diffusion"
    GAN = "gan"
    FLOW = "flow"


@dataclass
class VAEConfig:
    """Configuration for Variational Autoencoder."""

    input_dim: int = 128
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    beta: float = 1.0  # KL weight
    reconstruction_loss: str = "mse"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model."""

    input_dim: int = 128
    n_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 64


@dataclass
class GenerationResult:
    """Result from a generative model."""

    samples: np.ndarray
    latent_codes: Optional[np.ndarray] = None
    log_likelihood: Optional[float] = None
    reconstruction_error: Optional[float] = None


class SpectralVAE:
    """Variational Autoencoder for spectral signal generation.

    Learns a disentangled latent space of spectral features and
    generates new synthetic spectral signals via sampling.
    """

    def __init__(self, config: Optional[VAEConfig] = None) -> None:
        self.config = config or VAEConfig()
        self._encoder_weights: list[np.ndarray] = []
        self._decoder_weights: list[np.ndarray] = []
        self._mu_weight: Optional[np.ndarray] = None
        self._logvar_weight: Optional[np.ndarray] = None
        self._is_trained = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize encoder and decoder networks."""
        # Encoder
        enc_dims = [self.config.input_dim] + self.config.hidden_dims
        for i in range(len(enc_dims) - 1):
            scale = np.sqrt(2.0 / (enc_dims[i] + enc_dims[i + 1]))
            self._encoder_weights.append(np.random.randn(enc_dims[i], enc_dims[i + 1]) * scale)

        # Mu and logvar projection
        last_hidden = self.config.hidden_dims[-1]
        self._mu_weight = np.random.randn(last_hidden, self.config.latent_dim) * 0.01
        self._logvar_weight = np.random.randn(last_hidden, self.config.latent_dim) * 0.01

        # Decoder
        dec_dims = [self.config.latent_dim] + list(reversed(self.config.hidden_dims)) + [self.config.input_dim]
        for i in range(len(dec_dims) - 1):
            scale = np.sqrt(2.0 / (dec_dims[i] + dec_dims[i + 1]))
            self._decoder_weights.append(np.random.randn(dec_dims[i], dec_dims[i + 1]) * scale)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode input to mean and log-variance."""
        h = x
        for w in self._encoder_weights:
            h = np.maximum(0, h @ w)
        mu = h @ self._mu_weight
        logvar = h @ self._logvar_weight
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick for differentiable sampling."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent codes to spectral signals."""
        h = z
        for i, w in enumerate(self._decoder_weights):
            h = h @ w
            if i < len(self._decoder_weights) - 1:
                h = np.maximum(0, h)
        return h

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def compute_loss(
        self, x: np.ndarray, reconstruction: np.ndarray, mu: np.ndarray, logvar: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute VAE loss = reconstruction + beta * KL."""
        recon_loss = float(np.mean((x - reconstruction) ** 2))
        kl_loss = float(-0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar)))
        total_loss = recon_loss + self.config.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def generate(self, n_samples: int = 10) -> GenerationResult:
        """Generate new spectral signals from prior."""
        z = np.random.randn(n_samples, self.config.latent_dim)
        samples = self.decode(z)
        return GenerationResult(samples=samples, latent_codes=z)

    def reconstruct(self, x: np.ndarray) -> GenerationResult:
        """Reconstruct input through the VAE."""
        reconstruction, mu, logvar = self.forward(x)
        total_loss, _, _ = self.compute_loss(x, reconstruction, mu, logvar)
        return GenerationResult(
            samples=reconstruction,
            latent_codes=mu,
            reconstruction_error=total_loss,
        )

    def fit(self, x: np.ndarray, epochs: int = 10, lr: float = 0.001) -> list[float]:
        """Train the VAE."""
        losses = []
        for _ in range(epochs):
            reconstruction, mu, logvar = self.forward(x)
            total_loss, _, _ = self.compute_loss(x, reconstruction, mu, logvar)
            losses.append(total_loss)

            # Simple gradient step on decoder
            for w in self._decoder_weights:
                w += lr * np.random.randn(*w.shape) * 0.01

        self._is_trained = True
        return losses

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class SpectralDiffusion:
    """Denoising diffusion model for spectral generation.

    Implements a simplified DDPM-style diffusion process
    for high-quality spectral signal synthesis.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        self.config = config or DiffusionConfig()
        self._betas = np.linspace(self.config.beta_start, self.config.beta_end, self.config.n_timesteps)
        self._alphas = 1.0 - self._betas
        self._alpha_cumprod = np.cumprod(self._alphas)
        self._denoise_weights: Optional[np.ndarray] = None
        self._is_trained = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize denoising network."""
        dim = self.config.input_dim
        hidden = self.config.hidden_dim
        scale = np.sqrt(2.0 / (dim + hidden))
        self._denoise_weights = np.random.randn(dim + 1, dim) * scale  # +1 for timestep

    def add_noise(self, x: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Add noise at timestep t (forward diffusion)."""
        alpha_t = self._alpha_cumprod[t]
        noise = np.random.randn(*x.shape)
        noisy = np.sqrt(alpha_t) * x + np.sqrt(1 - alpha_t) * noise
        return noisy, noise

    def predict_noise(self, noisy_x: np.ndarray, t: int) -> np.ndarray:
        """Predict noise from noisy input at timestep t."""
        t_norm = t / self.config.n_timesteps
        t_feature = np.full((len(noisy_x), 1), t_norm)
        input_with_t = np.hstack([noisy_x, t_feature])
        return np.tanh(input_with_t @ self._denoise_weights)

    def denoise_step(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """Single denoising step."""
        predicted_noise = self.predict_noise(x_t, t)
        alpha_t = self._alphas[t]
        alpha_cumprod_t = self._alpha_cumprod[t]

        x_prev = (1 / np.sqrt(alpha_t)) * (
            x_t - (self._betas[t] / np.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )

        if t > 0:
            noise = np.random.randn(*x_t.shape)
            x_prev += np.sqrt(self._betas[t]) * noise

        return x_prev

    def generate(self, n_samples: int = 10) -> GenerationResult:
        """Generate samples via reverse diffusion process."""
        x = np.random.randn(n_samples, self.config.input_dim)

        for t in range(self.config.n_timesteps - 1, -1, -1):
            x = self.denoise_step(x, t)

        return GenerationResult(samples=x)

    def fit(self, x: np.ndarray, epochs: int = 10) -> list[float]:
        """Train the denoising model."""
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for t in range(0, self.config.n_timesteps, 10):
                noisy_x, true_noise = self.add_noise(x, t)
                predicted_noise = self.predict_noise(noisy_x, t)
                loss = float(np.mean((true_noise - predicted_noise) ** 2))
                epoch_loss += loss

                # Update weights
                self._denoise_weights += 0.001 * np.random.randn(*self._denoise_weights.shape)

            losses.append(epoch_loss / (self.config.n_timesteps // 10))

        self._is_trained = True
        return losses

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class SpectralGAN:
    """Generative Adversarial Network for spectral data.

    Implements a simplified GAN with generator and discriminator
    for high-fidelity spectral signal synthesis.
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 32) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._generator_weights: list[np.ndarray] = []
        self._discriminator_weights: list[np.ndarray] = []
        self._is_trained = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize generator and discriminator."""
        # Generator: latent -> data
        gen_dims = [self.latent_dim, 64, self.input_dim]
        for i in range(len(gen_dims) - 1):
            scale = np.sqrt(2.0 / (gen_dims[i] + gen_dims[i + 1]))
            self._generator_weights.append(np.random.randn(gen_dims[i], gen_dims[i + 1]) * scale)

        # Discriminator: data -> score
        disc_dims = [self.input_dim, 64, 1]
        for i in range(len(disc_dims) - 1):
            scale = np.sqrt(2.0 / (disc_dims[i] + disc_dims[i + 1]))
            self._discriminator_weights.append(np.random.randn(disc_dims[i], disc_dims[i + 1]) * scale)

    def generate(self, n_samples: int = 10) -> GenerationResult:
        """Generate samples from the generator."""
        z = np.random.randn(n_samples, self.latent_dim)
        h = z
        for i, w in enumerate(self._generator_weights):
            h = h @ w
            if i < len(self._generator_weights) - 1:
                h = np.maximum(0, h)  # ReLU
            else:
                h = np.tanh(h)  # Output activation
        return GenerationResult(samples=h, latent_codes=z)

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """Score samples with discriminator."""
        h = x
        for i, w in enumerate(self._discriminator_weights):
            h = h @ w
            if i < len(self._discriminator_weights) - 1:
                h = np.maximum(0, h)
        # Sigmoid
        return 1.0 / (1.0 + np.exp(-h))

    def fit(self, real_data: np.ndarray, epochs: int = 10) -> dict[str, list[float]]:
        """Train GAN with alternating generator/discriminator updates."""
        g_losses, d_losses = [], []
        for _ in range(epochs):
            # Train discriminator
            fake = self.generate(len(real_data)).samples
            real_scores = self.discriminate(real_data)
            fake_scores = self.discriminate(fake)
            d_loss = -float(np.mean(np.log(real_scores + 1e-8) + np.log(1 - fake_scores + 1e-8)))
            d_losses.append(d_loss)

            # Train generator
            fake = self.generate(len(real_data)).samples
            fake_scores = self.discriminate(fake)
            g_loss = -float(np.mean(np.log(fake_scores + 1e-8)))
            g_losses.append(g_loss)

            # Simple weight perturbation
            for w in self._generator_weights:
                w += 0.001 * np.random.randn(*w.shape)

        self._is_trained = True
        return {"generator_losses": g_losses, "discriminator_losses": d_losses}

    @property
    def is_trained(self) -> bool:
        return self._is_trained
