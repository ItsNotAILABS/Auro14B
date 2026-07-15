"""Inference engine for spectral AI models.

Provides batch inference, confidence estimation, and prediction
result packaging for production deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class PredictionResult:
    """Result from a model inference call.

    Args:
        predictions: Model output predictions.
        confidence: Confidence scores for each prediction.
        latent_features: Optional latent representations.
        metadata: Additional inference metadata.
    """

    predictions: np.ndarray
    confidence: np.ndarray
    latent_features: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_prediction(self) -> Any:
        """Return the highest-confidence prediction."""
        idx = np.argmax(self.confidence)
        return self.predictions[idx] if self.predictions.ndim > 1 else self.predictions[idx]

    @property
    def mean_confidence(self) -> float:
        """Return mean confidence across predictions."""
        return float(np.mean(self.confidence))


class InferenceEngine:
    """Production inference engine for spectral models.

    Handles batching, preprocessing, postprocessing, and confidence
    calibration for deployed models.

    Args:
        model: Trained model instance (autoencoder, classifier, or transformer).
        model_type: Type of model ('autoencoder', 'classifier', 'transformer').
        confidence_threshold: Minimum confidence for valid predictions.
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "autoencoder",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._inference_count = 0
        self._preprocessing_fn: Optional[Any] = None
        self._postprocessing_fn: Optional[Any] = None

    def set_preprocessing(self, fn: Any) -> None:
        """Set a preprocessing function applied before inference."""
        self._preprocessing_fn = fn

    def set_postprocessing(self, fn: Any) -> None:
        """Set a postprocessing function applied after inference."""
        self._postprocessing_fn = fn

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to input data."""
        data = np.atleast_2d(data)
        if self._preprocessing_fn is not None:
            data = self._preprocessing_fn(data)
        return data

    def _compute_confidence_autoencoder(self, data: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Compute confidence based on reconstruction quality."""
        mse_per_sample = np.mean((data - reconstructed) ** 2, axis=1)
        # Lower MSE = higher confidence
        max_mse = np.max(mse_per_sample) + 1e-10
        confidence = 1.0 - (mse_per_sample / max_mse)
        return confidence

    def _compute_confidence_classifier(self, proba: np.ndarray) -> np.ndarray:
        """Compute confidence from class probabilities (max probability)."""
        return np.max(proba, axis=1)

    def _compute_confidence_transformer(self, features: np.ndarray) -> np.ndarray:
        """Compute confidence from feature magnitude."""
        norms = np.linalg.norm(features, axis=-1)
        max_norm = np.max(norms) + 1e-10
        return norms / max_norm

    def predict(self, data: np.ndarray) -> PredictionResult:
        """Run inference on input data.

        Args:
            data: Input spectral data.

        Returns:
            PredictionResult with predictions and confidence.
        """
        processed = self._preprocess(data)
        self._inference_count += 1

        if self.model_type == "autoencoder":
            latent = self.model.encode(processed)
            reconstructed = self.model.decode(latent)
            confidence = self._compute_confidence_autoencoder(processed, reconstructed)
            predictions = reconstructed
            if self._postprocessing_fn:
                predictions = self._postprocessing_fn(predictions)
            return PredictionResult(
                predictions=predictions,
                confidence=confidence,
                latent_features=latent,
                metadata={"model_type": "autoencoder", "inference_id": self._inference_count},
            )

        elif self.model_type == "classifier":
            proba = self.model.predict_proba(processed)
            labels = np.argmax(proba, axis=1)
            confidence = self._compute_confidence_classifier(proba)
            return PredictionResult(
                predictions=labels,
                confidence=confidence,
                latent_features=None,
                metadata={
                    "model_type": "classifier",
                    "class_probabilities": proba,
                    "inference_id": self._inference_count,
                },
            )

        elif self.model_type == "transformer":
            if processed.ndim == 2 and processed.shape[0] > 1:
                # Batch inference
                features_list = []
                for sample in processed:
                    feat = self.model.extract_features(sample)
                    features_list.append(feat)
                features = np.array(features_list)
            else:
                features = self.model.extract_features(processed.squeeze())
                features = features[np.newaxis, :]

            confidence = self._compute_confidence_transformer(features)
            return PredictionResult(
                predictions=features,
                confidence=confidence,
                latent_features=features,
                metadata={"model_type": "transformer", "inference_id": self._inference_count},
            )

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def predict_batch(self, data_list: list[np.ndarray]) -> list[PredictionResult]:
        """Run inference on a list of inputs.

        Args:
            data_list: List of input arrays.

        Returns:
            List of PredictionResult objects.
        """
        return [self.predict(d) for d in data_list]

    def is_confident(self, result: PredictionResult) -> bool:
        """Check if all predictions meet confidence threshold."""
        return bool(np.all(result.confidence >= self.confidence_threshold))

    @property
    def total_inferences(self) -> int:
        """Total number of inference calls made."""
        return self._inference_count
