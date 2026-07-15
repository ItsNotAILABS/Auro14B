"""Benchmark suites for spectral foundation model evaluation.

Standardized benchmarks for comparing pretrained spectral models
across multiple tasks and modalities.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mesie.foundation.evaluation.metrics import (
    ReconstructionMetrics,
    RepresentationMetrics,
    DownstreamMetrics,
)
from mesie.foundation.evaluation.probing import LinearProbe, KNNProbe


class SpectralBenchmarkSuite:
    """Complete benchmark suite for spectral foundation models.

    Runs a standardized set of evaluations covering:
    - Reconstruction quality
    - Representation quality
    - Cross-modal transfer
    - Few-shot learning
    - Anomaly detection
    - Frequency resolution

    Attributes:
        benchmarks: List of active benchmarks.
        results: Accumulated results.
    """

    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        sample_rate: float = 1.0,
        latent_dim: int = 768,
    ):
        """Initialize benchmark suite.

        Args:
            modalities: Modalities to evaluate.
            sample_rate: Signal sampling rate.
            latent_dim: Latent space dimension.
        """
        self.modalities = modalities or [
            "seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"
        ]
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim

        self.reconstruction_metrics = ReconstructionMetrics(sample_rate)
        self.representation_metrics = RepresentationMetrics(latent_dim)
        self.downstream_metrics = DownstreamMetrics()

        self.results: Dict[str, Any] = {}

    def run_all(
        self,
        encode_fn: Callable[[np.ndarray, str], np.ndarray],
        decode_fn: Optional[Callable[[np.ndarray, str], np.ndarray]] = None,
        data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Run complete benchmark suite.

        Args:
            encode_fn: Function mapping (signal, modality) -> embedding.
            decode_fn: Optional decoder for reconstruction tests.
            data: Optional test data per modality.

        Returns:
            Complete benchmark results.
        """
        start_time = time.time()

        # Generate test data if not provided
        if data is None:
            data = self._generate_test_data()

        results: Dict[str, Any] = {
            "metadata": {
                "modalities": self.modalities,
                "latent_dim": self.latent_dim,
                "sample_rate": self.sample_rate,
            }
        }

        # 1. Representation quality
        results["representation"] = self._evaluate_representations(encode_fn, data)

        # 2. Reconstruction quality (if decoder available)
        if decode_fn is not None:
            results["reconstruction"] = self._evaluate_reconstruction(
                encode_fn, decode_fn, data
            )

        # 3. Cross-modal retrieval
        results["cross_modal"] = self._evaluate_cross_modal(encode_fn, data)

        # 4. Few-shot classification
        results["few_shot"] = self._evaluate_few_shot(encode_fn, data)

        # 5. Anomaly detection
        results["anomaly_detection"] = self._evaluate_anomaly_detection(encode_fn, data)

        results["total_time_seconds"] = time.time() - start_time
        self.results = results
        return results

    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic test data for each modality."""
        data = {}
        n_samples = 100
        signal_length = 1024

        for modality in self.modalities:
            if modality == "seismic":
                # P-wave and S-wave signals
                signals = np.random.randn(n_samples, signal_length) * 0.5
                for i in range(n_samples):
                    arrival = np.random.randint(100, 400)
                    freq = np.random.uniform(1, 20)
                    t = np.arange(signal_length) / self.sample_rate
                    envelope = np.exp(-(t - arrival / self.sample_rate) ** 2 * freq ** 2)
                    signals[i] += envelope * np.sin(2 * np.pi * freq * t)
            elif modality == "audio":
                signals = np.random.randn(n_samples, signal_length) * 0.1
                for i in range(n_samples):
                    freq = np.random.uniform(100, 4000)
                    t = np.arange(signal_length) / self.sample_rate
                    signals[i] += np.sin(2 * np.pi * freq * t)
            else:
                signals = np.random.randn(n_samples, signal_length)

            data[modality] = signals

        return data

    def _evaluate_representations(
        self,
        encode_fn: Callable,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate representation quality."""
        embeddings = {}
        for modality, signals in data.items():
            embeddings[modality] = encode_fn(signals, modality)

        # Per-modality metrics
        results: Dict[str, Any] = {}
        for modality, emb in embeddings.items():
            results[modality] = self.representation_metrics.evaluate_all(emb)

        # Cross-modality metrics
        results["modality_separability"] = \
            self.representation_metrics.modality_separability(embeddings)

        # Overall uniformity
        all_emb = np.concatenate(list(embeddings.values()), axis=0)
        results["overall_uniformity"] = self.representation_metrics.uniformity(all_emb)
        results["overall_isotropy"] = self.representation_metrics.isotropy(all_emb)

        return results

    def _evaluate_reconstruction(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate reconstruction quality."""
        results: Dict[str, Any] = {}

        for modality, signals in data.items():
            embeddings = encode_fn(signals, modality)
            reconstructions = decode_fn(embeddings, modality)

            modality_metrics = []
            for i in range(min(len(signals), 50)):
                metrics = self.reconstruction_metrics.evaluate(
                    signals[i], reconstructions[i]
                )
                modality_metrics.append(metrics)

            # Average metrics
            avg_metrics = {}
            for key in modality_metrics[0]:
                values = [m[key] for m in modality_metrics]
                avg_metrics[f"{key}_mean"] = float(np.mean(values))
                avg_metrics[f"{key}_std"] = float(np.std(values))

            results[modality] = avg_metrics

        return results

    def _evaluate_cross_modal(
        self,
        encode_fn: Callable,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate cross-modal retrieval."""
        embeddings = {}
        for modality, signals in data.items():
            embeddings[modality] = encode_fn(signals, modality)

        results: Dict[str, Any] = {}
        modality_list = list(embeddings.keys())

        for i, mod_a in enumerate(modality_list):
            for mod_b in modality_list[i+1:]:
                emb_a = embeddings[mod_a]
                emb_b = embeddings[mod_b]

                # Normalize
                emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=-1, keepdims=True) + 1e-8)
                emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=-1, keepdims=True) + 1e-8)

                # Cross-modal similarity
                sim = np.dot(emb_a_norm, emb_b_norm.T)
                mean_sim = float(np.mean(sim))
                max_sim = float(np.mean(np.max(sim, axis=-1)))

                results[f"{mod_a}-{mod_b}"] = {
                    "mean_similarity": mean_sim,
                    "mean_max_similarity": max_sim,
                }

        return results

    def _evaluate_few_shot(
        self,
        encode_fn: Callable,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate few-shot classification."""
        results: Dict[str, Any] = {}

        # Create classification task: identify modality from embedding
        all_embeddings = []
        all_labels = []

        for i, (modality, signals) in enumerate(data.items()):
            emb = encode_fn(signals, modality)
            all_embeddings.append(emb)
            all_labels.append(np.full(len(emb), i))

        X = np.concatenate(all_embeddings, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # Few-shot: use only k samples per class
        for k in [1, 5, 10, 20]:
            train_idx = []
            test_idx = []

            for cls in range(len(data)):
                cls_idx = np.where(y == cls)[0]
                perm = np.random.permutation(cls_idx)
                train_idx.extend(perm[:k].tolist())
                test_idx.extend(perm[k:].tolist())

            if not test_idx:
                continue

            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)

            # KNN evaluation
            knn = KNNProbe(k=min(k, 5))
            knn.fit(X[train_idx], y[train_idx])
            knn_results = knn.evaluate(X[test_idx], y[test_idx])

            # Linear probe
            probe = LinearProbe(X.shape[1], len(data), max_epochs=50)
            probe.fit(X[train_idx], y[train_idx])
            linear_results = probe.evaluate(X[test_idx], y[test_idx])

            results[f"{k}_shot"] = {
                "knn_accuracy": knn_results["accuracy"],
                "linear_accuracy": linear_results["accuracy"],
            }

        return results

    def _evaluate_anomaly_detection(
        self,
        encode_fn: Callable,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate anomaly detection capability."""
        results: Dict[str, Any] = {}

        for modality, signals in data.items():
            # Use first half as "normal", add anomalies to second half
            n = len(signals)
            normal = signals[:n//2]
            anomalous = signals[n//2:].copy()

            # Inject anomalies
            for i in range(len(anomalous)):
                anomaly_type = np.random.choice(["spike", "dropout", "frequency_shift"])
                if anomaly_type == "spike":
                    pos = np.random.randint(0, anomalous.shape[1])
                    anomalous[i, pos] += np.random.randn() * 10
                elif anomaly_type == "dropout":
                    start = np.random.randint(0, anomalous.shape[1] - 50)
                    anomalous[i, start:start+50] = 0
                elif anomaly_type == "frequency_shift":
                    anomalous[i] = np.roll(anomalous[i], 100)

            # Encode
            normal_emb = encode_fn(normal, modality)
            anomalous_emb = encode_fn(anomalous, modality)

            # Compute anomaly scores (distance from normal centroid)
            centroid = np.mean(normal_emb, axis=0)
            normal_dists = np.linalg.norm(normal_emb - centroid, axis=-1)
            anomalous_dists = np.linalg.norm(anomalous_emb - centroid, axis=-1)

            # AUC-like metric
            threshold = np.percentile(normal_dists, 95)
            detection_rate = float(np.mean(anomalous_dists > threshold))
            false_positive_rate = float(np.mean(normal_dists > threshold))

            results[modality] = {
                "detection_rate": detection_rate,
                "false_positive_rate": false_positive_rate,
                "separation": float(np.mean(anomalous_dists) - np.mean(normal_dists)),
            }

        return results


class CrossModalRetrievalBenchmark:
    """Cross-modal retrieval benchmark.

    Tests ability to retrieve semantically similar signals
    across different modalities.
    """

    def __init__(self, top_k: List[int] = None):
        """Initialize retrieval benchmark.

        Args:
            top_k: List of k values for recall@k.
        """
        self.top_k = top_k or [1, 5, 10, 20]

    def evaluate(
        self,
        query_embeddings: np.ndarray,
        gallery_embeddings: np.ndarray,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate retrieval performance.

        Args:
            query_embeddings: Query embeddings [Q, D].
            gallery_embeddings: Gallery embeddings [G, D].
            query_labels: Query labels [Q].
            gallery_labels: Gallery labels [G].

        Returns:
            Retrieval metrics.
        """
        # Normalize
        q_norm = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=-1, keepdims=True) + 1e-8
        )
        g_norm = gallery_embeddings / (
            np.linalg.norm(gallery_embeddings, axis=-1, keepdims=True) + 1e-8
        )

        # Compute similarity
        similarity = np.dot(q_norm, g_norm.T)

        # Sort by similarity
        sorted_indices = np.argsort(similarity, axis=-1)[:, ::-1]

        metrics: Dict[str, float] = {}

        # Recall@k
        for k in self.top_k:
            recall = 0.0
            for i in range(len(query_labels)):
                top_k_labels = gallery_labels[sorted_indices[i, :k]]
                if query_labels[i] in top_k_labels:
                    recall += 1.0
            metrics[f"recall@{k}"] = recall / len(query_labels)

        # Mean Reciprocal Rank
        mrr = 0.0
        for i in range(len(query_labels)):
            retrieved_labels = gallery_labels[sorted_indices[i]]
            ranks = np.where(retrieved_labels == query_labels[i])[0]
            if len(ranks) > 0:
                mrr += 1.0 / (ranks[0] + 1)
        metrics["mrr"] = mrr / len(query_labels)

        return metrics


class FewShotClassificationBenchmark:
    """Few-shot classification benchmark.

    Evaluates how well pretrained representations support
    classification with very limited labeled data.
    """

    def __init__(
        self,
        n_ways: List[int] = None,
        k_shots: List[int] = None,
        num_episodes: int = 100,
    ):
        """Initialize few-shot benchmark.

        Args:
            n_ways: Number of classes per episode.
            k_shots: Number of support examples per class.
            num_episodes: Number of evaluation episodes.
        """
        self.n_ways = n_ways or [5, 10, 20]
        self.k_shots = k_shots or [1, 5, 10]
        self.num_episodes = num_episodes

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Run few-shot evaluation.

        Args:
            embeddings: All embeddings [N, D].
            labels: All labels [N].

        Returns:
            Few-shot classification results.
        """
        results: Dict[str, float] = {}
        unique_classes = np.unique(labels)

        for n_way in self.n_ways:
            if n_way > len(unique_classes):
                continue

            for k_shot in self.k_shots:
                accuracies = []

                for episode in range(self.num_episodes):
                    # Sample classes
                    episode_classes = np.random.choice(
                        unique_classes, n_way, replace=False
                    )

                    support_emb = []
                    support_labels = []
                    query_emb = []
                    query_labels = []

                    for cls_idx, cls in enumerate(episode_classes):
                        cls_mask = labels == cls
                        cls_emb = embeddings[cls_mask]

                        if len(cls_emb) < k_shot + 5:
                            continue

                        perm = np.random.permutation(len(cls_emb))
                        support_emb.append(cls_emb[perm[:k_shot]])
                        support_labels.extend([cls_idx] * k_shot)
                        query_emb.append(cls_emb[perm[k_shot:k_shot+5]])
                        query_labels.extend([cls_idx] * min(5, len(cls_emb) - k_shot))

                    if not support_emb or not query_emb:
                        continue

                    support_emb = np.concatenate(support_emb, axis=0)
                    query_emb = np.concatenate(query_emb, axis=0)
                    support_labels = np.array(support_labels)
                    query_labels = np.array(query_labels)

                    # Prototype classification
                    prototypes = []
                    for cls_idx in range(n_way):
                        cls_support = support_emb[support_labels == cls_idx]
                        prototypes.append(np.mean(cls_support, axis=0))
                    prototypes = np.array(prototypes)

                    # Classify queries
                    dists = np.sum(
                        (query_emb[:, None] - prototypes[None, :]) ** 2, axis=-1
                    )
                    predictions = np.argmin(dists, axis=-1)
                    acc = float(np.mean(predictions == query_labels))
                    accuracies.append(acc)

                if accuracies:
                    results[f"{n_way}way_{k_shot}shot_mean"] = float(np.mean(accuracies))
                    results[f"{n_way}way_{k_shot}shot_std"] = float(np.std(accuracies))

        return results


class AnomalyDetectionBenchmark:
    """Anomaly detection benchmark.

    Tests how well pretrained representations support
    unsupervised anomaly detection.
    """

    def __init__(
        self,
        contamination_ratios: Optional[List[float]] = None,
        methods: Optional[List[str]] = None,
    ):
        """Initialize anomaly detection benchmark.

        Args:
            contamination_ratios: Fraction of anomalies.
            methods: Detection methods to evaluate.
        """
        self.contamination_ratios = contamination_ratios or [0.01, 0.05, 0.1]
        self.methods = methods or ["distance", "density", "isolation"]

    def evaluate(
        self,
        normal_embeddings: np.ndarray,
        anomalous_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate anomaly detection.

        Args:
            normal_embeddings: Normal sample embeddings.
            anomalous_embeddings: Anomalous sample embeddings.

        Returns:
            Detection metrics.
        """
        results: Dict[str, float] = {}

        for method in self.methods:
            if method == "distance":
                metrics = self._distance_based(normal_embeddings, anomalous_embeddings)
            elif method == "density":
                metrics = self._density_based(normal_embeddings, anomalous_embeddings)
            elif method == "isolation":
                metrics = self._isolation_based(normal_embeddings, anomalous_embeddings)
            else:
                continue

            for key, val in metrics.items():
                results[f"{method}_{key}"] = val

        return results

    def _distance_based(
        self, normal: np.ndarray, anomalous: np.ndarray
    ) -> Dict[str, float]:
        """Distance-based anomaly detection."""
        centroid = np.mean(normal, axis=0)
        normal_dists = np.linalg.norm(normal - centroid, axis=-1)
        anomaly_dists = np.linalg.norm(anomalous - centroid, axis=-1)

        # AUC approximation
        all_dists = np.concatenate([normal_dists, anomaly_dists])
        all_labels = np.concatenate([
            np.zeros(len(normal_dists)),
            np.ones(len(anomaly_dists))
        ])

        # Sort by score
        sorted_idx = np.argsort(all_dists)[::-1]
        sorted_labels = all_labels[sorted_idx]

        # Compute AUC
        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        tpr = tp / (np.sum(all_labels) + 1e-10)
        fpr = fp / (np.sum(1 - all_labels) + 1e-10)
        auc = float(np.trapz(tpr, fpr))

        return {
            "auc": auc,
            "mean_normal_dist": float(np.mean(normal_dists)),
            "mean_anomaly_dist": float(np.mean(anomaly_dists)),
        }

    def _density_based(
        self, normal: np.ndarray, anomalous: np.ndarray
    ) -> Dict[str, float]:
        """Density-based anomaly detection (simplified LOF)."""
        k = min(10, len(normal) - 1)

        # Compute k-NN distances for normal points
        normal_dists_matrix = np.sum(
            (normal[:, None] - normal[None, :]) ** 2, axis=-1
        )
        np.fill_diagonal(normal_dists_matrix, np.inf)
        normal_knn_dists = np.sort(normal_dists_matrix, axis=-1)[:, :k]
        normal_avg_knn = np.mean(normal_knn_dists, axis=-1)

        # Compute k-NN distances for anomalous points (to normal points)
        anomaly_dists_matrix = np.sum(
            (anomalous[:, None] - normal[None, :]) ** 2, axis=-1
        )
        anomaly_knn_dists = np.sort(anomaly_dists_matrix, axis=-1)[:, :k]
        anomaly_avg_knn = np.mean(anomaly_knn_dists, axis=-1)

        # LOF-like score: ratio of anomaly density to normal density
        threshold = np.percentile(normal_avg_knn, 95)
        detection_rate = float(np.mean(anomaly_avg_knn > threshold))

        return {
            "detection_rate_at_5pct_fpr": detection_rate,
            "mean_normal_density": float(np.mean(normal_avg_knn)),
            "mean_anomaly_density": float(np.mean(anomaly_avg_knn)),
        }

    def _isolation_based(
        self, normal: np.ndarray, anomalous: np.ndarray
    ) -> Dict[str, float]:
        """Simplified isolation-based scoring."""
        # Random projections for isolation
        n_projections = 100
        dim = normal.shape[-1]

        projections = np.random.randn(n_projections, dim)
        projections /= np.linalg.norm(projections, axis=-1, keepdims=True)

        # Project all points
        normal_proj = np.dot(normal, projections.T)
        anomaly_proj = np.dot(anomalous, projections.T)

        # Isolation score: how often a point is at extreme of projection
        normal_scores = np.zeros(len(normal))
        anomaly_scores = np.zeros(len(anomalous))

        all_proj = np.concatenate([normal_proj, anomaly_proj], axis=0)

        for p in range(n_projections):
            proj_min = np.min(all_proj[:, p])
            proj_max = np.max(all_proj[:, p])
            proj_range = proj_max - proj_min + 1e-10

            # How extreme is each point?
            normal_extremity = np.minimum(
                normal_proj[:, p] - proj_min,
                proj_max - normal_proj[:, p]
            ) / proj_range
            anomaly_extremity = np.minimum(
                anomaly_proj[:, p] - proj_min,
                proj_max - anomaly_proj[:, p]
            ) / proj_range

            normal_scores += (1 - normal_extremity)
            anomaly_scores += (1 - anomaly_extremity)

        normal_scores /= n_projections
        anomaly_scores /= n_projections

        threshold = np.percentile(normal_scores, 95)
        detection_rate = float(np.mean(anomaly_scores > threshold))

        return {
            "detection_rate_at_5pct_fpr": detection_rate,
            "mean_normal_score": float(np.mean(normal_scores)),
            "mean_anomaly_score": float(np.mean(anomaly_scores)),
        }
