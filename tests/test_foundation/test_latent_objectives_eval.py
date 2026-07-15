"""Tests for latent space, objectives, and evaluation modules."""

import numpy as np
import pytest

from mesie.foundation.latent.universal_latent_space import (
    UniversalSpectralLatentSpace,
    LatentSpaceConfig,
    ModalityProjector,
    MomentumEncoder,
    PrototypeLayer,
)
from mesie.foundation.latent.alignment import (
    ContrastiveAligner,
    DistillationAligner,
    OptimalTransportAligner,
)
from mesie.foundation.latent.projections import (
    LinearProjection,
    MLPProjection,
    GatedProjection,
    ModalityAdaptiveProjection,
    ProjectionFactory,
)
from mesie.foundation.objectives.losses import (
    SpectralReconstructionLoss,
    FrequencyBandLoss,
    MultiScaleLoss,
    PerceptualSpectralLoss,
)
from mesie.foundation.objectives.masked_spectral import (
    MaskedSpectralModeling,
    BandMasking,
    StructuredMasking,
    HierarchicalMasking,
)
from mesie.foundation.objectives.contrastive import (
    SpectralInfoNCE,
    BarlowTwins,
    VICReg,
    DINO,
)
from mesie.foundation.objectives.physics_informed import (
    PhysicsInformedLoss,
    ConservationLoss,
    CausalityLoss,
    SpectralConsistencyLoss,
)
from mesie.foundation.evaluation.metrics import (
    SpectralMetrics,
    ReconstructionMetrics,
    RepresentationMetrics,
    DownstreamMetrics,
)
from mesie.foundation.evaluation.probing import (
    LinearProbe,
    KNNProbe,
    MLPProbe,
)


class TestModalityProjector:
    """Tests for modality projector."""

    def test_output_shape(self):
        proj = ModalityProjector(input_dim=256, latent_dim=128)
        x = np.random.randn(4, 256)
        out = proj.forward(x)
        assert out.shape == (4, 128)

    def test_normalization(self):
        """Output should be L2 normalized."""
        proj = ModalityProjector(input_dim=256, latent_dim=128)
        x = np.random.randn(4, 256)
        out = proj.forward(x, training=False)
        norms = np.linalg.norm(out, axis=-1)
        assert np.allclose(norms, 1.0, atol=0.01)


class TestMomentumEncoder:
    """Tests for momentum encoder."""

    def test_initialization(self):
        me = MomentumEncoder(momentum=0.999)
        params = [np.random.randn(10, 10)]
        me.initialize(params)
        assert me.target_params is not None
        assert np.allclose(me.target_params[0], params[0])

    def test_ema_update(self):
        me = MomentumEncoder(momentum=0.99)
        params = [np.zeros((10, 10))]
        me.initialize(params)
        new_params = [np.ones((10, 10))]
        me.update(new_params)
        # Target should move toward new params
        assert np.mean(me.target_params[0]) > 0


class TestPrototypeLayer:
    """Tests for prototype layer."""

    def test_assignment(self):
        proto = PrototypeLayer(num_prototypes=100, latent_dim=64)
        z = np.random.randn(32, 64)
        soft, hard = proto.assign(z)
        assert soft.shape == (32, 100)
        assert hard.shape == (32,)
        assert np.allclose(np.sum(soft, axis=-1), 1.0, atol=1e-5)

    def test_sinkhorn(self):
        proto = PrototypeLayer(num_prototypes=100, latent_dim=64)
        z = np.random.randn(32, 64)
        balanced = proto.sinkhorn_assignments(z)
        assert balanced.shape == (32, 100)


class TestUniversalLatentSpace:
    """Tests for universal latent space."""

    def test_creation(self):
        space = UniversalSpectralLatentSpace()
        assert len(space.projectors) == 7

    def test_encode(self):
        space = UniversalSpectralLatentSpace(
            config=LatentSpaceConfig(latent_dim=128)
        )
        x = np.random.randn(4, 512)
        z = space.encode(x, "seismic")
        assert z.shape == (4, 128)

    def test_align_modalities(self):
        space = UniversalSpectralLatentSpace(
            config=LatentSpaceConfig(latent_dim=128)
        )
        x_a = np.random.randn(4, 512)
        x_b = np.random.randn(4, 256)
        result = space.align_modalities(x_a, "seismic", x_b, "vibration")
        assert "alignment_loss" in result
        assert result["alignment_loss"] >= 0

    def test_interpolation(self):
        space = UniversalSpectralLatentSpace(
            config=LatentSpaceConfig(latent_dim=128)
        )
        z_a = np.random.randn(128)
        z_b = np.random.randn(128)
        interp = space.interpolate(z_a, z_b, alpha=0.5)
        assert interp.shape == (128,)


class TestContrastiveAligner:
    """Tests for contrastive alignment."""

    def test_loss_computation(self):
        aligner = ContrastiveAligner(latent_dim=64)
        embeddings = {
            "seismic": np.random.randn(16, 64),
            "audio": np.random.randn(16, 64),
        }
        loss, metrics = aligner.compute_loss(embeddings)
        assert loss >= 0
        assert "total_alignment_loss" in metrics


class TestOptimalTransportAligner:
    """Tests for OT alignment."""

    def test_sinkhorn(self):
        aligner = OptimalTransportAligner(latent_dim=64)
        embeddings = {
            "seismic": np.random.randn(16, 64),
            "audio": np.random.randn(16, 64),
        }
        loss, metrics = aligner.compute_loss(embeddings)
        assert loss >= 0


class TestProjections:
    """Tests for projection heads."""

    def test_linear(self):
        proj = LinearProjection(128, 64)
        x = np.random.randn(4, 128)
        out = proj.forward(x)
        assert out.shape == (4, 64)

    def test_mlp(self):
        proj = MLPProjection(128, 64, hidden_dim=256, num_layers=3)
        x = np.random.randn(4, 128)
        out = proj.forward(x)
        assert out.shape == (4, 64)

    def test_gated(self):
        proj = GatedProjection(128, 64)
        x = np.random.randn(4, 128)
        out = proj.forward(x)
        assert out.shape == (4, 64)

    def test_adaptive(self):
        proj = ModalityAdaptiveProjection(128, 64, num_experts=4)
        x = np.random.randn(4, 128)
        out = proj.forward(x)
        assert out.shape == (4, 64)

    def test_factory(self):
        for ptype in ProjectionFactory.available():
            proj = ProjectionFactory.create(ptype, input_dim=64, output_dim=32)
            assert proj is not None


class TestSpectralReconstructionLoss:
    """Tests for reconstruction loss."""

    def test_zero_loss_for_identical(self):
        loss_fn = SpectralReconstructionLoss()
        signal = np.random.randn(4, 256)
        total, components = loss_fn.compute(signal, signal)
        assert total < 0.01

    def test_positive_loss_for_different(self):
        loss_fn = SpectralReconstructionLoss()
        pred = np.random.randn(4, 256)
        target = np.random.randn(4, 256)
        total, components = loss_fn.compute(pred, target)
        assert total > 0


class TestFrequencyBandLoss:
    """Tests for frequency band loss."""

    def test_computation(self):
        loss_fn = FrequencyBandLoss(num_bands=8, sample_rate=1000)
        pred = np.random.randn(4, 1024)
        target = np.random.randn(4, 1024)
        total, band_losses = loss_fn.compute(pred, target)
        assert total > 0
        assert "total" in band_losses


class TestMaskedSpectralModeling:
    """Tests for masked spectral modeling."""

    def test_mask_creation(self):
        msm = MaskedSpectralModeling(mask_ratio=0.15, strategy="random")
        mask = msm.create_mask((4, 128))
        assert mask.shape == (4, 128)
        # Should mask approximately 15% of tokens
        assert 0.05 < np.mean(mask) < 0.3

    def test_span_masking(self):
        msm = MaskedSpectralModeling(mask_ratio=0.15, strategy="span")
        mask = msm.create_mask((4, 128))
        assert mask.shape == (4, 128)

    def test_apply_mask(self):
        msm = MaskedSpectralModeling(mask_ratio=0.3)
        x = np.random.randn(4, 64, 32)
        mask = msm.create_mask((4, 64))
        masked_x, targets = msm.apply_mask(x, mask)
        assert masked_x.shape == x.shape
        assert targets.shape == x.shape

    def test_loss_computation(self):
        msm = MaskedSpectralModeling(mask_ratio=0.3)
        predictions = np.random.randn(4, 64, 32)
        targets = np.random.randn(4, 64, 32)
        mask = msm.create_mask((4, 64))
        loss, metrics = msm.compute_loss(predictions, targets, mask)
        assert loss > 0
        assert "masked_mse" in metrics


class TestContrastiveObjectives:
    """Tests for contrastive objectives."""

    def test_infonce(self):
        loss_fn = SpectralInfoNCE(temperature=0.07)
        anchor = np.random.randn(16, 64)
        positive = anchor + np.random.randn(16, 64) * 0.1
        loss, metrics = loss_fn.compute(anchor, positive)
        assert loss > 0
        assert "accuracy" in metrics

    def test_barlow_twins(self):
        loss_fn = BarlowTwins(latent_dim=64)
        z_a = np.random.randn(32, 64)
        z_b = z_a + np.random.randn(32, 64) * 0.1
        loss, metrics = loss_fn.compute(z_a, z_b)
        assert "on_diag_loss" in metrics

    def test_vicreg(self):
        loss_fn = VICReg(latent_dim=64)
        z_a = np.random.randn(32, 64)
        z_b = z_a + np.random.randn(32, 64) * 0.1
        loss, metrics = loss_fn.compute(z_a, z_b)
        assert "invariance_loss" in metrics
        assert "variance_loss" in metrics
        assert "covariance_loss" in metrics

    def test_dino(self):
        loss_fn = DINO(out_dim=128)
        student = np.random.randn(16, 128)
        teacher = np.random.randn(16, 128)
        loss, metrics = loss_fn.compute(student, teacher)
        assert loss > 0


class TestPhysicsInformedLoss:
    """Tests for physics-informed losses."""

    def test_conservation(self):
        cons = ConservationLoss()
        signal = np.random.randn(4, 256)
        loss = cons.parseval_loss(signal)
        assert loss >= 0

    def test_energy_conservation(self):
        cons = ConservationLoss()
        pred = np.random.randn(4, 256)
        target = pred * 1.1
        loss = cons.energy_conservation(pred, target)
        assert loss >= 0

    def test_combined_physics_loss(self):
        loss_fn = PhysicsInformedLoss()
        signal = np.random.randn(4, 256)
        total, components = loss_fn.compute(signal)
        assert "total" in components


class TestSpectralMetrics:
    """Tests for evaluation metrics."""

    def test_snr(self):
        metrics = SpectralMetrics()
        signal = np.sin(np.linspace(0, 10, 1000))
        noise = np.random.randn(1000) * 0.1
        snr = metrics.signal_to_noise_ratio(signal, noise)
        assert snr > 0

    def test_lsd(self):
        metrics = SpectralMetrics()
        ref = np.random.randn(1000)
        est = ref + np.random.randn(1000) * 0.1
        lsd = metrics.log_spectral_distance(ref, est)
        assert lsd >= 0

    def test_spectral_convergence(self):
        metrics = SpectralMetrics()
        ref = np.random.randn(1000)
        est = ref + np.random.randn(1000) * 0.01
        sc = metrics.spectral_convergence(ref, est)
        assert 0 <= sc <= 1


class TestRepresentationMetrics:
    """Tests for representation metrics."""

    def test_uniformity(self):
        metrics = RepresentationMetrics()
        z = np.random.randn(100, 64)
        u = metrics.uniformity(z)
        assert isinstance(u, float)

    def test_isotropy(self):
        metrics = RepresentationMetrics()
        z = np.random.randn(100, 64)
        iso = metrics.isotropy(z)
        assert 0 <= iso <= 1

    def test_intrinsic_dim(self):
        metrics = RepresentationMetrics()
        z = np.random.randn(100, 64)
        dim = metrics.intrinsic_dimensionality(z)
        assert dim > 0


class TestProbing:
    """Tests for probing tasks."""

    def test_linear_probe(self):
        probe = LinearProbe(input_dim=64, num_classes=5, max_epochs=10)
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 5, 100)
        history = probe.fit(X, y)
        assert len(history["train_acc"]) == 10
        preds = probe.predict(X)
        assert preds.shape == (100,)

    def test_knn_probe(self):
        probe = KNNProbe(k=5)
        X_train = np.random.randn(50, 64)
        y_train = np.random.randint(0, 3, 50)
        probe.fit(X_train, y_train)
        X_test = np.random.randn(10, 64)
        preds = probe.predict(X_test)
        assert preds.shape == (10,)

    def test_mlp_probe(self):
        probe = MLPProbe(input_dim=64, num_classes=5, max_epochs=5)
        X = np.random.randn(100, 64)
        y = np.random.randint(0, 5, 100)
        history = probe.fit(X, y)
        assert len(history["train_acc"]) == 5
