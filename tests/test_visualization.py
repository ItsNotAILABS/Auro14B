"""Tests for visualization, reporting, and statistics modules."""

import numpy as np
import pytest

from mesie.cognitive.visualization import (
    ColorMapper,
    ColorMap,
    DashboardGenerator,
    DataExporter,
    ExportFormat,
    PlotType,
    ReportCompiler,
    ReportSection,
    SpectralPlotter,
    SpectralStatisticsEngine,
    VisualizationConfig,
)


class TestColorMapper:
    def test_map_value(self):
        cm = ColorMapper(vmin=0, vmax=1)
        r, g, b = cm.map_value(0.5)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_map_array(self):
        cm = ColorMapper(vmin=0, vmax=10)
        values = np.linspace(0, 10, 20)
        colors = cm.map_array(values)
        assert colors.shape == (20, 3)
        assert np.all(colors >= 0)
        assert np.all(colors <= 255)

    def test_different_colormaps(self):
        for cmap in [ColorMap.VIRIDIS, ColorMap.PLASMA, ColorMap.INFERNO, ColorMap.THERMAL, ColorMap.SPECTRAL]:
            cm = ColorMapper(colormap=cmap)
            r, g, b = cm.map_value(0.5)
            assert 0 <= r <= 255


class TestSpectralPlotter:
    def test_plot_spectrum(self):
        plotter = SpectralPlotter()
        spectrum = np.abs(np.random.randn(256))
        fig = plotter.plot_spectrum(spectrum, title="Test")
        assert fig.plot_type == PlotType.SPECTRUM
        assert len(fig.data.y) == 256
        assert fig.figure_id.startswith("fig_")

    def test_plot_spectrogram(self):
        plotter = SpectralPlotter()
        spectra = np.abs(np.random.randn(10, 64))
        fig = plotter.plot_spectrogram(spectra)
        assert fig.plot_type == PlotType.SPECTROGRAM
        assert fig.data.z is not None

    def test_plot_waterfall(self):
        plotter = SpectralPlotter()
        spectra = [np.random.randn(128) for _ in range(5)]
        fig = plotter.plot_waterfall(spectra)
        assert fig.plot_type == PlotType.WATERFALL

    def test_plot_comparison(self):
        plotter = SpectralPlotter()
        spectra = [np.random.randn(100), np.random.randn(100)]
        fig = plotter.plot_comparison(spectra, labels=["A", "B"])
        assert len(fig.data.labels) == 2

    def test_figure_count(self):
        plotter = SpectralPlotter()
        plotter.plot_spectrum(np.ones(10))
        plotter.plot_spectrum(np.ones(10))
        assert plotter.figure_count == 2


class TestDashboardGenerator:
    def test_add_panel(self):
        dash = DashboardGenerator()
        dash.add_panel("Spectrum", PlotType.SPECTRUM, "sensor_1")
        assert dash.n_panels == 1

    def test_add_indicator(self):
        dash = DashboardGenerator()
        dash.add_indicator("Temperature", 25.3, "°C", threshold=50.0)
        spec = dash.generate()
        assert spec["n_indicators"] == 1
        assert spec["indicators"][0]["status"] == "normal"

    def test_add_alert(self):
        dash = DashboardGenerator()
        dash.add_alert("High vibration detected", severity="warning")
        spec = dash.generate()
        assert spec["n_alerts"] == 1

    def test_generate_full(self):
        dash = DashboardGenerator(name="Test Dashboard", n_columns=2)
        dash.add_panel("FFT", PlotType.SPECTRUM, "fft_data")
        dash.add_panel("Trend", PlotType.TIME_SERIES, "trend_data")
        dash.add_indicator("SNR", 45.0, "dB")
        spec = dash.generate()
        assert spec["name"] == "Test Dashboard"
        assert spec["n_panels"] == 2


class TestReportCompiler:
    def test_add_summary(self):
        report = ReportCompiler(title="Test Report")
        report.add_summary("All systems nominal.", {"score": 0.95})
        assert report.n_sections == 1

    def test_add_spectral_analysis(self):
        report = ReportCompiler()
        spectrum = np.random.randn(128)
        features = {"centroid": 64.0, "bandwidth": 20.0, "flatness": 0.5}
        report.add_spectral_analysis(spectrum, features)
        assert report.n_sections == 1

    def test_add_statistical(self):
        report = ReportCompiler()
        data = np.random.randn(1000)
        report.add_statistical_section(data)
        compiled = report.compile()
        assert compiled["n_sections"] == 1
        assert "mean" in compiled["sections"][0]["metrics"]

    def test_compile_full(self):
        report = ReportCompiler(title="Full Report")
        report.add_summary("Summary text", {"quality": 0.9})
        report.add_spectral_analysis(np.ones(64), {"centroid": 32.0})
        report.add_anomaly_section([{"type": "peak_shift", "description": "Peak at 100Hz shifted"}])
        compiled = report.compile()
        assert compiled["title"] == "Full Report"
        assert compiled["n_sections"] == 3


class TestDataExporter:
    def test_export_spectrum(self):
        exp = DataExporter()
        spectrum = np.random.randn(256)
        result = exp.export_spectrum(spectrum)
        assert result.n_records == 256
        assert result.size_bytes > 0

    def test_export_batch(self):
        exp = DataExporter()
        spectra = [np.random.randn(128) for _ in range(10)]
        result = exp.export_batch(spectra)
        assert result.n_records == 10

    def test_export_report(self):
        exp = DataExporter()
        report = {"title": "Test", "n_sections": 2, "sections": []}
        result = exp.export_report(report)
        assert result.format == ExportFormat.JSON

    def test_export_count(self):
        exp = DataExporter()
        exp.export_spectrum(np.ones(10))
        exp.export_spectrum(np.ones(10))
        assert exp.export_count == 2


class TestSpectralStatisticsEngine:
    def test_descriptive_stats(self):
        engine = SpectralStatisticsEngine()
        data = np.random.randn(1000)
        stats = engine.descriptive_stats(data)
        assert "mean" in stats
        assert "std" in stats
        assert "skewness" in stats
        assert abs(stats["mean"]) < 0.2  # Should be near 0 for normal

    def test_correlation_matrix(self):
        engine = SpectralStatisticsEngine()
        data = np.random.randn(100, 4)
        corr = engine.correlation_matrix(data)
        assert corr.shape == (4, 4)
        assert np.allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_spectral_correlation(self):
        engine = SpectralStatisticsEngine()
        spec1 = np.sin(np.linspace(0, 10, 128))
        spec2 = np.sin(np.linspace(0, 10, 128)) * 1.1  # Scaled version
        result = engine.spectral_correlation(spec1, spec2)
        assert result["pearson"] > 0.99
        assert result["cosine_similarity"] > 0.99

    def test_bootstrap_confidence(self):
        engine = SpectralStatisticsEngine(n_bootstrap=500)
        data = np.random.randn(100) + 5.0
        estimate, lower, upper = engine.bootstrap_confidence(data, "mean")
        assert lower < estimate < upper
        assert 4.5 < estimate < 5.5

    def test_detect_outliers_iqr(self):
        engine = SpectralStatisticsEngine()
        data = np.random.randn(100)
        data[0] = 100  # Outlier
        mask = engine.detect_outliers(data, method="iqr")
        assert mask[0] == True

    def test_detect_outliers_zscore(self):
        engine = SpectralStatisticsEngine()
        data = np.random.randn(100)
        data[-1] = 50  # Outlier
        mask = engine.detect_outliers(data, method="zscore", threshold=3.0)
        assert mask[-1] == True

    def test_trend_analysis(self):
        engine = SpectralStatisticsEngine()
        # Create increasing trend
        t = np.arange(100, dtype=float)
        data = 2 * t + 10 + np.random.randn(100) * 0.5
        result = engine.trend_analysis(data)
        assert result["slope"] > 1.5
        assert result["r_squared"] > 0.9
        assert result["is_increasing"] == True

    def test_trend_analysis_flat(self):
        engine = SpectralStatisticsEngine()
        data = np.ones(50) + 0.01 * np.random.randn(50)
        result = engine.trend_analysis(data)
        assert abs(result["slope"]) < 0.1
