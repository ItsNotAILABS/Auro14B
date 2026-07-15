"""Spectral Visualization and Reporting Engine.

Provides comprehensive visualization capabilities for spectral data,
including plot generation, interactive dashboards, report compilation,
and data export in multiple formats.

Key Components:
    - SpectralPlotter: Generate spectral plots and figures
    - DashboardGenerator: Create monitoring dashboards
    - ReportCompiler: Compile analysis reports
    - DataExporter: Export data in various formats
    - VisualizationConfig: Configuration for visual outputs
    - ColorMapper: Scientific color mapping for spectra
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class PlotType(Enum):
    """Types of spectral plots."""
    SPECTRUM = "spectrum"
    SPECTROGRAM = "spectrogram"
    WATERFALL = "waterfall"
    POLAR = "polar"
    HEATMAP = "heatmap"
    CONSTELLATION = "constellation"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    TIME_SERIES = "time_series"
    PHASE = "phase"
    COHERENCE = "coherence"
    TRANSFER_FUNCTION = "transfer_function"


class ColorMap(Enum):
    """Color maps for spectral visualization."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    SPECTRAL = "spectral"
    JET = "jet"
    THERMAL = "thermal"
    SCIENTIFIC = "scientific"


class ExportFormat(Enum):
    """Data export formats."""
    CSV = "csv"
    JSON = "json"
    NPY = "npy"
    HDF5 = "hdf5"
    MAT = "mat"
    PARQUET = "parquet"


class ReportSection(Enum):
    """Report section types."""
    SUMMARY = "summary"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    STATISTICAL = "statistical"
    ANOMALIES = "anomalies"
    TRENDS = "trends"
    RECOMMENDATIONS = "recommendations"
    RAW_DATA = "raw_data"


class AxisScale(Enum):
    """Axis scale types."""
    LINEAR = "linear"
    LOG = "log"
    DB = "db"
    OCTAVE = "octave"
    MEL = "mel"
    BARK = "bark"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class VisualizationConfig:
    """Configuration for spectral visualization.

    Args:
        width: Figure width in pixels.
        height: Figure height in pixels.
        dpi: Dots per inch.
        colormap: Color map name.
        x_scale: X-axis scale.
        y_scale: Y-axis scale.
        grid: Show grid.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        legend: Show legend.
        annotations: Enable annotations.
    """
    width: int = 800
    height: int = 600
    dpi: int = 100
    colormap: ColorMap = ColorMap.VIRIDIS
    x_scale: AxisScale = AxisScale.LINEAR
    y_scale: AxisScale = AxisScale.LINEAR
    grid: bool = True
    title: str = ""
    x_label: str = "Frequency"
    y_label: str = "Amplitude"
    legend: bool = True
    annotations: bool = True


@dataclass
class PlotData:
    """Data container for a single plot.

    Args:
        x: X-axis data.
        y: Y-axis data.
        z: Optional Z data (for 3D/heatmap).
        labels: Series labels.
        markers: Annotation markers.
        metadata: Additional plot info.
    """
    x: np.ndarray
    y: np.ndarray
    z: Optional[np.ndarray] = None
    labels: List[str] = field(default_factory=list)
    markers: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureSpec:
    """Specification for a rendered figure.

    Args:
        figure_id: Unique identifier.
        plot_type: Type of plot.
        data: Plot data.
        config: Visualization config.
        rendered_at: Render timestamp.
    """
    figure_id: str
    plot_type: PlotType
    data: PlotData
    config: VisualizationConfig = field(default_factory=VisualizationConfig)
    rendered_at: float = field(default_factory=time.time)


@dataclass
class ReportContent:
    """Content for a report section.

    Args:
        section: Section type.
        title: Section title.
        text: Section text content.
        figures: Associated figures.
        tables: Data tables.
        metrics: Key metrics.
    """
    section: ReportSection
    title: str
    text: str = ""
    figures: List[FigureSpec] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result of data export.

    Args:
        format: Export format used.
        size_bytes: Size of exported data.
        n_records: Number of data records.
        metadata: Export metadata.
    """
    format: ExportFormat
    size_bytes: int = 0
    n_records: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Color Mapper
# =============================================================================


class ColorMapper:
    """Scientific color mapping for spectral data.

    Maps data values to colors using perceptually uniform
    color maps suitable for scientific visualization.

    Args:
        colormap: Color map to use.
        vmin: Minimum value for mapping.
        vmax: Maximum value for mapping.
        n_levels: Number of discrete levels.
    """

    def __init__(
        self,
        colormap: ColorMap = ColorMap.VIRIDIS,
        vmin: float = 0.0,
        vmax: float = 1.0,
        n_levels: int = 256,
    ) -> None:
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax
        self.n_levels = n_levels
        self._lut = self._generate_lookup_table()

    def map_value(self, value: float) -> Tuple[int, int, int]:
        """Map a single value to RGB color.

        Args:
            value: Input value.

        Returns:
            Tuple of (R, G, B) in 0-255.
        """
        normalized = (value - self.vmin) / (self.vmax - self.vmin + 1e-12)
        normalized = np.clip(normalized, 0, 1)
        idx = int(normalized * (self.n_levels - 1))
        return self._lut[idx]

    def map_array(self, values: np.ndarray) -> np.ndarray:
        """Map array of values to RGB colors.

        Args:
            values: Input values.

        Returns:
            Array of shape (n, 3) with RGB values.
        """
        values = np.atleast_1d(values).flatten()
        normalized = (values - self.vmin) / (self.vmax - self.vmin + 1e-12)
        normalized = np.clip(normalized, 0, 1)
        indices = (normalized * (self.n_levels - 1)).astype(int)

        colors = np.zeros((len(values), 3), dtype=int)
        for i, idx in enumerate(indices):
            colors[i] = self._lut[idx]

        return colors

    def _generate_lookup_table(self) -> List[Tuple[int, int, int]]:
        """Generate color lookup table."""
        lut = []
        for i in range(self.n_levels):
            t = i / (self.n_levels - 1)

            if self.colormap == ColorMap.VIRIDIS:
                r, g, b = self._viridis(t)
            elif self.colormap == ColorMap.PLASMA:
                r, g, b = self._plasma(t)
            elif self.colormap == ColorMap.INFERNO:
                r, g, b = self._inferno(t)
            elif self.colormap == ColorMap.THERMAL:
                r, g, b = self._thermal(t)
            elif self.colormap == ColorMap.SPECTRAL:
                r, g, b = self._spectral(t)
            else:
                r, g, b = self._viridis(t)

            lut.append((int(r * 255), int(g * 255), int(b * 255)))

        return lut

    def _viridis(self, t: float) -> Tuple[float, float, float]:
        """Viridis-like colormap."""
        r = 0.267004 + t * (0.993248 - 0.267004)
        g = 0.004874 + t * (0.906157 - 0.004874)
        b = 0.329415 + t * (0.143936 - 0.329415)
        return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))

    def _plasma(self, t: float) -> Tuple[float, float, float]:
        """Plasma-like colormap."""
        r = 0.050383 + t * 0.9
        g = 0.029803 + t * (0.8 - 0.029803) * (1 - abs(2*t - 1))
        b = 0.529975 - t * 0.5
        return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))

    def _inferno(self, t: float) -> Tuple[float, float, float]:
        """Inferno-like colormap."""
        r = min(1.0, t * 3)
        g = max(0.0, min(1.0, (t - 0.3) * 2))
        b = max(0.0, min(1.0, 0.5 - abs(t - 0.5)))
        return (r, g, b)

    def _thermal(self, t: float) -> Tuple[float, float, float]:
        """Thermal colormap (blue-red)."""
        r = t
        g = 0.0
        b = 1.0 - t
        return (r, g, b)

    def _spectral(self, t: float) -> Tuple[float, float, float]:
        """Spectral rainbow colormap."""
        # Approximate rainbow: violet -> blue -> green -> yellow -> red
        if t < 0.2:
            r, g, b = 0.5, 0.0, 1.0
        elif t < 0.4:
            r, g, b = 0.0, (t - 0.2) * 5, 1.0
        elif t < 0.6:
            r, g, b = 0.0, 1.0, 1.0 - (t - 0.4) * 5
        elif t < 0.8:
            r, g, b = (t - 0.6) * 5, 1.0, 0.0
        else:
            r, g, b = 1.0, 1.0 - (t - 0.8) * 5, 0.0
        return (np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))

    @property
    def range(self) -> Tuple[float, float]:
        """Value range."""
        return (self.vmin, self.vmax)


# =============================================================================
# Spectral Plotter
# =============================================================================


class SpectralPlotter:
    """Generate spectral plot specifications.

    Creates plot data and configurations for spectral visualization.
    Does not directly render (no matplotlib dependency), but produces
    structured data ready for rendering.

    Args:
        default_config: Default visualization configuration.
    """

    def __init__(
        self,
        default_config: Optional[VisualizationConfig] = None,
    ) -> None:
        self.default_config = default_config or VisualizationConfig()
        self._figure_count: int = 0
        self._color_mapper = ColorMapper()

    def plot_spectrum(
        self,
        spectrum: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        title: str = "Spectrum",
        config: Optional[VisualizationConfig] = None,
    ) -> FigureSpec:
        """Create a spectrum plot specification.

        Args:
            spectrum: Spectral data.
            frequencies: Frequency axis (optional).
            title: Plot title.
            config: Optional custom config.

        Returns:
            FigureSpec ready for rendering.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        n = len(spectrum)

        if frequencies is None:
            frequencies = np.arange(n, dtype=float)

        cfg = config or self.default_config
        cfg_copy = VisualizationConfig(
            width=cfg.width, height=cfg.height, dpi=cfg.dpi,
            colormap=cfg.colormap, x_scale=cfg.x_scale, y_scale=cfg.y_scale,
            grid=cfg.grid, title=title,
            x_label=cfg.x_label, y_label=cfg.y_label,
        )

        # Detect and annotate peaks
        markers = []
        if cfg.annotations:
            peaks = self._find_peaks(spectrum)
            for idx, amp in peaks[:5]:
                markers.append({
                    "x": float(frequencies[idx]),
                    "y": float(amp),
                    "label": f"Peak @ {frequencies[idx]:.1f}",
                })

        self._figure_count += 1
        return FigureSpec(
            figure_id=f"fig_{self._figure_count}",
            plot_type=PlotType.SPECTRUM,
            data=PlotData(
                x=frequencies[:n],
                y=spectrum,
                markers=markers,
                metadata={"n_points": n, "y_range": (float(np.min(spectrum)), float(np.max(spectrum)))},
            ),
            config=cfg_copy,
        )

    def plot_spectrogram(
        self,
        spectra: np.ndarray,
        title: str = "Spectrogram",
        config: Optional[VisualizationConfig] = None,
    ) -> FigureSpec:
        """Create a spectrogram plot specification.

        Args:
            spectra: 2D array (time x frequency).
            title: Plot title.
            config: Optional custom config.

        Returns:
            FigureSpec for spectrogram.
        """
        spectra = np.atleast_2d(spectra)
        n_time, n_freq = spectra.shape

        cfg = config or self.default_config

        self._figure_count += 1
        return FigureSpec(
            figure_id=f"fig_{self._figure_count}",
            plot_type=PlotType.SPECTROGRAM,
            data=PlotData(
                x=np.arange(n_freq, dtype=float),
                y=np.arange(n_time, dtype=float),
                z=spectra,
                metadata={"n_time": n_time, "n_freq": n_freq},
            ),
            config=VisualizationConfig(
                title=title, colormap=cfg.colormap,
                x_label="Frequency", y_label="Time",
            ),
        )

    def plot_waterfall(
        self,
        spectra: List[np.ndarray],
        title: str = "Waterfall",
    ) -> FigureSpec:
        """Create a waterfall plot specification.

        Args:
            spectra: List of spectra at different times.
            title: Plot title.

        Returns:
            FigureSpec for waterfall.
        """
        max_len = max(len(s) for s in spectra)
        waterfall_data = np.zeros((len(spectra), max_len))
        for i, s in enumerate(spectra):
            waterfall_data[i, :len(s)] = s

        self._figure_count += 1
        return FigureSpec(
            figure_id=f"fig_{self._figure_count}",
            plot_type=PlotType.WATERFALL,
            data=PlotData(
                x=np.arange(max_len, dtype=float),
                y=np.arange(len(spectra), dtype=float),
                z=waterfall_data,
                metadata={"n_spectra": len(spectra), "max_freq_bins": max_len},
            ),
            config=VisualizationConfig(title=title),
        )

    def plot_comparison(
        self,
        spectra: List[np.ndarray],
        labels: List[str],
        title: str = "Comparison",
    ) -> FigureSpec:
        """Create a multi-spectrum comparison plot.

        Args:
            spectra: List of spectra to compare.
            labels: Labels for each spectrum.
            title: Plot title.

        Returns:
            FigureSpec for comparison.
        """
        max_len = max(len(s) for s in spectra)
        combined = np.zeros((len(spectra), max_len))
        for i, s in enumerate(spectra):
            combined[i, :len(s)] = s

        self._figure_count += 1
        return FigureSpec(
            figure_id=f"fig_{self._figure_count}",
            plot_type=PlotType.SPECTRUM,
            data=PlotData(
                x=np.arange(max_len, dtype=float),
                y=combined[0] if len(spectra) > 0 else np.array([]),
                z=combined if len(spectra) > 1 else None,
                labels=labels,
                metadata={"n_spectra": len(spectra)},
            ),
            config=VisualizationConfig(title=title, legend=True),
        )

    def plot_phase(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray,
        title: str = "Phase Plot",
    ) -> FigureSpec:
        """Create a magnitude/phase plot.

        Args:
            magnitude: Magnitude spectrum.
            phase: Phase spectrum.
            title: Plot title.

        Returns:
            FigureSpec for phase plot.
        """
        self._figure_count += 1
        return FigureSpec(
            figure_id=f"fig_{self._figure_count}",
            plot_type=PlotType.PHASE,
            data=PlotData(
                x=np.arange(len(magnitude), dtype=float),
                y=magnitude,
                z=phase.reshape(1, -1) if phase is not None else None,
                labels=["Magnitude", "Phase"],
                metadata={"has_phase": True},
            ),
            config=VisualizationConfig(title=title),
        )

    def _find_peaks(self, spectrum: np.ndarray) -> List[Tuple[int, float]]:
        """Find significant peaks in spectrum."""
        n = len(spectrum)
        peaks = []
        threshold = np.mean(spectrum) + np.std(spectrum)

        for i in range(1, n - 1):
            if (spectrum[i] > spectrum[i-1] and
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > threshold):
                peaks.append((i, float(spectrum[i])))

        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks

    @property
    def figure_count(self) -> int:
        """Total figures created."""
        return self._figure_count


# =============================================================================
# Dashboard Generator
# =============================================================================


class DashboardGenerator:
    """Generate monitoring dashboard layouts.

    Creates structured dashboard specifications with
    multiple panels, real-time indicators, and alerts.

    Args:
        name: Dashboard name.
        n_columns: Number of layout columns.
        refresh_rate: Refresh rate in seconds.
    """

    def __init__(
        self,
        name: str = "Spectral Monitor",
        n_columns: int = 3,
        refresh_rate: float = 1.0,
    ) -> None:
        self.name = name
        self.n_columns = n_columns
        self.refresh_rate = refresh_rate
        self._panels: List[Dict[str, Any]] = []
        self._indicators: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []

    def add_panel(
        self,
        title: str,
        plot_type: PlotType,
        data_source: str,
        position: Optional[Tuple[int, int]] = None,
        size: Tuple[int, int] = (1, 1),
    ) -> None:
        """Add a visualization panel to the dashboard.

        Args:
            title: Panel title.
            plot_type: Type of visualization.
            data_source: Data source identifier.
            position: Grid position (row, col).
            size: Panel size (rows, cols).
        """
        panel = {
            "title": title,
            "plot_type": plot_type.value,
            "data_source": data_source,
            "position": position or (len(self._panels) // self.n_columns, len(self._panels) % self.n_columns),
            "size": size,
        }
        self._panels.append(panel)

    def add_indicator(
        self,
        name: str,
        value: float,
        unit: str = "",
        threshold: Optional[float] = None,
        trend: str = "stable",
    ) -> None:
        """Add a key indicator to the dashboard.

        Args:
            name: Indicator name.
            value: Current value.
            unit: Value unit.
            threshold: Alert threshold.
            trend: Current trend ('up', 'down', 'stable').
        """
        indicator = {
            "name": name,
            "value": value,
            "unit": unit,
            "threshold": threshold,
            "trend": trend,
            "status": "normal" if threshold is None or value < threshold else "alert",
        }
        self._indicators.append(indicator)

    def add_alert(
        self,
        message: str,
        severity: str = "info",
        source: str = "",
    ) -> None:
        """Add an alert to the dashboard.

        Args:
            message: Alert message.
            severity: 'info', 'warning', 'critical'.
            source: Alert source.
        """
        self._alerts.append({
            "message": message,
            "severity": severity,
            "source": source,
            "timestamp": time.time(),
        })

    def generate(self) -> Dict[str, Any]:
        """Generate the complete dashboard specification.

        Returns:
            Dictionary with full dashboard layout and data.
        """
        return {
            "name": self.name,
            "n_columns": self.n_columns,
            "refresh_rate": self.refresh_rate,
            "panels": self._panels,
            "indicators": self._indicators,
            "alerts": self._alerts,
            "generated_at": time.time(),
            "n_panels": len(self._panels),
            "n_indicators": len(self._indicators),
            "n_alerts": len(self._alerts),
        }

    @property
    def n_panels(self) -> int:
        """Number of panels."""
        return len(self._panels)


# =============================================================================
# Report Compiler
# =============================================================================


class ReportCompiler:
    """Compile comprehensive spectral analysis reports.

    Assembles analysis results, visualizations, and metrics
    into structured report documents.

    Args:
        title: Report title.
        author: Report author.
    """

    def __init__(
        self,
        title: str = "Spectral Analysis Report",
        author: str = "MESIE System",
    ) -> None:
        self.title = title
        self.author = author
        self._sections: List[ReportContent] = []
        self._metadata: Dict[str, Any] = {
            "title": title,
            "author": author,
            "created_at": time.time(),
        }

    def add_section(self, content: ReportContent) -> None:
        """Add a section to the report.

        Args:
            content: Section content.
        """
        self._sections.append(content)

    def add_summary(
        self,
        text: str,
        key_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add an executive summary section.

        Args:
            text: Summary text.
            key_metrics: Key metrics to highlight.
        """
        self._sections.insert(0, ReportContent(
            section=ReportSection.SUMMARY,
            title="Executive Summary",
            text=text,
            metrics=key_metrics or {},
        ))

    def add_spectral_analysis(
        self,
        spectrum: np.ndarray,
        features: Dict[str, float],
        figures: Optional[List[FigureSpec]] = None,
    ) -> None:
        """Add spectral analysis section.

        Args:
            spectrum: Analyzed spectrum.
            features: Computed features.
            figures: Associated visualizations.
        """
        text = f"Spectral analysis of {len(spectrum)} frequency bins.\n"
        text += f"Key features: centroid={features.get('centroid', 0):.2f}, "
        text += f"bandwidth={features.get('bandwidth', 0):.2f}, "
        text += f"flatness={features.get('flatness', 0):.4f}"

        self._sections.append(ReportContent(
            section=ReportSection.SPECTRAL_ANALYSIS,
            title="Spectral Analysis",
            text=text,
            figures=figures or [],
            metrics=features,
        ))

    def add_statistical_section(
        self,
        data: np.ndarray,
        description: str = "",
    ) -> None:
        """Add statistical analysis section.

        Args:
            data: Data to analyze.
            description: Section description.
        """
        data = np.atleast_1d(data).flatten()
        stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "skewness": float(self._skewness(data)),
            "kurtosis": float(self._kurtosis(data)),
        }

        self._sections.append(ReportContent(
            section=ReportSection.STATISTICAL,
            title="Statistical Analysis",
            text=description or f"Statistical analysis of {len(data)} data points.",
            metrics=stats,
        ))

    def add_anomaly_section(
        self,
        anomalies: List[Dict[str, Any]],
    ) -> None:
        """Add anomaly report section.

        Args:
            anomalies: List of detected anomalies.
        """
        text = f"Detected {len(anomalies)} anomalies.\n"
        for i, anom in enumerate(anomalies[:10]):
            text += f"  {i+1}. {anom.get('type', 'unknown')}: {anom.get('description', '')}\n"

        self._sections.append(ReportContent(
            section=ReportSection.ANOMALIES,
            title="Anomaly Detection",
            text=text,
            metrics={"n_anomalies": len(anomalies)},
        ))

    def compile(self) -> Dict[str, Any]:
        """Compile the full report.

        Returns:
            Complete report as structured dictionary.
        """
        report = {
            "metadata": self._metadata,
            "title": self.title,
            "author": self.author,
            "compiled_at": time.time(),
            "n_sections": len(self._sections),
            "sections": [],
        }

        for section in self._sections:
            report["sections"].append({
                "type": section.section.value,
                "title": section.title,
                "text": section.text,
                "n_figures": len(section.figures),
                "metrics": section.metrics,
                "tables": section.tables,
            })

        return report

    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-12:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-12:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3.0)

    @property
    def n_sections(self) -> int:
        """Number of report sections."""
        return len(self._sections)


# =============================================================================
# Data Exporter
# =============================================================================


class DataExporter:
    """Export spectral data in multiple formats.

    Serializes spectral data and analysis results for
    external use and archival.

    Args:
        default_format: Default export format.
        compression: Enable compression.
    """

    def __init__(
        self,
        default_format: ExportFormat = ExportFormat.CSV,
        compression: bool = False,
    ) -> None:
        self.default_format = default_format
        self.compression = compression
        self._export_count: int = 0

    def export_spectrum(
        self,
        spectrum: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        format: Optional[ExportFormat] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        """Export a spectrum.

        Args:
            spectrum: Spectral data.
            frequencies: Frequency axis.
            format: Export format.
            metadata: Additional metadata.

        Returns:
            ExportResult with export info.
        """
        spectrum = np.atleast_1d(spectrum).flatten()
        fmt = format or self.default_format
        self._export_count += 1

        if frequencies is None:
            frequencies = np.arange(len(spectrum), dtype=float)

        # Compute export size estimate
        n = len(spectrum)
        if fmt == ExportFormat.CSV:
            # Estimate: ~20 chars per row (freq,value\n)
            size = n * 20
        elif fmt == ExportFormat.JSON:
            size = n * 30
        elif fmt == ExportFormat.NPY:
            size = n * 8 * 2  # Two float64 arrays
        else:
            size = n * 8 * 2

        return ExportResult(
            format=fmt,
            size_bytes=size,
            n_records=n,
            metadata={
                "shape": (n,),
                "dtype": "float64",
                "has_frequencies": True,
                **(metadata or {}),
            },
        )

    def export_batch(
        self,
        spectra: List[np.ndarray],
        format: Optional[ExportFormat] = None,
    ) -> ExportResult:
        """Export multiple spectra.

        Args:
            spectra: List of spectra.
            format: Export format.

        Returns:
            ExportResult.
        """
        fmt = format or self.default_format
        self._export_count += 1
        total_points = sum(len(s) for s in spectra)

        return ExportResult(
            format=fmt,
            size_bytes=total_points * 8,
            n_records=len(spectra),
            metadata={
                "n_spectra": len(spectra),
                "total_points": total_points,
                "mean_length": total_points / max(1, len(spectra)),
            },
        )

    def export_report(
        self,
        report: Dict[str, Any],
        format: Optional[ExportFormat] = None,
    ) -> ExportResult:
        """Export a compiled report.

        Args:
            report: Report dictionary.
            format: Export format.

        Returns:
            ExportResult.
        """
        fmt = format or ExportFormat.JSON
        self._export_count += 1

        # Estimate JSON size
        import json
        try:
            serializable = {k: v for k, v in report.items() if not isinstance(v, np.ndarray)}
            size = len(json.dumps(serializable, default=str))
        except (TypeError, ValueError):
            size = 1000

        return ExportResult(
            format=fmt,
            size_bytes=size,
            n_records=report.get("n_sections", 0),
            metadata={"report_title": report.get("title", "Unknown")},
        )

    @property
    def export_count(self) -> int:
        """Total exports performed."""
        return self._export_count


# =============================================================================
# Spectral Statistics Engine
# =============================================================================


class SpectralStatisticsEngine:
    """Comprehensive statistical analysis for spectral data.

    Provides descriptive statistics, hypothesis testing,
    correlation analysis, and distribution fitting.

    Args:
        confidence_level: Default confidence level.
        n_bootstrap: Bootstrap sample count.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> None:
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap

    def descriptive_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive descriptive statistics.

        Args:
            data: Input data.

        Returns:
            Dictionary of statistics.
        """
        data = np.atleast_1d(data).flatten()
        n = len(data)
        if n == 0:
            return {}

        stats = {
            "n": float(n),
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1) if n > 1 else 0),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.ptp(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
            "variance": float(np.var(data, ddof=1) if n > 1 else 0),
            "cv": float(np.std(data) / (np.abs(np.mean(data)) + 1e-12)),
            "rms": float(np.sqrt(np.mean(data ** 2))),
        }

        # Higher moments
        if n > 2:
            mean = np.mean(data)
            std = np.std(data) + 1e-12
            centered = (data - mean) / std
            stats["skewness"] = float(np.mean(centered ** 3))
            stats["kurtosis"] = float(np.mean(centered ** 4) - 3.0)
        else:
            stats["skewness"] = 0.0
            stats["kurtosis"] = 0.0

        return stats

    def correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix for multi-channel data.

        Args:
            data: Data matrix (n_samples x n_channels).

        Returns:
            Correlation matrix.
        """
        data = np.atleast_2d(data)
        n_channels = data.shape[1] if data.ndim > 1 else 1
        if n_channels == 1:
            return np.array([[1.0]])

        return np.corrcoef(data.T)

    def spectral_correlation(
        self,
        spec1: np.ndarray,
        spec2: np.ndarray,
    ) -> Dict[str, float]:
        """Compute correlation between two spectra.

        Args:
            spec1: First spectrum.
            spec2: Second spectrum.

        Returns:
            Correlation metrics.
        """
        n = min(len(spec1), len(spec2))
        s1 = spec1[:n]
        s2 = spec2[:n]

        # Pearson correlation
        pearson = float(np.corrcoef(s1, s2)[0, 1]) if np.std(s1) > 1e-12 and np.std(s2) > 1e-12 else 0.0

        # Spectral angle
        dot = np.dot(s1, s2)
        norms = np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-12
        cos_angle = np.clip(dot / norms, -1, 1)
        spectral_angle = float(np.arccos(cos_angle))

        # RMS difference
        rms_diff = float(np.sqrt(np.mean((s1 - s2) ** 2)))

        return {
            "pearson": pearson,
            "spectral_angle_rad": spectral_angle,
            "rms_difference": rms_diff,
            "max_difference": float(np.max(np.abs(s1 - s2))),
            "cosine_similarity": float((cos_angle + 1) / 2),
        }

    def bootstrap_confidence(
        self,
        data: np.ndarray,
        statistic: str = "mean",
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval.

        Args:
            data: Input data.
            statistic: Statistic to bootstrap ('mean', 'median', 'std').

        Returns:
            Tuple of (estimate, lower_bound, upper_bound).
        """
        data = np.atleast_1d(data).flatten()
        n = len(data)

        stat_fn = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
        }.get(statistic, np.mean)

        # Bootstrap
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(float(stat_fn(sample)))

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1.0 - self.confidence_level
        lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
        estimate = float(stat_fn(data))

        return estimate, lower, upper

    def detect_outliers(
        self,
        data: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> np.ndarray:
        """Detect outliers in spectral data.

        Args:
            data: Input data.
            method: Detection method ('iqr', 'zscore', 'mad').
            threshold: Outlier threshold.

        Returns:
            Boolean mask (True = outlier).
        """
        data = np.atleast_1d(data).flatten()

        if method == "iqr":
            q25 = np.percentile(data, 25)
            q75 = np.percentile(data, 75)
            iqr = q75 - q25
            lower = q25 - threshold * iqr
            upper = q75 + threshold * iqr
            return (data < lower) | (data > upper)

        elif method == "zscore":
            z = np.abs((data - np.mean(data)) / (np.std(data) + 1e-12))
            return z > threshold

        elif method == "mad":
            median = np.median(data)
            mad = np.median(np.abs(data - median)) + 1e-12
            modified_z = 0.6745 * (data - median) / mad
            return np.abs(modified_z) > threshold

        return np.zeros(len(data), dtype=bool)

    def trend_analysis(
        self,
        time_series: np.ndarray,
        method: str = "linear",
    ) -> Dict[str, float]:
        """Analyze trend in spectral time series.

        Args:
            time_series: Sequential measurements.
            method: Trend analysis method.

        Returns:
            Trend metrics.
        """
        time_series = np.atleast_1d(time_series).flatten()
        n = len(time_series)
        if n < 3:
            return {"slope": 0.0, "r_squared": 0.0}

        t = np.arange(n, dtype=float)

        # Linear regression
        coeffs = np.polyfit(t, time_series, 1)
        predicted = np.polyval(coeffs, t)
        residuals = time_series - predicted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((time_series - np.mean(time_series)) ** 2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

        # Change point detection (simple)
        mid = n // 2
        mean_first = np.mean(time_series[:mid])
        mean_second = np.mean(time_series[mid:])
        change_magnitude = abs(mean_second - mean_first)

        return {
            "slope": float(coeffs[0]),
            "intercept": float(coeffs[1]),
            "r_squared": float(r_squared),
            "residual_std": float(np.std(residuals)),
            "change_magnitude": float(change_magnitude),
            "is_increasing": bool(coeffs[0] > 0),
            "trend_strength": float(abs(r_squared * coeffs[0])),
        }
