"""Diagram generation utilities."""

from __future__ import annotations


ARCHITECTURE_DIAGRAM = """
flowchart TD
    A[Input Spectral Records] --> B[Validation Layer]
    B --> C[Normalization + Interpolation]
    C --> D[Feature Extraction]
    D --> E[Electro-Spectral Feature Layer]
    D --> F[Node Topology Mapping]
    E --> G[Spectral Matcher]
    F --> G
    G --> H[Match Scores + Rankings]
    D --> I[Spectral Generator]
    I --> J[Single / RotDnn / PSD / FAS Outputs]
    D --> K[Spectral Embedding Encoder]
    K --> L[AI Retrieval + Cognitive Memory]
"""

EMBEDDING_PIPELINE = """
flowchart LR
    A[Frequency Grid] --> D[Structured Spectral Object]
    B[Amplitude Values] --> D
    C[Component Metadata] --> D
    D --> E[Feature Vector]
    E --> F[Embedding Encoder]
    F --> G[Spectral Embedding]
    G --> H[Search / Cluster / Memory / Agent State]
"""

COGNITIVE_INTEGRATION = """
flowchart TD
    A[Sensor / Signal / Simulation State] --> B[MESIE Spectral Object]
    B --> C[Spectral Embedding]
    C --> D[Memory Store]
    C --> E[Attention Weighting]
    C --> F[Anomaly Detection]
    C --> G[Agent State Model]
    D --> H[Cognitive Architecture]
    E --> H
    F --> H
    G --> H
"""
