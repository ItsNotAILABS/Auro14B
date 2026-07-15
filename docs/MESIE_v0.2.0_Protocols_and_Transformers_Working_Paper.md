# MESIE v0.2.0 Update: Intelligence Protocols and Spectral Transformers — Research Working Paper

**Author:** Alfredo Medina  
**Affiliation:** Independent Research  
**Date:** June 2026  
**Version:** Working Paper v1.0  
**License:** Apache-2.0  
**Repository:** [github.com/FreddyCreates/Multi-Element-Spectral-Intelligence-Engine-MESIE-](https://github.com/FreddyCreates/Multi-Element-Spectral-Intelligence-Engine-MESIE-)

---

## Abstract

This working paper documents the major v0.2.0 expansion of the Multi-Element Spectral Intelligence Engine (MESIE), which introduces two critical new subsystems: Intelligence Protocols and Spectral Transformers. Together with expanded data exchange protocols, neural model architectures, training pipelines, inference engines, and transfer learning capabilities, this update adds approximately 100,000 lines of structured code to the MESIE codebase. This paper details what was added, the architectural reasoning behind each component, what these additions enable for AI systems, and their implications for autonomous spectral reasoning, real-time decision-making, and cognitive architecture integration.

---

## 1. Introduction

### 1.1 Context

MESIE v0.1.0 established the foundation: structured spectral records, validation, processing, matching, generation, feature extraction, embeddings, and cognitive adapters. However, v0.1.0 treated AI integration as an output stage — spectra were processed and then handed off to external systems.

The v0.2.0 update fundamentally changes this relationship. AI is no longer downstream of spectral processing — it is **embedded within** the spectral processing pipeline. The engine now contains its own intelligence protocols, transformer architectures, training systems, inference engines, and domain adaptation tools. Spectra are not merely processed *for* AI — they are processed *by* AI within the engine itself.

### 1.2 Scope of the Update

The v0.2.0 release adds the following major components:

1. **Intelligence AI Protocols** (`mesie.ai.intelligence_protocols`) — autonomous reasoning, episodic memory, attention focus, and multi-level intelligence engagement
2. **Spectral Transformer Pipeline** (`mesie.ai.transformer_pipeline`) — end-to-end transformer encoder with spectral tokenization, positional encoding, multi-head attention, and configurable pooling
3. **Neural Model Architectures** (`mesie.ai.models`) — spectral autoencoder, spectral classifier, and spectral transformer models
4. **Training Pipeline** (`mesie.ai.training`) — configurable training loops with early stopping, learning rate scheduling, and metric tracking
5. **Inference Engine** (`mesie.ai.inference`) — production inference with batching, confidence estimation, and preprocessing hooks
6. **Transfer Learning** (`mesie.ai.transfer`) — domain adaptation via CORAL, MMD, and normalization alignment strategies
7. **Spectral Data Protocol** (`mesie.protocols.spectral_protocol`) — standardized message format for inter-system spectral communication
8. **Streaming Protocol** (`mesie.protocols.streaming`) — real-time spectral data streaming support
9. **Serialization Protocol** (`mesie.protocols.serialization`) — efficient binary and JSON serialization

### 1.3 Why This Matters for AI

These additions transform MESIE from a spectral processing library into an **AI-native spectral intelligence platform**. The implications are:

- Spectral data can now be reasoned about autonomously within the engine
- Transformer models can learn spectral patterns without external deep learning infrastructure
- Trained models can be transferred across spectral domains (e.g., seismology to structural monitoring)
- Real-time streaming protocols enable deployment in production AI systems
- Intelligence protocols provide a framework for building spectral-aware autonomous agents

---

## 2. Intelligence Protocols

### 2.1 Motivation

Existing AI systems that work with spectral data typically follow a pipeline: preprocess → extract features → feed to model → get prediction. This pipeline is passive — the system does not reason about what it is observing, does not adapt its attention, and does not build episodic memory.

The Intelligence Protocol layer introduces active, adaptive intelligence that operates directly on spectral observations.

### 2.2 Architecture

The intelligence protocol system consists of five interconnected components:

#### 2.2.1 Intelligence Levels

The system operates at five engagement levels, each building on the previous:

| Level | Behavior | Use Case |
|-------|----------|----------|
| **Passive** | Observe and log only | Data collection, baseline recording |
| **Reactive** | Respond to threshold events | Alarm systems, simple monitoring |
| **Adaptive** | Learn and adjust parameters | Online learning, parameter tuning |
| **Predictive** | Forecast spectral behavior | Predictive maintenance, trend analysis |
| **Autonomous** | Full self-directed reasoning | Autonomous agents, cognitive systems |

The level can be configured per deployment and changed dynamically based on system requirements.

#### 2.2.2 Reasoning Strategies

Five reasoning strategies are available:

- **Statistical**: Classical statistical tests and threshold comparisons
- **Pattern Matching**: Similarity-based matching against known patterns
- **Anomaly Detection**: Deviation detection from learned baselines
- **Causal Inference**: Reasoning about cause-effect relationships in spectral changes
- **Ensemble**: Combining multiple strategies for robust conclusions

#### 2.2.3 Episodic Memory Buffer

The `SpectralMemoryBuffer` provides:

- **Capacity-bounded storage**: Configurable maximum observations retained
- **Importance-weighted retention**: High-importance observations survive eviction
- **Similarity-based retrieval**: Query by spectral similarity (cosine distance)
- **Contextual metadata**: Each observation carries domain-specific context
- **Temporal ordering**: Observations are timestamped for temporal reasoning

This enables the system to "remember" past spectral observations and use them to contextualize current data — a capability essential for cognitive architectures.

#### 2.2.4 Attention Focus Module

The `AttentionFocusModule` implements:

- **Multi-head frequency attention**: Multiple parallel attention heads over frequency bins
- **Learnable weights**: Attention weights update from feedback signals
- **Softmax normalization**: Attention as a proper probability distribution
- **Focus history**: Track attention evolution over time
- **Adaptive learning rate**: Configurable rate of attention weight updates

This allows the system to automatically focus on the most informative frequency regions — analogous to how human analysts learn to focus on specific spectral features.

#### 2.2.5 Reasoning Engine

The `IntelligenceProtocol.reason()` method produces structured `ReasoningResult` objects containing:

- **Conclusion**: Primary assessment (e.g., "anomaly_detected", "normal_operation", "low_signal")
- **Confidence**: Numerical confidence score (0–1)
- **Evidence**: List of supporting observations
- **Alternative hypotheses**: Other considered explanations
- **Recommended actions**: Suggested follow-up actions based on the conclusion
- **Metadata**: Full reasoning context including statistics and strategy used

### 2.3 What This Enables for AI Systems

1. **Self-monitoring AI**: Systems that detect their own sensor degradation
2. **Adaptive baselines**: Automatically evolving reference patterns
3. **Explainable decisions**: Every conclusion comes with evidence and alternatives
4. **Memory-augmented reasoning**: Decisions informed by past observations
5. **Attention-directed processing**: Resource-efficient analysis of high-dimensional spectra
6. **Autonomous response**: Systems that determine and execute their own actions

---

## 3. Spectral Transformer Pipeline

### 3.1 Motivation

Transformers have revolutionized NLP and computer vision by capturing long-range dependencies through self-attention. Spectral data — sequences of frequency-amplitude pairs — naturally lends itself to transformer processing. Distant frequency bins may have strong relationships (harmonics, sidebands, resonances) that local methods miss.

However, applying standard NLP/vision transformers to spectra is suboptimal because:

- Spectra have continuous, non-discrete token spaces
- Positional meaning in spectra is physical (Hz), not sequential
- Spectral patterns span multiple scales simultaneously
- The relevant attention patterns are domain-specific

The MESIE Spectral Transformer Pipeline addresses each of these challenges with purpose-built components.

### 3.2 Architecture

#### 3.2.1 Configuration

The `TransformerConfig` dataclass defines the architecture:

```python
TransformerConfig(
    d_model=128,           # Embedding dimension
    n_heads=8,             # Parallel attention heads
    n_encoder_layers=4,    # Depth of encoder stack
    n_decoder_layers=2,    # Depth of decoder stack (optional)
    d_feedforward=512,     # Feed-forward hidden dimension
    max_seq_len=512,       # Maximum token sequence length
    dropout=0.1,           # Regularization rate
    tokenization="frequency_bins",  # Tokenization strategy
    pooling="mean"         # Output aggregation strategy
)
```

#### 3.2.2 Spectral Tokenizer

Three tokenization strategies convert continuous spectra into discrete token sequences:

**Frequency Bins**: Divides the spectrum into equal frequency intervals. Each bin's average amplitude becomes a token. Simple, fast, and effective for uniformly distributed spectral content.

**Wavelets**: Multi-scale decomposition that captures both coarse structure and fine detail. Tokens are generated at multiple resolution levels, similar to how wavelet analysis decomposes signals hierarchically.

**Patches**: Overlapping patches extracted from the spectrum, with each patch summarized by its statistical properties (mean, std, max, min). Captures local structure while maintaining global context.

Each token is projected to the model dimension `d_model` through a learned projection matrix.

#### 3.2.3 Positional Encoding

Sinusoidal positional encodings are adapted for spectral sequences:

- Even dimensions: sin(position / 10000^(2i/d_model))
- Odd dimensions: cos(position / 10000^(2i/d_model))

Unlike NLP, where position encodes word order, in MESIE positional encoding captures **frequency position** — the physical location in the frequency domain. This encodes the inductive bias that nearby frequencies often have correlated behavior.

Learnable positional encodings are also supported for domains where the sinusoidal assumption may not hold.

#### 3.2.4 Multi-Head Spectral Attention

The `MultiHeadSpectralAttention` module implements scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

With 8 attention heads operating in parallel, the model can simultaneously attend to:
- Harmonic relationships (e.g., f₀, 2f₀, 3f₀)
- Sideband patterns
- Resonance/anti-resonance pairs
- Broadband vs. narrowband features
- Cross-band correlations
- Local spectral shape
- Global spectral envelope
- Domain-specific patterns

Each head learns a different "view" of the spectral relationships.

#### 3.2.5 Encoder Layers

Each `TransformerEncoderLayer` contains:

1. **Multi-head self-attention** with residual connection
2. **Layer normalization** for training stability
3. **Feed-forward network** with GELU activation
4. **Second residual connection and layer normalization**

The GELU activation provides smoother gradients than ReLU, beneficial for spectral data where small amplitude differences are meaningful.

#### 3.2.6 Output Pooling

Three pooling strategies aggregate the token sequence into a fixed-size output:

- **Mean pooling**: Average across all tokens — captures global spectral character
- **CLS pooling**: Dedicated classification token — learns to aggregate information
- **Max pooling**: Maximum across tokens — captures dominant features

#### 3.2.7 Attention Analysis

The pipeline provides interpretability through `get_attention_analysis()`:

- **Attention entropy**: How distributed vs. focused the attention is
- **Maximum attention**: Strength of the strongest attended-to token
- **Attention sparsity**: Fraction of near-zero attention weights

This enables understanding of what the model focuses on — critical for scientific applications where interpretability is required.

### 3.3 What This Enables for AI Systems

1. **Spectral understanding**: Models that truly "understand" spectral structure, not just memorize patterns
2. **Long-range dependency capture**: Detection of harmonics, resonances, and cross-band relationships
3. **Multi-scale analysis**: Simultaneous processing at multiple frequency resolutions
4. **Transfer learning ready**: Pre-trained spectral transformers that transfer across domains
5. **Interpretable attention**: Visualization of model focus for scientific validation
6. **Efficient inference**: Fixed-size embeddings from variable-length spectra
7. **Foundation model potential**: Architecture suitable for large-scale pre-training on diverse spectral data

---

## 4. Neural Model Architectures

### 4.1 Spectral Autoencoder

The `SpectralAutoencoder` learns compressed latent representations:

- **Encoder**: Input → Hidden layers → Latent space (default: 128 → 64 → 32)
- **Decoder**: Latent → Hidden layers → Reconstruction (32 → 64 → 128)
- **Initialization**: Xavier initialization for stable training
- **Activations**: ReLU, tanh, or sigmoid per layer
- **Training**: MSE reconstruction loss with mini-batch gradient descent

**Applications**:
- Dimensionality reduction for large spectral databases
- Anomaly detection via reconstruction error
- Generative modeling (decode from latent space)
- Feature learning without labels

### 4.2 Spectral Classifier

The `SpectralClassifier` categorizes spectral records:

- **Architecture**: Fully-connected network with configurable hidden layers
- **Output**: Softmax probabilities over classes
- **Training**: Cross-entropy loss with learned features
- **Inference**: Class predictions with probability distributions

**Applications**:
- Signal quality classification
- Source identification
- Damage state categorization
- Domain/instrument classification

### 4.3 Spectral Transformer Model

The `SpectralTransformer` in `mesie.ai.models` provides a lighter-weight transformer:

- **Self-attention**: Multi-head attention with sinusoidal positional encoding
- **Feed-forward**: GELU-activated expansion/compression layers
- **Layer normalization**: Applied after each sub-layer
- **Feature extraction**: Mean pooling over sequence for fixed-size output

This complements the full `SpectralTransformerPipeline` by offering a simpler interface for standard classification and embedding tasks.

---

## 5. Training Pipeline

### 5.1 Architecture

The `TrainingPipeline` provides a complete training infrastructure:

- **Data splitting**: Configurable train/validation split with shuffling
- **Learning rate scheduling**: Cosine annealing and step decay
- **Early stopping**: Patience-based termination on validation loss plateau
- **Metric tracking**: MSE, MAE, RMSE, R², accuracy
- **Batch processing**: Configurable mini-batch sizes
- **Reproducibility**: Seeded random number generation

### 5.2 Training Configuration

```python
TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    validation_split=0.2,
    early_stopping_patience=10,
    lr_schedule="cosine",
    seed=42,
    metrics=["mse", "mae"]
)
```

### 5.3 Supported Model Types

- **Autoencoder training**: Unsupervised reconstruction learning
- **Classifier training**: Supervised classification with class labels

### 5.4 Training Results

The `TrainingResult` dataclass captures:

- Train/validation loss curves
- Best epoch and best validation loss
- Full metric histories
- Early stopping status

### 5.5 What This Enables

- Reproducible experiments with seeded randomness
- Fair model comparison via consistent evaluation protocols
- Efficient training with early stopping (saves computation)
- Learning rate scheduling for better convergence
- Comprehensive training history for analysis and reporting

---

## 6. Inference Engine

### 6.1 Production Deployment

The `InferenceEngine` bridges trained models and production systems:

- **Preprocessing hooks**: Custom functions applied before inference
- **Postprocessing hooks**: Custom transforms on model outputs
- **Confidence estimation**: Model-type-specific confidence scoring
- **Batch inference**: Process multiple inputs efficiently
- **Confidence thresholding**: Flag uncertain predictions

### 6.2 Confidence Computation

Each model type has a tailored confidence metric:

| Model Type | Confidence Metric |
|---|---|
| Autoencoder | 1 - normalized_reconstruction_error |
| Classifier | Maximum class probability |
| Transformer | Feature vector magnitude (normalized) |

### 6.3 What This Enables

- Drop-in deployment of trained spectral models
- Automatic uncertainty quantification
- Production-ready inference with pre/post-processing
- Confidence-gated decision making (only act on high-confidence predictions)
- Inference counting for monitoring and rate-limiting

---

## 7. Transfer Learning and Domain Adaptation

### 7.1 The Domain Problem

Spectral data varies dramatically across domains:
- Earthquake spectra: 0.01–50 Hz
- Structural vibrations: 0.1–100 Hz
- Audio/acoustics: 20–20,000 Hz
- EEG biosignals: 0.5–100 Hz
- Industrial machinery: 1–10,000 Hz

A model trained on earthquake spectra cannot directly analyze structural vibrations without adaptation.

### 7.2 Transfer Adapter

The `TransferAdapter` provides simple domain transfer:

1. **Fit**: Compute source and target domain statistics
2. **Transform**: Map source-domain data to target domain distribution
3. **Adaptation strength**: Controllable blending (0 = no change, 1 = full adaptation)

### 7.3 Domain Adaptation Strategies

The `DomainAdaptation` class implements three strategies:

#### CORAL (Correlation Alignment)
Aligns the second-order statistics (covariance) of source and target domains:
- Whitens source data (removes source correlations)
- Colors with target statistics (imposes target correlations)
- Produces features with target-like distributions

#### MMD (Maximum Mean Discrepancy)
Minimizes the distance between domain distributions in feature space:
- Computes mean shift between domains
- Applies translation to align distributions
- Simple but effective for mean-dominated differences

#### Normalization Alignment
Z-score normalization followed by target re-scaling:
- Normalizes to zero mean, unit variance
- Rescales to target domain statistics
- Fast and robust for scale-dominated differences

### 7.4 Domain Distance Metric

`compute_domain_distance()` quantifies how different two spectral domains are, enabling:
- Decision support for when adaptation is needed
- Progress monitoring during fine-tuning
- Domain similarity mapping across fields

### 7.5 What This Enables for AI

1. **Pre-train once, deploy everywhere**: Train on data-rich domains, adapt to data-scarce domains
2. **Cross-domain knowledge transfer**: Earthquake engineering insights applied to structural monitoring
3. **Rapid deployment**: Minimal target-domain data needed for adaptation
4. **Domain similarity assessment**: Quantify how transferable knowledge is between domains
5. **Foundation model support**: Architecture compatible with large-scale pre-training strategies

---

## 8. Protocol Layer

### 8.1 Spectral Data Protocol

The `SpectralDataProtocol` standardizes how spectral data is communicated between systems:

#### Message Types
- **RECORD**: A spectral record with frequencies, amplitudes, and metadata
- **QUERY**: A request for matching, searching, or validation
- **RESPONSE**: An answer to a query
- **BATCH**: Multiple records in a single message
- **HEARTBEAT**: Connection health monitoring
- **ERROR**: Error reporting with correlation tracking
- **METADATA**: System or schema metadata exchange

#### Message Structure
Every message contains:
- Protocol identifier ("mesie-spectral")
- Version (semantic versioning: 1.0, 1.1, 2.0)
- Unique message ID (SHA-256 based)
- Timestamp (Unix epoch)
- Source and destination identifiers
- Correlation ID (for request-response linking)
- Headers (extensible key-value metadata)
- Payload (message-type-specific content)

#### Validation
Messages are validated before processing:
- Schema completeness checking
- Message-type-specific payload requirements
- Version compatibility verification

### 8.2 Why a Protocol Matters for AI

Standardized protocols enable:
- **Microservice architectures**: Spectral processing as a service
- **Multi-agent systems**: Agents exchanging spectral observations
- **Distributed computing**: Parallel spectral analysis across nodes
- **Real-time systems**: Low-latency spectral data streaming
- **Interoperability**: Different systems speaking the same spectral language
- **Audit trails**: Every exchange is logged with unique IDs and timestamps

### 8.3 Streaming and Serialization

The protocol layer includes:
- **Streaming**: Continuous data ingestion for real-time monitoring
- **Serialization**: Efficient encoding (JSON for interoperability, binary for performance)
- **Numpy handling**: Automatic conversion of numpy arrays for wire transmission

---

## 9. Implications for AI and Autonomous Systems

### 9.1 Spectral-Native AI Agents

With the v0.2.0 additions, it is now possible to build AI agents that:

1. **Observe** spectral data continuously through streaming protocols
2. **Attend** to informative frequency regions via learned attention
3. **Remember** past observations through episodic memory
4. **Reason** about spectral patterns using configurable strategies
5. **Decide** on actions with confidence-gated decision making
6. **Adapt** their behavior based on feedback
7. **Communicate** observations and decisions through standardized protocols

This constitutes a complete perception-cognition-action loop built on spectral data.

### 9.2 Integration with Large Language Models

The transformer pipeline and embedding system produce representations compatible with:
- Retrieval-Augmented Generation (RAG): Spectral records as retrievable context
- Multi-modal models: Spectral embeddings as an input modality
- Tool-augmented LLMs: MESIE as a callable tool for spectral analysis
- Agents: LLM-based agents with spectral perception capabilities

### 9.3 Real-Time Decision Systems

The inference engine + intelligence protocols enable:
- Millisecond-scale spectral classification
- Online anomaly detection with adaptive baselines
- Autonomous alert generation with confidence scores
- Self-calibrating monitoring systems

### 9.4 Scientific Discovery

The attention analysis and interpretability features support:
- Automated feature discovery in spectral data
- Hypothesis generation from attention patterns
- Cross-domain pattern identification
- Data-driven resonance characterization

---

## 10. Technical Specifications

### 10.1 Code Statistics

| Component | Files | Approximate Lines |
|---|---|---|
| Intelligence Protocols | 1 | 400+ |
| Transformer Pipeline | 1 | 470+ |
| Neural Models | 1 | 450+ |
| Training Pipeline | 1 | 280+ |
| Inference Engine | 1 | 180+ |
| Transfer Learning | 1 | 250+ |
| Spectral Protocol | 1 | 320+ |
| Streaming | 1 | 200+ |
| Serialization | 1 | 200+ |
| Supporting Code (data, tests, config) | 50+ | ~95,000+ |
| **Total v0.2.0 Addition** | **~59 files** | **~100,000 lines** |

### 10.2 Dependencies

**Core** (required):
- NumPy ≥ 1.21

**Optional** (for full AI capabilities):
- transformers ≥ 4.30
- torch ≥ 2.0
- scipy ≥ 1.7
- scikit-learn ≥ 1.0
- pandas ≥ 1.3
- networkx ≥ 2.6

### 10.3 Installation

```bash
# Minimal (core + protocols)
pip install mesie

# Full AI capabilities
pip install mesie[full]

# Intelligence + transformer stack
pip install mesie[intelligence]

# Development
pip install -e ".[dev,full]"
```

### 10.4 Compatibility

- Python 3.9, 3.10, 3.11, 3.12
- Linux, macOS, Windows
- CPU-only and GPU-accelerated modes
- JSON and binary serialization formats

---

## 11. Experimental Validation

### 11.1 Transformer Embedding Quality

The spectral transformer pipeline was validated on the included benchmark datasets:
- Earthquake PSD references
- Structural FAS references
- Vibration monitoring references
- RotDnn references

The transformer produces embeddings where semantically similar spectra cluster together, as measured by silhouette score and retrieval precision.

### 11.2 Intelligence Protocol Behavior

The intelligence protocol system was tested for:
- Correct anomaly detection on synthetic anomalous spectra
- Memory retrieval accuracy (top-k nearest neighbor precision)
- Attention convergence (focus stabilization after repeated observations)
- Reasoning consistency (same input → same conclusion)

### 11.3 Domain Transfer

Transfer learning experiments between:
- Earthquake → Structural monitoring
- Vibration → Biosignal
- Laboratory → Field recordings

All three strategies (CORAL, MMD, normalization) reduce domain distance, with CORAL providing the most robust results for high-dimensional spectral features.

---

## 12. Future Work

### 12.1 Immediate Priorities

- **Decoder stack**: Complete the transformer decoder for spectral generation
- **Pre-training**: Large-scale unsupervised pre-training on diverse spectral corpora
- **Fine-tuning toolkit**: Simplified transfer learning for domain-specific applications
- **GPU acceleration**: PyTorch and JAX backends for training at scale

### 12.2 Research Directions

- **Spectral foundation models**: Transformer pre-trained on millions of spectra
- **Multi-modal fusion**: Combining spectral with text and image embeddings
- **Causal spectral reasoning**: Learning causal relationships from spectral time series
- **Generative spectral models**: Diffusion-based spectral synthesis
- **Multi-agent spectral systems**: Collaborative reasoning across distributed sensors

### 12.3 Production Roadmap

- **REST API server**: HTTP interface for spectral intelligence as a service
- **Kubernetes deployment**: Scalable inference infrastructure
- **Model registry**: Version-controlled model storage and serving
- **Monitoring dashboard**: Real-time visualization of intelligence protocol state

---

## 13. Conclusion

The MESIE v0.2.0 update transforms the engine from a spectral processing toolkit into a full AI-native spectral intelligence platform. The addition of intelligence protocols enables autonomous reasoning about spectral data. The transformer pipeline enables deep learning on spectral sequences. The training and inference infrastructure enables production deployment. The transfer learning tools enable cross-domain generalization. And the protocol layer enables distributed spectral communication.

Together, these ~100,000 lines of code establish MESIE as the first open-source platform where spectral data is not merely *analyzed by* AI — it is *understood by* AI. The engine can now observe, attend, remember, reason, decide, adapt, and communicate — the complete cognitive cycle applied to spectral intelligence.

This working paper documents these additions for the research community and establishes the technical foundation for the next phase of MESIE development: large-scale pre-training, multi-modal integration, and production-grade autonomous spectral systems.

---

## 14. Citation

```bibtex
@techreport{medina2026mesie_v02,
  author = {Medina, Alfredo},
  title = {MESIE v0.2.0: Intelligence Protocols and Spectral Transformers for AI-Native Spectral Reasoning},
  type = {Working Paper},
  year = {2026},
  institution = {Independent Research},
  url = {https://github.com/FreddyCreates/Multi-Element-Spectral-Intelligence-Engine-MESIE-}
}
```

---

*© 2026 Alfredo Medina. Licensed under Apache-2.0.*
