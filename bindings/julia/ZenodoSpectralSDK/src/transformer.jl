"""
Spectral Transformer Pipeline — Transformer-based spectral intelligence model.

Implements the core transformer architecture for spectral data processing,
following MESIE's transformer pipeline design:
- Spectral tokenization (frequency-band windowing)
- Positional encoding (sinusoidal)
- Multi-head spectral attention
- Feed-forward encoding
- Spectral embedding generation

Based on MESIE's SpectralTransformerPipeline (Python) adapted for Julia with
high-performance linear algebra.
"""

# --- Configuration ---

"""
    SpectralTransformer

Configuration for the spectral transformer model.

# Fields
- `d_model::Int`: Model dimension (embedding size)
- `n_heads::Int`: Number of attention heads
- `n_layers::Int`: Number of transformer encoder layers
- `d_ff::Int`: Feed-forward hidden dimension
- `max_seq_len::Int`: Maximum sequence length (number of spectral tokens)
- `n_bands::Int`: Number of frequency bands for tokenization
- `dropout::Float64`: Dropout rate (used during training)
"""
struct SpectralTransformer
    d_model::Int
    n_heads::Int
    n_layers::Int
    d_ff::Int
    max_seq_len::Int
    n_bands::Int
    dropout::Float64

    # Learned parameters (initialized randomly for inference demo)
    W_q::Matrix{Float64}
    W_k::Matrix{Float64}
    W_v::Matrix{Float64}
    W_o::Matrix{Float64}
    W_ff1::Matrix{Float64}
    W_ff2::Matrix{Float64}
    b_ff1::Vector{Float64}
    b_ff2::Vector{Float64}

    function SpectralTransformer(;
        d_model::Int=64,
        n_heads::Int=4,
        n_layers::Int=2,
        d_ff::Int=128,
        max_seq_len::Int=128,
        n_bands::Int=16,
        dropout::Float64=0.1,
        seed::Int=42
    )
        rng = Random.MersenneTwister(seed)
        scale = 1.0 / sqrt(d_model)

        W_q = randn(rng, d_model, d_model) .* scale
        W_k = randn(rng, d_model, d_model) .* scale
        W_v = randn(rng, d_model, d_model) .* scale
        W_o = randn(rng, d_model, d_model) .* scale
        W_ff1 = randn(rng, d_ff, d_model) .* scale
        W_ff2 = randn(rng, d_model, d_ff) .* scale
        b_ff1 = zeros(d_ff)
        b_ff2 = zeros(d_model)

        new(d_model, n_heads, n_layers, d_ff, max_seq_len, n_bands, dropout,
            W_q, W_k, W_v, W_o, W_ff1, W_ff2, b_ff1, b_ff2)
    end
end


"""
    tokenize_spectrum(amplitude::Vector{Float64}, frequency::Vector{Float64};
                      n_bands::Int=16, d_model::Int=64) -> Matrix{Float64}

Tokenize a spectral signal into a sequence of spectral tokens.

Each token represents a frequency band and contains statistical features:
- Band mean amplitude
- Band peak amplitude
- Band energy (sum of squares)
- Band spectral slope
- Band variance
- Frequency center of band
- Bandwidth
- Spectral centroid within band

The token is padded/projected to d_model dimensions.

# Returns
Matrix of size (n_bands, d_model) where each row is a token embedding.
"""
function tokenize_spectrum(amplitude::Vector{Float64}, frequency::Vector{Float64};
    n_bands::Int=16, d_model::Int=64)

    n = length(amplitude)
    if n == 0
        return zeros(Float64, n_bands, d_model)
    end

    band_size = max(1, div(n, n_bands))
    tokens = zeros(Float64, n_bands, d_model)

    for i in 1:n_bands
        start_idx = (i - 1) * band_size + 1
        end_idx = min(i * band_size, n)

        if start_idx > n
            continue
        end

        band_amp = amplitude[start_idx:end_idx]
        band_freq = frequency[start_idx:end_idx]

        # Extract spectral features for this band
        band_mean = mean(band_amp)
        band_peak = maximum(band_amp)
        band_energy = sum(band_amp .^ 2)
        band_var = length(band_amp) > 1 ? var(band_amp) : 0.0

        # Spectral slope (linear regression approximation)
        if length(band_amp) > 1
            x = collect(1.0:length(band_amp))
            x_mean = mean(x)
            band_slope = sum((x .- x_mean) .* (band_amp .- band_mean)) /
                         (sum((x .- x_mean) .^ 2) + 1e-12)
        else
            band_slope = 0.0
        end

        # Frequency features
        freq_center = mean(band_freq)
        bandwidth = length(band_freq) > 1 ? band_freq[end] - band_freq[1] : 0.0

        # Spectral centroid within band
        total_amp = sum(band_amp) + 1e-12
        spectral_centroid = sum(band_freq .* band_amp) / total_amp

        # Build feature vector and project to d_model dimensions
        features = [band_mean, band_peak, band_energy, band_var,
                    band_slope, freq_center, bandwidth, spectral_centroid]

        # Fill token with features, repeating/cycling to fill d_model
        for j in 1:d_model
            tokens[i, j] = features[((j - 1) % length(features)) + 1]
        end
    end

    # Normalize tokens
    for i in 1:n_bands
        nrm = norm(tokens[i, :])
        if nrm > 0
            tokens[i, :] ./= nrm
        end
    end

    return tokens
end


"""
    positional_encoding(seq_len::Int, d_model::Int) -> Matrix{Float64}

Generate sinusoidal positional encodings for the transformer.

Uses the standard sin/cos positional encoding from "Attention Is All You Need":
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""
function positional_encoding(seq_len::Int, d_model::Int)
    pe = zeros(Float64, seq_len, d_model)

    for pos in 1:seq_len
        for i in 1:d_model
            angle = (pos - 1) / (10000.0^((2 * ((i - 1) ÷ 2)) / d_model))
            if i % 2 == 1
                pe[pos, i] = sin(angle)
            else
                pe[pos, i] = cos(angle)
            end
        end
    end

    return pe
end


"""
    spectral_attention(Q::Matrix{Float64}, K::Matrix{Float64}, V::Matrix{Float64};
                       mask::Union{Nothing, Matrix{Float64}}=nothing) -> Matrix{Float64}

Compute scaled dot-product attention for spectral sequences.

    Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V

# Arguments
- `Q`: Query matrix (seq_len × d_k)
- `K`: Key matrix (seq_len × d_k)
- `V`: Value matrix (seq_len × d_v)
- `mask`: Optional attention mask

# Returns
Attention output matrix (seq_len × d_v).
"""
function spectral_attention(Q::Matrix{Float64}, K::Matrix{Float64}, V::Matrix{Float64};
    mask::Union{Nothing,Matrix{Float64}}=nothing)

    d_k = size(K, 2)
    scale = sqrt(Float64(d_k))

    # Compute attention scores
    scores = (Q * K') ./ scale

    # Apply mask if provided
    if mask !== nothing
        scores .+= mask
    end

    # Softmax along last dimension (columns for each row)
    attention_weights = _softmax_rows(scores)

    # Apply attention to values
    return attention_weights * V
end


"""
    multi_head_attention(model::SpectralTransformer, x::Matrix{Float64}) -> Matrix{Float64}

Apply multi-head attention to the input sequence.

Splits the input into multiple heads, applies scaled dot-product attention
to each head independently, then concatenates and projects.
"""
function multi_head_attention(model::SpectralTransformer, x::Matrix{Float64})
    seq_len = size(x, 1)
    d_k = div(model.d_model, model.n_heads)

    # Project to Q, K, V
    Q = x * model.W_q'
    K = x * model.W_k'
    V = x * model.W_v'

    # Multi-head: split, attend, concatenate
    output = zeros(Float64, seq_len, model.d_model)

    for h in 1:model.n_heads
        start_col = (h - 1) * d_k + 1
        end_col = h * d_k

        Q_h = Q[:, start_col:end_col]
        K_h = K[:, start_col:end_col]
        V_h = V[:, start_col:end_col]

        attn_h = spectral_attention(Q_h, K_h, V_h)
        output[:, start_col:end_col] = attn_h
    end

    # Output projection
    return output * model.W_o'
end


"""
    feed_forward(model::SpectralTransformer, x::Matrix{Float64}) -> Matrix{Float64}

Apply position-wise feed-forward network with ReLU activation.

    FFN(x) = max(0, x * W1 + b1) * W2 + b2
"""
function feed_forward(model::SpectralTransformer, x::Matrix{Float64})
    # First linear + ReLU
    hidden = x * model.W_ff1' .+ model.b_ff1'
    hidden = max.(hidden, 0.0)  # ReLU

    # Second linear
    return hidden * model.W_ff2' .+ model.b_ff2'
end


"""
    layer_norm(x::Matrix{Float64}; eps::Float64=1e-6) -> Matrix{Float64}

Apply layer normalization across the feature dimension.
"""
function layer_norm(x::Matrix{Float64}; eps::Float64=1e-6)
    μ = mean(x, dims=2)
    σ = std(x, dims=2) .+ eps
    return (x .- μ) ./ σ
end


"""
    transformer_encode(model::SpectralTransformer, tokens::Matrix{Float64}) -> Matrix{Float64}

Run the full transformer encoder stack on tokenized spectral input.

Applies:
1. Positional encoding addition
2. For each layer: Multi-head attention → Add & Norm → FFN → Add & Norm
"""
function transformer_encode(model::SpectralTransformer, tokens::Matrix{Float64})
    seq_len = size(tokens, 1)

    # Add positional encoding
    pe = positional_encoding(seq_len, model.d_model)
    x = tokens .+ pe[1:seq_len, :]

    # Apply transformer layers
    for _ in 1:model.n_layers
        # Multi-head self-attention with residual
        attn_out = multi_head_attention(model, x)
        x = layer_norm(x .+ attn_out)

        # Feed-forward with residual
        ff_out = feed_forward(model, x)
        x = layer_norm(x .+ ff_out)
    end

    return x
end


"""
    transformer_embed(model::SpectralTransformer, amplitude::Vector{Float64},
                      frequency::Vector{Float64}) -> Vector{Float64}

Generate a spectral embedding using the full transformer pipeline.

Pipeline: tokenize → positional encode → transformer encode → pool → normalize

# Returns
L2-normalized embedding vector of size d_model.
"""
function transformer_embed(model::SpectralTransformer, amplitude::Vector{Float64},
    frequency::Vector{Float64})

    # Tokenize
    tokens = tokenize_spectrum(amplitude, frequency;
        n_bands=model.n_bands, d_model=model.d_model)

    # Encode
    encoded = transformer_encode(model, tokens)

    # Global average pooling across sequence dimension
    embedding = vec(mean(encoded, dims=1))

    # L2 normalize
    nrm = norm(embedding)
    if nrm > 0
        embedding ./= nrm
    end

    return embedding
end


"""
    transformer_pipeline(record::Dict; model::Union{Nothing, SpectralTransformer}=nothing) -> Dict

Full transformer pipeline for a spectral record (dict format).

Extracts frequency/amplitude arrays, runs the transformer, returns embedding and metadata.
"""
function transformer_pipeline(record::Dict; model::Union{Nothing,SpectralTransformer}=nothing)
    if model === nothing
        model = SpectralTransformer()
    end

    # Extract arrays
    comps = get(record, "components", nothing)
    if comps !== nothing && length(comps) > 0
        c = comps[1]
        freq = Float64.(get(c, "frequency", Float64[]))
        amp = Float64.(get(c, "amplitude", Float64[]))
    else
        freq = Float64.(get(record, "frequency", Float64[]))
        amp = Float64.(get(record, "amplitude", Float64[]))
    end

    if isempty(freq) || isempty(amp)
        return Dict(
            "embedding" => zeros(Float64, model.d_model),
            "n_tokens" => 0,
            "d_model" => model.d_model,
            "runtime" => "julia",
            "model" => "SpectralTransformer",
        )
    end

    embedding = transformer_embed(model, amp, freq)

    return Dict(
        "embedding" => embedding,
        "n_tokens" => model.n_bands,
        "d_model" => model.d_model,
        "n_heads" => model.n_heads,
        "n_layers" => model.n_layers,
        "n_points" => length(freq),
        "runtime" => "julia",
        "model" => "SpectralTransformer",
    )
end


# --- Internal helpers ---

"""
    _softmax_rows(x::Matrix{Float64}) -> Matrix{Float64}

Apply softmax to each row of the matrix (numerically stable).
"""
function _softmax_rows(x::Matrix{Float64})
    result = similar(x)
    for i in 1:size(x, 1)
        row = x[i, :]
        row_max = maximum(row)
        exp_row = exp.(row .- row_max)
        result[i, :] = exp_row ./ sum(exp_row)
    end
    return result
end
