"""
MESIEPolyglot — Julia module for spectral intelligence computations.

Provides high-performance spectral validation, matching, embedding,
and fingerprinting routines callable from Python via juliacall or JSON IPC.
"""
module MESIEPolyglot

using JSON
using LinearAlgebra
using Statistics

export validate_record, match_records, spectral_embed, spectral_fingerprint
export primary_arrays, health_check, dispatch_action


"""
    primary_arrays(record::Dict) -> (Vector{Float64}, Vector{Float64})

Extract the primary frequency and amplitude arrays from a record dict.
Handles both top-level and nested component formats.
"""
function primary_arrays(record::Dict)
    comps = get(record, "components", nothing)
    if comps !== nothing && length(comps) > 0
        c = comps[1]
        return Float64.(get(c, "frequency", Float64[])), Float64.(get(c, "amplitude", Float64[]))
    end
    return Float64.(get(record, "frequency", Float64[])), Float64.(get(record, "amplitude", Float64[]))
end


"""
    validate_record(record::Dict) -> Dict

Validate a spectral record for structural and physical correctness.
Returns a dict with validation status, quality level, errors, and warnings.
"""
function validate_record(record::Dict)
    freq, amp = primary_arrays(record)
    errors = String[]
    warnings = String[]

    if isempty(freq) || isempty(amp)
        push!(errors, "missing frequency/amplitude")
    end
    if length(freq) != length(amp)
        push!(errors, "length mismatch: freq=$(length(freq)), amp=$(length(amp))")
    end
    if any(x -> x < 0, amp)
        push!(errors, "negative amplitudes detected")
    end
    if !isempty(freq) && !issorted(freq)
        push!(warnings, "frequency array not sorted")
    end
    if !isempty(amp) && all(x -> x == 0.0, amp)
        push!(warnings, "all-zero amplitudes")
    end

    level = if isempty(errors) && isempty(warnings)
        5
    elseif isempty(errors)
        4
    else
        2
    end

    return Dict(
        "is_valid" => isempty(errors),
        "level" => level,
        "errors" => errors,
        "warnings" => warnings,
        "n_points" => length(freq),
        "runtime" => "julia",
    )
end


"""
    match_records(a::Dict, b::Dict) -> Dict

Compute spectral similarity between two records using cosine similarity and RMSE.
Returns composite score and individual metrics.
"""
function match_records(a::Dict, b::Dict)
    _, aa = primary_arrays(a)
    _, ab = primary_arrays(b)
    n = min(length(aa), length(ab))

    if n == 0
        return Dict(
            "composite_score" => 0.0,
            "metrics" => Dict("cosine" => 0.0, "rmse" => 1.0, "pearson" => 0.0),
            "n_compared" => 0,
            "runtime" => "julia",
        )
    end

    va, vb = aa[1:n], ab[1:n]

    cosine = dot(va, vb) / (norm(va) * norm(vb) + 1e-12)
    rmse = sqrt(mean((va .- vb) .^ 2))
    pearson = if std(va) > 0 && std(vb) > 0
        cor(va, vb)
    else
        0.0
    end

    score = clamp(0.5 * cosine + 0.3 * (1.0 / (1.0 + rmse)) + 0.2 * ((pearson + 1.0) / 2.0), 0.0, 1.0)

    return Dict(
        "composite_score" => score,
        "metrics" => Dict("cosine" => cosine, "rmse" => rmse, "pearson" => pearson),
        "n_compared" => n,
        "runtime" => "julia",
    )
end


"""
    spectral_embed(record::Dict; n_bands::Int=8) -> Vector{Float64}

Compute a spectral embedding vector by partitioning the amplitude spectrum
into bands and extracting statistical features from each band.
"""
function spectral_embed(record::Dict; n_bands::Int=8)
    freq, amp = primary_arrays(record)
    if isempty(amp)
        return zeros(Float64, n_bands * 4)
    end

    # Partition into bands
    n = length(amp)
    band_size = max(1, div(n, n_bands))
    embedding = Float64[]

    for i in 1:n_bands
        start_idx = (i - 1) * band_size + 1
        end_idx = min(i * band_size, n)
        if start_idx > n
            append!(embedding, [0.0, 0.0, 0.0, 0.0])
            continue
        end
        band = amp[start_idx:end_idx]
        push!(embedding, mean(band))          # mean energy
        push!(embedding, std(band))           # variability
        push!(embedding, maximum(band))       # peak
        push!(embedding, sum(band) / (sum(amp) + 1e-12))  # relative energy
    end

    # L2 normalize
    nrm = norm(embedding)
    if nrm > 0
        embedding ./= nrm
    end

    return embedding
end


"""
    spectral_fingerprint(record::Dict; resolution::Int=16) -> Vector{UInt8}

Compute a compact binary fingerprint for fast approximate matching via Hamming distance.
"""
function spectral_fingerprint(record::Dict; resolution::Int=16)
    _, amp = primary_arrays(record)
    if isempty(amp)
        return zeros(UInt8, resolution)
    end

    # Downsample amplitude to resolution bins
    n = length(amp)
    bin_size = max(1, div(n, resolution))
    bins = Float64[]

    for i in 1:resolution
        start_idx = (i - 1) * bin_size + 1
        end_idx = min(i * bin_size, n)
        if start_idx > n
            push!(bins, 0.0)
        else
            push!(bins, mean(amp[start_idx:end_idx]))
        end
    end

    # Convert to binary fingerprint: above median = 1, below = 0
    med = median(bins)
    return UInt8[b > med ? 1 : 0 for b in bins]
end


"""
    health_check() -> Dict

Return runtime health status.
"""
function health_check()
    return Dict(
        "status" => "ok",
        "runtime" => "julia",
        "version" => string(VERSION),
        "threads" => Threads.nthreads(),
    )
end


"""
    dispatch_action(request::Dict) -> Dict

Route a JSON request to the appropriate function based on the "action" field.
"""
function dispatch_action(request::Dict)
    action = get(request, "action", "health")

    data = if action == "health"
        health_check()
    elseif action == "validate"
        validate_record(get(request, "record", Dict()))
    elseif action == "match"
        a = get(request, "record_a", get(request, "record", Dict()))
        b = get(request, "record_b", Dict())
        match_records(a, b)
    elseif action == "embed"
        record = get(request, "record", Dict())
        n_bands = get(request, "n_bands", 8)
        Dict("embedding" => spectral_embed(record; n_bands=n_bands), "runtime" => "julia")
    elseif action == "fingerprint"
        record = get(request, "record", Dict())
        resolution = get(request, "resolution", 16)
        Dict("fingerprint" => spectral_fingerprint(record; resolution=resolution), "runtime" => "julia")
    else
        Dict("error" => "unsupported action: $action")
    end

    return Dict("ok" => !haskey(data, "error"), "data" => data)
end

end  # module
