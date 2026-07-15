#!/usr/bin/env julia
# AISVectorPolyglot Julia CLI — reads JSON stdin, writes JSON stdout

using JSON
using LinearAlgebra: dot, norm
using Statistics: mean

function primary_arrays(record)
    comps = get(record, "components", nothing)
    if comps !== nothing && length(comps) > 0
        c = comps[1]
        return Float64.(get(c, "frequency", [])), Float64.(get(c, "amplitude", []))
    end
    return Float64.(get(record, "frequency", [])), Float64.(get(record, "amplitude", []))
end

function validate_record(record)
    freq, amp = primary_arrays(record)
    errors = String[]
    if isempty(freq) || isempty(amp)
        push!(errors, "missing frequency/amplitude")
    end
    if length(freq) != length(amp)
        push!(errors, "length mismatch")
    end
    if any(x -> x < 0, amp)
        push!(errors, "negative amplitudes")
    end
    return Dict(
        "is_valid" => isempty(errors),
        "level" => isempty(errors) ? 5 : 2,
        "errors" => errors,
        "warnings" => String[],
        "runtime" => "julia",
    )
end

function match_records(a, b)
    _, aa = primary_arrays(a)
    _, ab = primary_arrays(b)
    n = min(length(aa), length(ab))
    if n == 0
        return Dict("composite_score" => 0.0, "metrics" => Dict("cosine" => 0.0, "rmse" => 1.0))
    end
    va, vb = aa[1:n], ab[1:n]
    cosine = dot(va, vb) / (norm(va) * norm(vb) + 1e-12)
    rmse = sqrt(mean((va .- vb) .^ 2))
    score = clamp(0.6 * cosine + 0.4 * (1.0 / (1.0 + rmse)), 0.0, 1.0)
    return Dict(
        "composite_score" => score,
        "metrics" => Dict("cosine" => cosine, "rmse" => rmse),
        "runtime" => "julia",
    )
end

input = read(stdin, String)
req = JSON.parse(input)
action = get(req, "action", "health")
data = if action == "health"
    Dict("status" => "ok")
elseif action == "validate"
    validate_record(get(req, "record", Dict()))
elseif action == "match"
    a = get(req, "record_a", get(req, "record", Dict()))
    b = get(req, "record_b", Dict())
    match_records(a, b)
else
    Dict("error" => "unsupported action")
end

println(JSON.json(Dict("ok" => true, "data" => data)))