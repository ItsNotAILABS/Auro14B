"""
AuroCompute — Julia spectral / phi / multi-view kernels for Auro mind.

Called from Python polyglot organ via `julia -e` or JSON CLI.
"""
module AuroCompute

export spectral_energy, phi_powers, multi_fft_embed, matmul_train_step
export health, run_action

const PHI = (1 + sqrt(5)) / 2

function health()
    return Dict(
        "ok" => true,
        "lang" => "julia",
        "version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "cpu_threads" => Sys.CPU_THREADS,
        "phi" => PHI,
    )
end

"""Spectral energy: sum of |rfft| magnitudes (or pure DFT fallback)."""
function spectral_energy(x::AbstractVector{<:Real})
    n = length(x)
    n == 0 && return 0.0
    # DFT magnitude sum (no FFTW dep required)
    e = 0.0
    half = n ÷ 2
    @inbounds for k in 0:half
        re = 0.0
        im = 0.0
        for t in 0:(n-1)
            ang = 2π * k * t / n
            re += x[t+1] * cos(ang)
            im -= x[t+1] * sin(ang)
        end
        e += sqrt(re * re + im * im)
    end
    return e
end

function phi_powers(n::Int=12)
    return [PHI^i for i in 1:n]
end

"""Multi-scale FFT-like band energy embedding (fixed 64 bins × scales)."""
function multi_fft_embed(bytes::Vector{UInt8}; scales::Int=4, bins::Int=64)
    raw = Float64.(bytes)
    isempty(raw) && (raw = zeros(16))
    out = Float64[]
    n = length(raw)
    for i in 0:(scales-1)
        win = max(32, Int(round(n / (PHI^i))))
        seg = raw[1:min(win, n)]
        if length(seg) < win
            append!(seg, zeros(win - length(seg)))
        end
        # periodogram via DFT sample
        m = length(seg)
        spec = zeros(m ÷ 2 + 1)
        @inbounds for k in 0:(m ÷ 2)
            re = 0.0
            im = 0.0
            for t in 0:(m-1)
                ang = 2π * k * t / m
                re += seg[t+1] * cos(ang)
                im -= seg[t+1] * sin(ang)
            end
            spec[k+1] = sqrt(re * re + im * im)
        end
        # resample to bins
        b = zeros(bins)
        for j in 1:bins
            idx = clamp(Int(round(1 + (j-1) * (length(spec)-1) / (bins-1))), 1, length(spec))
            b[j] = spec[idx]
        end
        energy = sum(b) + 1e-12
        p = b ./ energy
        ent = -sum(p .* log.(p .+ 1e-12))
        centroid = sum((0:bins-1) .* b) / energy
        flat = exp(sum(log.(b .+ 1e-12)) / bins) / (sum(b) / bins + 1e-12)
        append!(out, b)
        append!(out, [energy, ent, centroid, flat])
    end
    # L2 normalize
    nrm = sqrt(sum(abs2, out)) + 1e-12
    return out ./ nrm
end

"""One CE-style train step on small dense matmul (Julia multi-thread)."""
function matmul_train_step(W::Matrix{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}; lr::Float64=1e-3)
    # W: [out, in], X: [in, batch], Y: [out, batch]
    pred = W * X
    err = pred - Y
    # MSE
    loss = sum(abs2, err) / length(err)
    grad = (2.0 / size(X, 2)) * (err * X')
    W2 = W - lr * grad
    return Dict("loss" => loss, "W" => W2, "grad_norm" => sqrt(sum(abs2, grad)))
end

function run_action(action::String, payload::Dict)
    if action == "health"
        return health()
    elseif action == "spectral_energy"
        x = Float64.(payload["x"])
        return Dict("ok" => true, "energy" => spectral_energy(x), "lang" => "julia")
    elseif action == "phi_powers"
        n = Int(get(payload, "n", 12))
        p = phi_powers(n)
        return Dict("ok" => true, "powers" => p, "sum" => sum(p), "lang" => "julia")
    elseif action == "multi_fft_embed"
        raw = Vector{UInt8}(codeunits(String(get(payload, "text", ""))))
        v = multi_fft_embed(raw)
        return Dict("ok" => true, "dim" => length(v), "embedding" => v, "lang" => "julia")
    elseif action == "matmul_train_step"
        W = Matrix{Float64}(hcat(payload["W"]...)')  # rows
        # accept list-of-lists
        W = reduce(hcat, payload["W"])' |> Matrix{Float64}
        X = reduce(hcat, payload["X"])' |> Matrix{Float64}
        Y = reduce(hcat, payload["Y"])' |> Matrix{Float64}
        # if row-major lists: convert properly
        W = _lol_to_mat(payload["W"])
        X = _lol_to_mat(payload["X"])
        Y = _lol_to_mat(payload["Y"])
        lr = Float64(get(payload, "lr", 1e-3))
        r = matmul_train_step(W, X, Y; lr=lr)
        r["ok"] = true
        r["lang"] = "julia"
        r["W"] = [collect(r["W"][i, :]) for i in 1:size(r["W"], 1)]
        return r
    else
        return Dict("ok" => false, "error" => "unknown action $action")
    end
end

function _lol_to_mat(lol)
    rows = length(lol)
    cols = length(lol[1])
    M = zeros(rows, cols)
    for i in 1:rows
        for j in 1:cols
            M[i, j] = Float64(lol[i][j])
        end
    end
    return M
end

end # module
