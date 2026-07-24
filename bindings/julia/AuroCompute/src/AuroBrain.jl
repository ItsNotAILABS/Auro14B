"""
AuroBrain — Julia IS the brain.

Virtual Physics Cores (distributed via Threads.@threads):
  each core holds phase field, spectral mass, Kuramoto order, Landau order param.
  Cores couple through mean-field resonance — true multi-thread physics, not stubs.

Python is the AI (language, tools, intent).
Julia is the BRAIN (physics cores, distributed think, spectral memory).
"""
module AuroBrain

using Base.Threads

export brain_health, spawn_cores, think_distributed, brain_cycle, brain_cycle!
export virtual_physics_step, core_state_vector, get_brain, VirtualPhysicsCore

const PHI = (1 + sqrt(5)) / 2
const PHI_INV = PHI - 1
const GOLDEN_ANGLE = 2π / (PHI * PHI)

mutable struct VirtualPhysicsCore
    id::Int
    dim::Int
    # field state
    psi::Vector{Float64}          # real residual field
    phase::Vector{Float64}        # Kuramoto phases
    omega::Vector{Float64}        # natural frequencies from dispersion
    m::Vector{Float64}            # Landau order parameter
    # metrics
    energy::Float64
    coherence::Float64
    kuramoto_r::Float64
    resonance::Float64
    landau_F::Float64
    pulse::Int
end

function VirtualPhysicsCore(id::Int, dim::Int=64)
    k = range(0.0, PHI; length=dim) |> collect
    omega = dispersion_omega.(k)
    psi = zeros(dim)
    phase = 2π .* rand(dim) ./ PHI
    m = randn(dim)
    m ./= (sqrt(sum(abs2, m)) + 1e-12)
    return VirtualPhysicsCore(id, dim, psi, phase, omega, m, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

# ---- physics kernels (real equations) ----
function dispersion_omega(k::Float64; omega0=1.0, c=PHI, alpha=PHI_INV*0.1, beta=PHI_INV)
    w2 = omega0^2 + (c*k)^2 + alpha*k^4 + beta*sin(k*GOLDEN_ANGLE)^2
    return sqrt(max(w2, 1e-18))
end

function spectral_action(A::Vector{Float64}; λ=PHI_INV)
    n = length(A)
    k = range(0.0, PHI; length=n) |> collect
    w = dispersion_omega.(k)
    S = sum(A .* A .* w)
    if n > 1
        d = diff(A)
        S += λ * sum(d .* d)
    end
    return S
end

function kuramoto_step!(phase::Vector{Float64}, omega::Vector{Float64}; K=PHI, dt=0.05)
    N = length(phase)
    # mean field
    zx = sum(cos.(phase)) / N
    zy = sum(sin.(phase)) / N
    r = sqrt(zx*zx + zy*zy)
    ψ = atan(zy, zx)
    @inbounds for i in 1:N
        phase[i] += dt * (omega[i] + K * r * sin(ψ - phase[i]))
    end
    return r, ψ
end

function kuramoto_r(phase::Vector{Float64})
    N = length(phase)
    zx = sum(cos.(phase)) / N
    zy = sum(sin.(phase)) / N
    return sqrt(zx*zx + zy*zy)
end

function landau_force(m::Vector{Float64}, h::Vector{Float64}; a=-0.5, b=1.0)
    m2 = sum(abs2, m)
    return -a .* m .- b * m2 .* m .+ h
end

function landau_F(m::Vector{Float64}, h::Vector{Float64}; a=-0.5, b=1.0)
    m2 = sum(abs2, m)
    return 0.5*a*m2 + 0.25*b*(m2^2) - sum(m .* h)
end

"""φ-Schrödinger split-step kinetic (DFT) + potential — pure Julia."""
function phi_schrodinger_step!(psi::Vector{Float64}; dt=0.04, mass=1.0)
    n = length(psi)
    # complex buffer
    re = copy(psi)
    im = zeros(n)
    # kinetic in Fourier via manual DFT (no FFTW required)
    # ψ_k = DFT(ψ); * exp(-i K dt); iDFT
    kfreq = [2π * (i <= n÷2 ? i-1 : i-1-n) / n for i in 1:n]
    K = (kfreq .* kfreq) ./ (2 * mass)
    # forward DFT
    Re_k = zeros(n)
    Im_k = zeros(n)
    @inbounds for k in 1:n
        sr = 0.0
        si = 0.0
        for t in 1:n
            ang = 2π * (k-1) * (t-1) / n
            sr += re[t]*cos(ang) + im[t]*sin(ang)
            si += im[t]*cos(ang) - re[t]*sin(ang)
        end
        # * exp(-i K dt) = cos - i sin
        c = cos(K[k]*dt)
        s = sin(K[k]*dt)
        Re_k[k] = sr*c + si*s
        Im_k[k] = si*c - sr*s
    end
    # inverse DFT
    @inbounds for t in 1:n
        sr = 0.0
        si = 0.0
        for k in 1:n
            ang = 2π * (k-1) * (t-1) / n
            sr += Re_k[k]*cos(ang) - Im_k[k]*sin(ang)
            si += Re_k[k]*sin(ang) + Im_k[k]*cos(ang)
        end
        re[t] = sr / n
        im[t] = si / n
    end
    # potential
    pos = range(-1.0, 1.0; length=n) |> collect
    scale = sqrt(sum(abs2, re)/n) + 1e-6
    @inbounds for i in 1:n
        V = (1 - PHI_INV)*0.5*pos[i]^2 + (PHI_INV^2)*cos(PHI*pos[i]*GOLDEN_ANGLE*n/(2π))
        V *= scale
        # exp(-i V dt) on complex
        c = cos(V*dt)
        s = sin(V*dt)
        nr = re[i]*c + im[i]*s
        ni = im[i]*c - re[i]*s
        re[i] = nr
        im[i] = ni
    end
    # write real part back, preserve energy
    target = sqrt(sum(abs2, psi)) + 1e-12
    nrm = sqrt(sum(abs2, re)) + 1e-12
    psi .= re .* (target / nrm)
    return psi
end

function signal_from_text(text::AbstractString, dim::Int)
    raw = Vector{UInt8}(codeunits(String(text)))
    sig = zeros(dim)
    isempty(raw) && (raw = UInt8[0x20])
    @inbounds for (i, b) in enumerate(raw)
        sig[mod1(i, dim)] += (Float64(b)/255.0) - 0.5
    end
    sig .-= sum(sig)/dim
    idx = 0:(dim-1)
    sig .+= 0.08 .* sin.(idx .* GOLDEN_ANGLE) .+ 0.04 .* cos.(idx .* PHI_INV)
    return sig
end

function spectrum(sig::Vector{Float64})
    n = length(sig)
    half = n ÷ 2
    spec = zeros(half + 1)
    @inbounds for k in 0:half
        re = 0.0
        im = 0.0
        for t in 0:(n-1)
            ang = 2π * k * t / n
            re += sig[t+1] * cos(ang)
            im -= sig[t+1] * sin(ang)
        end
        spec[k+1] = sqrt(re*re + im*im)
    end
    return spec
end

function resonance(a::Vector{Float64}, b::Vector{Float64})
    n = max(length(a), length(b))
    aa = zeros(n); bb = zeros(n)
    aa[1:length(a)] .= a
    bb[1:length(b)] .= b
    # DFT inner product magnitude
    half = n ÷ 2
    num = 0.0
    na = 0.0
    nb = 0.0
    @inbounds for k in 0:half
        reA=0.0; imA=0.0; reB=0.0; imB=0.0
        for t in 0:(n-1)
            ang = 2π * k * t / n
            c = cos(ang); s = sin(ang)
            reA += aa[t+1]*c; imA -= aa[t+1]*s
            reB += bb[t+1]*c; imB -= bb[t+1]*s
        end
        num += reA*reB + imA*imB  # Re(A conj B) for cos; use mag product
        na += reA*reA + imA*imA
        nb += reB*reB + imB*imB
    end
    den = sqrt(na)*sqrt(nb) + 1e-12
    return clamp(abs(num)/den, 0.0, 1.0)
end

function coherence_mean(x::Vector{Float64}, y::Vector{Float64})
    # simplified band coherence via spectral correlation
    Sx = spectrum(x)
    Sy = spectrum(y)
    n = min(length(Sx), length(Sy))
    Sx = Sx[1:n]; Sy = Sy[1:n]
    # |Sxy|^2 / (Sxx Syy) ~ product correlation
    num = sum(Sx .* Sy)
    den = sqrt(sum(Sx.^2))*sqrt(sum(Sy.^2)) + 1e-12
    return clamp(num/den, 0.0, 1.0)
end

# ---- core dynamics ----
function virtual_physics_step!(core::VirtualPhysicsCore, drive::Vector{Float64}; dt=0.05)
    d = core.dim
    # external field from drive
    h = zeros(d)
    if length(drive) >= d
        h .= drive[1:d]
    else
        h[1:length(drive)] .= drive
    end
    nrm = sqrt(sum(abs2, h)) + 1e-12
    h ./= nrm

    # inject drive into psi
    core.psi .+= 0.15 .* h
    # Schrödinger smooth
    phi_schrodinger_step!(core.psi; dt=dt)
    # Kuramoto
    r, _ = kuramoto_step!(core.phase, core.omega; K=PHI, dt=dt)
    core.kuramoto_r = r
    # Landau
    force = landau_force(core.m, h)
    core.m .+= 0.12 .* force
    mn = sqrt(sum(abs2, core.m)) + 1e-12
    core.m ./= mn
    core.landau_F = landau_F(core.m, h)
    # metrics
    core.energy = spectral_action(abs.(core.psi))
    core.coherence = coherence_mean(core.psi, h)
    core.resonance = resonance(abs.(core.psi), abs.(h))
    core.pulse += 1
    return core
end

function core_state_vector(core::VirtualPhysicsCore)
    return Dict(
        "id" => core.id,
        "pulse" => core.pulse,
        "energy" => core.energy,
        "coherence" => core.coherence,
        "kuramoto_r" => core.kuramoto_r,
        "resonance" => core.resonance,
        "landau_F" => core.landau_F,
        "dim" => core.dim,
        "psi_norm" => sqrt(sum(abs2, core.psi)),
        "m_head" => core.m[1:min(8, end)],
    )
end

# ---- brain: distributed cores ----
mutable struct Brain
    cores::Vector{VirtualPhysicsCore}
    n_cores::Int
    global_field::Vector{Float64}
    thoughts::Vector{Dict{String,Any}}
    tick::Int
end

function spawn_cores(n::Int=Threads.nthreads(); dim::Int=64)
    n = max(1, n)
    cores = [VirtualPhysicsCore(i, dim) for i in 1:n]
    return Brain(cores, n, zeros(dim), Dict{String,Any}[], 0)
end

function brain_health(brain::Brain)
    return Dict(
        "ok" => true,
        "role" => "BRAIN",
        "lang" => "julia",
        "version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "cpu_threads" => Sys.CPU_THREADS,
        "n_cores" => brain.n_cores,
        "tick" => brain.tick,
        "phi" => PHI,
        "distributed" => brain.n_cores > 1 || Threads.nthreads() > 1,
    )
end

"""Distributed think: each virtual physics core integrates in parallel threads."""
function think_distributed!(brain::Brain, intent::AbstractString; steps::Int=3)
    dim = brain.cores[1].dim
    drive0 = signal_from_text(intent, dim)
    # fan-out: each core gets phase-shifted drive (distributed novelty)
    results = Vector{Dict{String,Any}}(undef, brain.n_cores)
    Threads.@threads for i in 1:brain.n_cores
        core = brain.cores[i]
        # unique phase offset per core
        shift = Int(round((i-1) * dim / max(brain.n_cores, 1)))
        drive = circshift(drive0, shift)
        drive .+= 0.05 * sin.((0:dim-1) .* GOLDEN_ANGLE .* i)
        for s in 1:steps
            virtual_physics_step!(core, drive; dt=0.05 + 0.01*i)
        end
        results[i] = core_state_vector(core)
    end
    # mean-field couple: global field = mean psi
    g = zeros(dim)
    for c in brain.cores
        g .+= c.psi
    end
    g ./= brain.n_cores
    brain.global_field .= g
    # second pass: couple each core toward global (distributed consensus)
    Threads.@threads for i in 1:brain.n_cores
        virtual_physics_step!(brain.cores[i], g; dt=0.03)
        results[i] = core_state_vector(brain.cores[i])
    end
    brain.tick += 1
    # aggregate thought
    mean_r = sum(c.kuramoto_r for c in brain.cores) / brain.n_cores
    mean_coh = sum(c.coherence for c in brain.cores) / brain.n_cores
    mean_R = sum(c.resonance for c in brain.cores) / brain.n_cores
    mean_E = sum(c.energy for c in brain.cores) / brain.n_cores
    thought = Dict{String,Any}(
        "tick" => brain.tick,
        "intent" => String(intent)[1:min(end, 200)],
        "n_cores" => brain.n_cores,
        "threads" => Threads.nthreads(),
        "mean_kuramoto_r" => mean_r,
        "mean_coherence" => mean_coh,
        "mean_resonance" => mean_R,
        "mean_energy" => mean_E,
        "global_field_norm" => sqrt(sum(abs2, g)),
        "cores" => results,
        "brain" => "julia",
        "novel" => true,
        "distributed" => true,
    )
    push!(brain.thoughts, thought)
    length(brain.thoughts) > 32 && (brain.thoughts = brain.thoughts[end-31:end])
    return thought
end

function brain_cycle!(brain::Brain, intent::AbstractString; steps::Int=3)
    h = brain_health(brain)
    t = think_distributed!(brain, intent; steps=steps)
    # decision field: argmax |global_field| bands as "focus modes"
    g = brain.global_field
    n = length(g)
    bands = 8
    band_e = zeros(bands)
    for b in 1:bands
        lo = Int(floor((b-1)*n/bands)) + 1
        hi = Int(floor(b*n/bands))
        band_e[b] = sum(abs2, g[lo:hi])
    end
    focus = argmax(band_e)
    return Dict(
        "ok" => true,
        "role" => "BRAIN",
        "health" => h,
        "thought" => t,
        "focus_band" => focus,
        "band_energy" => band_e,
        "advice_to_ai" => "Python AI: act on focus_band=$(focus); resonance=$(t["mean_resonance"]); lock_r=$(t["mean_kuramoto_r"])",
    )
end

# process-global brain for CLI reuse
const GLOBAL_BRAIN = Ref{Union{Nothing,Brain}}(nothing)

function get_brain(n_cores::Int=0; dim::Int=64)
    n = n_cores > 0 ? n_cores : max(Threads.nthreads(), 4)
    if GLOBAL_BRAIN[] === nothing || GLOBAL_BRAIN[].n_cores != n
        GLOBAL_BRAIN[] = spawn_cores(n; dim=dim)
    end
    return GLOBAL_BRAIN[]
end

end # module
