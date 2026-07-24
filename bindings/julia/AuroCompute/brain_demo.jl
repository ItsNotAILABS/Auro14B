#!/usr/bin/env julia
include(joinpath(@__DIR__, "src", "AuroBrain.jl"))
using .AuroBrain

br = get_brain(max(Threads.nthreads(), 4); dim=64)
println("health=", brain_health(br))
c = brain_cycle!(br, "distributed spectral resonance across virtual physics cores"; steps=4)
t = c["thought"]
println("focus=$(c["focus_band"]) r=$(t["mean_kuramoto_r"]) R=$(t["mean_resonance"]) E=$(t["mean_energy"]) cores=$(t["n_cores"]) thr=$(t["threads"]) tick=$(t["tick"])")
println("advice=$(c["advice_to_ai"])")
