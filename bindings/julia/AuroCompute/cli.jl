#!/usr/bin/env julia
# JSON stdin/stdout CLI for AuroCompute
include(joinpath(@__DIR__, "src", "AuroCompute.jl"))
using .AuroCompute
using JSON

function main()
    raw = read(stdin, String)
    isempty(strip(raw)) && (println(JSON.json(Dict("ok"=>false,"error"=>"empty stdin"))); return)
    req = JSON.parse(raw)
    action = String(get(req, "action", "health"))
    payload = Dict{String,Any}(String(k) => v for (k, v) in pairs(req))
    delete!(payload, "action")
    try
        out = run_action(action, payload)
        println(JSON.json(out))
    catch e
        println(JSON.json(Dict("ok"=>false, "error"=>sprint(showerror, e), "lang"=>"julia")))
    end
end

main()
