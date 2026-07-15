#!/usr/bin/env julia
"""
ZenodoSpectralSDK CLI — Command-line interface for Zenodo spectral intelligence.

Usage:
    julia cli.jl <action> [options...]

Actions:
    health          - Check SDK health status
    search          - Search Zenodo datasets
    popular         - List popular spectral datasets (curated)
    embed           - Generate transformer embedding from JSON record (stdin)
    match           - Match two spectral records (stdin JSON)
    anomaly         - Detect anomalies in amplitude data (stdin JSON)
    resonance       - Compute resonance analysis (stdin JSON)
    fingerprint     - Generate spectral fingerprint (stdin JSON)
    protocol        - Run intelligence protocol (stdin JSON)

Examples:
    julia cli.jl health
    julia cli.jl popular
    julia cli.jl search --query="spectral" --size=5
    echo '{"frequency":[1,2,3],"amplitude":[0.5,0.8,0.3]}' | julia cli.jl embed
"""

# Activate project
import Pkg
Pkg.activate(@__DIR__)

using JSON

include("src/ZenodoSpectralSDK.jl")
using .ZenodoSpectralSDK

function main()
    if isempty(ARGS)
        println(stderr, "Usage: julia cli.jl <action> [options...]")
        println(stderr, "Run 'julia cli.jl health' for a quick check.")
        exit(1)
    end

    action = ARGS[1]

    # Parse optional arguments
    options = Dict{String,String}()
    for arg in ARGS[2:end]
        if startswith(arg, "--")
            parts = split(arg[3:end], "=", limit=2)
            if length(parts) == 2
                options[parts[1]] = parts[2]
            else
                options[parts[1]] = "true"
            end
        end
    end

    result = if action == "health"
        dispatch_action(Dict("action" => "health"))

    elseif action == "search"
        query = get(options, "query", "")
        size = parse(Int, get(options, "size", "10"))
        dispatch_action(Dict(
            "action" => "search_datasets",
            "query" => query,
            "size" => size,
            "token" => get(options, "token", ""),
        ))

    elseif action == "popular"
        dispatch_action(Dict("action" => "curated_popular"))

    elseif action == "fetch"
        record_id = parse(Int, get(options, "id", "0"))
        dispatch_action(Dict(
            "action" => "fetch_record",
            "record_id" => record_id,
            "token" => get(options, "token", ""),
        ))

    elseif action in ["embed", "match", "anomaly", "resonance", "fingerprint", "protocol"]
        # Read JSON from stdin
        input_json = read(stdin, String)
        request = JSON.parse(input_json)

        # Map CLI action to dispatch action
        dispatch_map = Dict(
            "embed" => "transformer_embed",
            "match" => "match",
            "anomaly" => "anomaly_detect",
            "resonance" => "resonance",
            "fingerprint" => "fingerprint",
            "protocol" => "protocol",
        )

        request["action"] = dispatch_map[action]
        dispatch_action(request)

    else
        Dict("ok" => false, "data" => Dict("error" => "unknown action: $action"))
    end

    # Output as JSON
    println(JSON.json(result, 2))
end

main()
