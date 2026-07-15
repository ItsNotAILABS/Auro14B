"""
MESIE Research OS — Interactive Terminal Interface

A full research platform inside the Julia REPL for:
- Browsing and discovering Zenodo scientific datasets
- Running spectral intelligence analysis
- Managing research workspaces and environments
- Transformer-based embedding and matching
- Intelligence protocol execution

This module provides the interactive terminal UI that launches
when users run `julia launch.jl`.
"""

using Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Research OS State
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct ResearchWorkspace
    name::String
    created::DateTime
    datasets::Vector{Dict}
    analyses::Vector{Dict}
    embeddings::Vector{Dict}
    notes::Vector{String}
end

function ResearchWorkspace(name::String)
    ResearchWorkspace(name, now(), Dict[], Dict[], Dict[], String[])
end

# Global state
const WORKSPACES = Dict{String,ResearchWorkspace}()
const CURRENT_WORKSPACE = Ref{String}("")
const SESSION_START = Ref{DateTime}(now())

# ═══════════════════════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════════════════════

function print_header()
    println()
    println("┌──────────────────────────────────────────────────────────────────┐")
    println("│          🔬 MESIE Research OS — Spectral Intelligence            │")
    println("│              Powered by ZenodoSpectralSDK v0.1.0                 │")
    println("└──────────────────────────────────────────────────────────────────┘")
    println()
end

function print_divider()
    println("────────────────────────────────────────────────────────────────────")
end

function print_menu(title::String, options::Vector{Tuple{String,String}})
    println()
    print_divider()
    println("  $title")
    print_divider()
    for (key, desc) in options
        println("  [$key]  $desc")
    end
    print_divider()
    println()
end

function prompt(msg::String="mesie> ")
    print(msg)
    return strip(readline())
end

function status_line()
    ws = isempty(CURRENT_WORKSPACE[]) ? "none" : CURRENT_WORKSPACE[]
    elapsed = round(Int, (now() - SESSION_START[]).value / 1000)
    mins = div(elapsed, 60)
    secs = elapsed % 60
    println("  📂 Workspace: $ws  │  ⏱️  Session: $(mins)m $(secs)s  │  🧠 Julia $(VERSION)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Menu
# ═══════════════════════════════════════════════════════════════════════════════

function main_menu()
    print_menu("Main Menu", [
        ("1", "🌐 Browse Zenodo — Discover & search scientific datasets"),
        ("2", "🔬 Spectral Analysis — Run intelligence protocols on data"),
        ("3", "🧠 Transformer Lab — Embeddings, matching, fingerprinting"),
        ("4", "📂 Workspace — Manage research environments"),
        ("5", "📊 Popular Datasets — Curated high-impact datasets"),
        ("6", "⚡ Quick Analysis — Analyze data from file or input"),
        ("7", "🔧 System — Health check, settings, about"),
        ("q", "Exit Research OS"),
    ])
    status_line()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Zenodo Browser
# ═══════════════════════════════════════════════════════════════════════════════

function zenodo_browser()
    while true
        print_menu("🌐 Zenodo Dataset Browser", [
            ("1", "Search datasets by keyword"),
            ("2", "Fetch record by ID"),
            ("3", "Browse popular/trending datasets"),
            ("4", "Download dataset file"),
            ("5", "View record statistics"),
            ("b", "← Back to main menu"),
        ])

        choice = prompt("zenodo> ")

        if choice == "b" || choice == "back"
            return
        elseif choice == "1"
            zenodo_search()
        elseif choice == "2"
            zenodo_fetch()
        elseif choice == "3"
            zenodo_popular()
        elseif choice == "4"
            zenodo_download()
        elseif choice == "5"
            zenodo_stats()
        end
    end
end

function zenodo_search()
    println()
    query = prompt("  Search query: ")
    if isempty(query)
        println("  ⚠️  No query provided.")
        return
    end

    size_str = prompt("  Results per page (default 10): ")
    size = isempty(size_str) ? 10 : parse(Int, size_str)

    println()
    println("  🔍 Searching Zenodo for \"$query\"...")
    println()

    client = ZenodoClient()
    results = search_datasets(client; query=query, size=size)

    if haskey(results, "error")
        println("  ❌ Error: $(results["error"])")
        return
    end

    records = get(results, "records", Dict[])
    total = get(results, "total", 0)
    println("  Found $total results (showing $(length(records))):")
    println()

    for (i, r) in enumerate(records)
        title = get(r, "title", "Untitled")
        rid = get(r, "id", 0)
        views = get(get(r, "stats", Dict()), "views", 0)
        println("  [$i] #$rid — $title")
        println("      Views: $views | DOI: $(get(r, "doi", "n/a"))")
        println()
    end

    # Optionally save to workspace
    if !isempty(CURRENT_WORKSPACE[]) && !isempty(records)
        save = prompt("  Save results to workspace? (y/n): ")
        if lowercase(save) == "y"
            ws = WORKSPACES[CURRENT_WORKSPACE[]]
            append!(ws.datasets, records)
            println("  ✅ Saved $(length(records)) records to workspace '$(ws.name)'")
        end
    end
end

function zenodo_fetch()
    println()
    id_str = prompt("  Record ID: ")
    if isempty(id_str)
        println("  ⚠️  No ID provided.")
        return
    end

    record_id = parse(Int, id_str)
    println("  📥 Fetching record #$record_id...")

    client = ZenodoClient()
    record = fetch_record(client, record_id)

    if haskey(record, "error")
        println("  ❌ Error: $(record["error"])")
        return
    end

    println()
    println("  ╭─ Record #$(get(record, "id", 0)) ─────────────────────────────")
    println("  │ Title: $(get(record, "title", ""))")
    println("  │ DOI: $(get(record, "doi", ""))")
    println("  │ Date: $(get(record, "publication_date", ""))")
    println("  │ License: $(get(record, "license", ""))")
    println("  │ Type: $(get(record, "resource_type", ""))")
    creators = get(record, "creators", String[])
    if !isempty(creators)
        println("  │ Creators: $(join(creators, ", "))")
    end
    keywords = get(record, "keywords", String[])
    if !isempty(keywords)
        println("  │ Keywords: $(join(keywords, ", "))")
    end
    files = get(record, "files", Dict[])
    if !isempty(files)
        println("  │ Files ($(length(files))):")
        for f in files
            println("  │   • $(get(f, "key", "?")) ($(get(f, "size", 0)) bytes)")
        end
    end
    stats = get(record, "stats", Dict())
    println("  │ Views: $(get(stats, "views", 0)) | Downloads: $(get(stats, "downloads", 0))")
    println("  ╰──────────────────────────────────────────────────────────────")
    println()
end

function zenodo_popular()
    println()
    println("  🔍 Searching Zenodo for trending datasets...")

    client = ZenodoClient()
    results = list_popular_datasets(client; query="spectral signal time-series", top_n=10)

    if length(results) == 1 && haskey(results[1], "error")
        println("  ⚠️  Could not reach Zenodo API. Showing curated list instead.")
        results = POPULAR_SPECTRAL_DATASETS
    end

    println()
    println("  📈 Popular Datasets:")
    println()
    for (i, r) in enumerate(results)
        title = get(r, "title", "Untitled")
        views = get(get(r, "stats", Dict()), "views", get(r, "estimated_views", 0))
        rid = get(r, "id", get(r, "record_id", 0))
        println("  [$i] #$rid — $title ($(views) views)")
    end
    println()
end

function zenodo_download()
    println()
    id_str = prompt("  Record ID to download from: ")
    if isempty(id_str)
        return
    end

    record_id = parse(Int, id_str)
    println("  📥 Downloading first file from record #$record_id...")

    client = ZenodoClient()
    path = download_dataset_file(client, record_id)

    if isempty(path)
        println("  ❌ Download failed or no files available.")
    else
        println("  ✅ Downloaded to: $path")
    end
end

function zenodo_stats()
    println()
    id_str = prompt("  Record ID: ")
    if isempty(id_str)
        return
    end

    record_id = parse(Int, id_str)
    println("  📊 Fetching statistics for record #$record_id...")

    client = ZenodoClient()
    stats = get_record_stats(client, record_id)

    if haskey(stats, "error")
        println("  ❌ Error: $(stats["error"])")
        return
    end

    println()
    println("  ╭─ Statistics for Record #$record_id ────────────────────────")
    println("  │ Views:            $(get(stats, "views", 0))")
    println("  │ Downloads:        $(get(stats, "downloads", 0))")
    println("  │ Unique Views:     $(get(stats, "unique_views", 0))")
    println("  │ Unique Downloads: $(get(stats, "unique_downloads", 0))")
    println("  │ Version Views:    $(get(stats, "version_views", 0))")
    println("  │ Version Downloads: $(get(stats, "version_downloads", 0))")
    println("  ╰──────────────────────────────────────────────────────────────")
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Spectral Analysis
# ═══════════════════════════════════════════════════════════════════════════════

function spectral_analysis_menu()
    while true
        print_menu("🔬 Spectral Analysis Lab", [
            ("1", "Anomaly Detection — Find outliers in spectral data"),
            ("2", "Resonance Analysis — Identify natural frequencies"),
            ("3", "Pattern Classification — Classify spectral type"),
            ("4", "Full Protocol — Run complete intelligence analysis"),
            ("5", "Generate sample data — Create test spectral data"),
            ("b", "← Back to main menu"),
        ])

        choice = prompt("analysis> ")

        if choice == "b" || choice == "back"
            return
        elseif choice == "1"
            run_anomaly_detection()
        elseif choice == "2"
            run_resonance_analysis()
        elseif choice == "3"
            run_classification()
        elseif choice == "4"
            run_full_protocol()
        elseif choice == "5"
            generate_sample_data()
        end
    end
end

function generate_sample_data()
    println()
    println("  📐 Sample Data Generator")
    println("  Choose signal type:")
    println("    [1] Sine wave (harmonic)")
    println("    [2] Multi-frequency (narrowband)")
    println("    [3] Random noise (broadband)")
    println("    [4] Earthquake-like (damped oscillation)")
    println()

    choice = prompt("  Signal type: ")
    n_str = prompt("  Number of points (default 200): ")
    n = isempty(n_str) ? 200 : parse(Int, n_str)

    freq = collect(range(0.1, stop=100.0, length=n))
    amp = if choice == "1"
        sin.(freq .* 0.3) .+ 0.5
    elseif choice == "2"
        sin.(freq .* 0.1) .+ 0.5 .* sin.(freq .* 0.5) .+ 0.3 .* sin.(freq .* 1.2)
    elseif choice == "3"
        randn(n)
    elseif choice == "4"
        exp.(-freq .* 0.02) .* sin.(freq .* 2.0)
    else
        sin.(freq .* 0.3) .+ 0.5
    end

    record = Dict("frequency" => freq, "amplitude" => amp)

    println()
    println("  ✅ Generated $(n)-point spectral record")
    println("     Frequency range: $(round(freq[1], digits=2)) — $(round(freq[end], digits=2)) Hz")
    println("     Amplitude range: $(round(minimum(amp), digits=3)) — $(round(maximum(amp), digits=3))")

    # Save to workspace
    if !isempty(CURRENT_WORKSPACE[])
        ws = WORKSPACES[CURRENT_WORKSPACE[]]
        push!(ws.datasets, record)
        println("  📂 Saved to workspace '$(ws.name)' (dataset #$(length(ws.datasets)))")
    end

    # Offer immediate analysis
    println()
    run_analysis = prompt("  Run analysis on this data? (y/n): ")
    if lowercase(run_analysis) == "y"
        _analyze_record(record)
    end
end

function run_anomaly_detection()
    record = _get_or_create_record()
    if record === nothing; return; end

    amp = Float64.(get(record, "amplitude", Float64[]))
    if isempty(amp)
        println("  ⚠️  No amplitude data found.")
        return
    end

    println()
    println("  🔍 Running anomaly detection...")
    intel = SpectralIntelligence()
    result = detect_anomalies(intel, amp)

    println()
    println("  ╭─ Anomaly Detection Results ─────────────────────────────────")
    println("  │ Method: $(result["method"])")
    println("  │ Threshold: $(result["threshold"])")
    println("  │ Anomalies Found: $(result["n_anomalies"])")
    if result["n_anomalies"] > 0
        println("  │ Locations: $(result["anomalies"][1:min(10, end)])")
        println("  │ Severities: $(result["severity"][1:min(10, end)])")
        println("  │ Max Score: $(round(maximum(result["scores"]), digits=3))")
    end
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    _save_analysis("anomaly_detection", result)
end

function run_resonance_analysis()
    record = _get_or_create_record()
    if record === nothing; return; end

    freq = Float64.(get(record, "frequency", Float64[]))
    amp = Float64.(get(record, "amplitude", Float64[]))

    if isempty(freq) || isempty(amp)
        println("  ⚠️  Need both frequency and amplitude data.")
        return
    end

    println()
    println("  🔍 Running resonance analysis...")
    intel = SpectralIntelligence()
    result = compute_resonance(intel, freq, amp)

    println()
    println("  ╭─ Resonance Analysis Results ────────────────────────────────")
    println("  │ Peaks Found: $(result["n_peaks"])")
    println("  │ Dominant Frequency: $(round(result["dominant_frequency"], digits=2)) Hz")
    println("  │ Mean Q-Factor: $(round(result["mean_q_factor"], digits=3))")
    if !isempty(result["peaks"])
        println("  │ Top Peaks:")
        for (i, p) in enumerate(result["peaks"][1:min(5, end)])
            println("  │   #$i: $(round(p["frequency"], digits=2)) Hz (Q=$(round(p["q_factor"], digits=2)))")
        end
    end
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    _save_analysis("resonance", result)
end

function run_classification()
    record = _get_or_create_record()
    if record === nothing; return; end

    println()
    println("  🔍 Classifying spectral pattern...")
    intel = SpectralIntelligence()
    result = intelligence_protocol(intel, record; action="classify")

    println()
    println("  ╭─ Pattern Classification ────────────────────────────────────")
    println("  │ Class: $(result["class"])")
    println("  │ Confidence: $(round(result["confidence"] * 100, digits=1))%")
    features = get(result, "features", Dict())
    if !isempty(features)
        println("  │ Features:")
        println("  │   Energy: $(round(get(features, "energy", 0.0), digits=3))")
        println("  │   Peak Ratio: $(round(get(features, "peak_ratio", 0.0), digits=3))")
        println("  │   Spectral Flatness: $(round(get(features, "spectral_flatness", 0.0), digits=3))")
    end
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    _save_analysis("classification", result)
end

function run_full_protocol()
    record = _get_or_create_record()
    if record === nothing; return; end

    println()
    println("  🧠 Running full intelligence protocol (Level 3)...")
    println()

    intel = SpectralIntelligence(protocol_level=3)
    result = intelligence_protocol(intel, record; action="analyze")

    println("  ╭─ Intelligence Protocol Results ─────────────────────────────")
    println("  │ Protocol Level: $(result["protocol_level"])")
    println("  │ Data Points: $(result["n_points"])")
    println("  │ Spectral Energy: $(round(result["spectral_energy"], digits=3))")
    println("  │ Peak Frequency: $(round(result["peak_frequency"], digits=2)) Hz")
    println("  │")

    anomalies = result["anomaly_detection"]
    println("  │ 🔴 Anomalies: $(anomalies["n_anomalies"]) detected")

    resonance = result["resonance_analysis"]
    println("  │ 🔵 Resonance: $(resonance["n_peaks"]) peaks, dominant=$(round(resonance["dominant_frequency"], digits=2)) Hz")

    emb = result["embedding"]
    println("  │ 🟢 Embedding: $(length(emb))-dimensional vector (norm=$(round(norm(emb), digits=3)))")
    println("  │")
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    _save_analysis("full_protocol", result)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Transformer Lab
# ═══════════════════════════════════════════════════════════════════════════════

function transformer_lab_menu()
    while true
        print_menu("🧠 Transformer Lab", [
            ("1", "Generate Embedding — Create spectral vector representation"),
            ("2", "Compare Records — Match two spectral signals"),
            ("3", "Fingerprint — Create compact spectral identity"),
            ("4", "Batch Embed — Process multiple records"),
            ("b", "← Back to main menu"),
        ])

        choice = prompt("transformer> ")

        if choice == "b" || choice == "back"
            return
        elseif choice == "1"
            run_embedding()
        elseif choice == "2"
            run_matching()
        elseif choice == "3"
            run_fingerprint()
        elseif choice == "4"
            run_batch_embed()
        end
    end
end

function run_embedding()
    record = _get_or_create_record()
    if record === nothing; return; end

    println()
    println("  🧠 Generating transformer embedding...")

    result = transformer_pipeline(record)

    println()
    println("  ╭─ Transformer Embedding ─────────────────────────────────────")
    println("  │ Model: $(result["model"])")
    println("  │ Dimension: $(result["d_model"])")
    println("  │ Tokens: $(result["n_tokens"])")
    println("  │ Heads: $(get(result, "n_heads", "N/A"))")
    println("  │ Layers: $(get(result, "n_layers", "N/A"))")
    println("  │ Input Points: $(get(result, "n_points", 0))")
    emb = result["embedding"]
    println("  │ Embedding Norm: $(round(norm(emb), digits=4))")
    println("  │ Embedding (first 8): $(round.(emb[1:min(8, end)], digits=4))")
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    if !isempty(CURRENT_WORKSPACE[])
        ws = WORKSPACES[CURRENT_WORKSPACE[]]
        push!(ws.embeddings, result)
        println("  📂 Saved embedding to workspace (total: $(length(ws.embeddings)))")
    end
end

function run_matching()
    println()
    println("  Provide two spectral records for matching.")
    println("  Record A:")
    record_a = _get_or_create_record()
    if record_a === nothing; return; end

    println("  Record B (generating different signal for comparison):")
    n = length(get(record_a, "frequency", Float64[]))
    if n == 0; n = 100; end
    freq = collect(range(0.1, stop=100.0, length=n))
    record_b = Dict("frequency" => freq, "amplitude" => randn(n) .* 0.5)

    println()
    println("  🔬 Computing spectral match...")
    intel = SpectralIntelligence()
    result = spectral_match(intel, record_a, record_b)

    println()
    println("  ╭─ Spectral Match Results ────────────────────────────────────")
    println("  │ Composite Score: $(round(result["composite_score"], digits=4))")
    println("  │ Embedding Similarity: $(round(result["embedding_similarity"], digits=4))")
    println("  │ Confidence: $(result["confidence"])")
    println("  │ Points Compared: $(result["n_compared"])")
    metrics = result["metrics"]
    println("  │ Metrics:")
    println("  │   Cosine (embedding): $(round(get(metrics, "cosine_embedding", 0.0), digits=4))")
    println("  │   Cosine (raw): $(round(get(metrics, "cosine_raw", 0.0), digits=4))")
    println("  │   RMSE: $(round(get(metrics, "rmse", 0.0), digits=4))")
    println("  │   Pearson: $(round(get(metrics, "pearson", 0.0), digits=4))")
    println("  │ Model: $(result["model"])")
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    _save_analysis("match", result)
end

function run_fingerprint()
    record = _get_or_create_record()
    if record === nothing; return; end

    println()
    println("  🔑 Generating spectral fingerprint...")

    intel = SpectralIntelligence()
    result = spectral_fingerprint(intel, record)

    println()
    println("  ╭─ Spectral Fingerprint ──────────────────────────────────────")
    println("  │ Method: $(result["method"])")
    println("  │ Resolution: $(result["resolution"])")
    println("  │ Hash: $(result["hash"])")
    fp = result["fingerprint"]
    println("  │ Fingerprint ($(length(fp)) bits): $(join(string.(fp[1:min(32, end)]), ""))...")
    trad = result["traditional_fingerprint"]
    println("  │ Traditional ($(length(trad)) bits): $(join(string.(trad[1:min(32, end)]), ""))")
    println("  │ Runtime: $(result["runtime"])")
    println("  ╰──────────────────────────────────────────────────────────────")

    _save_analysis("fingerprint", result)
end

function run_batch_embed()
    println()
    n_str = prompt("  Number of records to generate and embed (default 5): ")
    n = isempty(n_str) ? 5 : parse(Int, n_str)

    println()
    println("  🧠 Batch embedding $n records...")
    println()

    model = SpectralTransformer()
    embeddings = Vector{Float64}[]

    for i in 1:n
        freq = collect(range(0.1, stop=100.0, length=200))
        amp = sin.(freq .* (0.1 * i)) .+ randn(200) .* 0.1
        record = Dict("frequency" => freq, "amplitude" => amp)

        result = transformer_pipeline(record; model=model)
        push!(embeddings, result["embedding"])
        println("  [$i/$n] Embedded — norm=$(round(norm(result["embedding"]), digits=3))")
    end

    # Compute pairwise similarities
    println()
    println("  📊 Pairwise Cosine Similarities:")
    println("       ", join(["  R$i " for i in 1:min(n, 6)]))
    for i in 1:min(n, 6)
        row = "  R$i  "
        for j in 1:min(n, 6)
            sim = dot(embeddings[i], embeddings[j]) / (norm(embeddings[i]) * norm(embeddings[j]) + 1e-12)
            row *= " $(round(sim, digits=2)) "
        end
        println(row)
    end
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Workspace Management
# ═══════════════════════════════════════════════════════════════════════════════

function workspace_menu()
    while true
        print_menu("📂 Research Workspace Manager", [
            ("1", "Create new workspace"),
            ("2", "Switch workspace"),
            ("3", "List workspaces"),
            ("4", "Workspace status"),
            ("5", "Add note to workspace"),
            ("6", "Export workspace summary"),
            ("b", "← Back to main menu"),
        ])

        choice = prompt("workspace> ")

        if choice == "b" || choice == "back"
            return
        elseif choice == "1"
            create_workspace()
        elseif choice == "2"
            switch_workspace()
        elseif choice == "3"
            list_workspaces()
        elseif choice == "4"
            workspace_status()
        elseif choice == "5"
            add_workspace_note()
        elseif choice == "6"
            export_workspace()
        end
    end
end

function create_workspace()
    println()
    name = prompt("  Workspace name: ")
    if isempty(name)
        println("  ⚠️  Name required.")
        return
    end

    if haskey(WORKSPACES, name)
        println("  ⚠️  Workspace '$name' already exists.")
        return
    end

    WORKSPACES[name] = ResearchWorkspace(name)
    CURRENT_WORKSPACE[] = name
    println("  ✅ Created and activated workspace '$name'")
end

function switch_workspace()
    if isempty(WORKSPACES)
        println("  ⚠️  No workspaces exist. Create one first.")
        return
    end

    println()
    for (i, (name, ws)) in enumerate(WORKSPACES)
        marker = name == CURRENT_WORKSPACE[] ? " ← current" : ""
        println("  [$i] $name ($(length(ws.datasets)) datasets, $(length(ws.analyses)) analyses)$marker")
    end
    println()

    name = prompt("  Workspace name: ")
    if haskey(WORKSPACES, name)
        CURRENT_WORKSPACE[] = name
        println("  ✅ Switched to '$name'")
    else
        println("  ⚠️  Workspace '$name' not found.")
    end
end

function list_workspaces()
    println()
    if isempty(WORKSPACES)
        println("  No workspaces yet. Create one from the menu.")
        return
    end

    println("  Research Workspaces:")
    println()
    for (name, ws) in WORKSPACES
        marker = name == CURRENT_WORKSPACE[] ? " ★" : ""
        println("  📂 $name$marker")
        println("     Created: $(Dates.format(ws.created, "yyyy-mm-dd HH:MM"))")
        println("     Datasets: $(length(ws.datasets)) | Analyses: $(length(ws.analyses)) | Embeddings: $(length(ws.embeddings))")
        println("     Notes: $(length(ws.notes))")
        println()
    end
end

function workspace_status()
    if isempty(CURRENT_WORKSPACE[])
        println("  ⚠️  No active workspace. Create or switch to one.")
        return
    end

    ws = WORKSPACES[CURRENT_WORKSPACE[]]
    println()
    println("  ╭─ Workspace: $(ws.name) ──────────────────────────────────────")
    println("  │ Created: $(Dates.format(ws.created, "yyyy-mm-dd HH:MM:SS"))")
    println("  │ Datasets: $(length(ws.datasets))")
    println("  │ Analyses: $(length(ws.analyses))")
    println("  │ Embeddings: $(length(ws.embeddings))")
    println("  │ Notes: $(length(ws.notes))")
    if !isempty(ws.notes)
        println("  │ Latest note: $(ws.notes[end])")
    end
    println("  ╰──────────────────────────────────────────────────────────────")
end

function add_workspace_note()
    if isempty(CURRENT_WORKSPACE[])
        println("  ⚠️  No active workspace.")
        return
    end

    println()
    note = prompt("  Note: ")
    if !isempty(note)
        ws = WORKSPACES[CURRENT_WORKSPACE[]]
        push!(ws.notes, "[$(Dates.format(now(), "HH:MM"))] $note")
        println("  ✅ Note added.")
    end
end

function export_workspace()
    if isempty(CURRENT_WORKSPACE[])
        println("  ⚠️  No active workspace.")
        return
    end

    ws = WORKSPACES[CURRENT_WORKSPACE[]]
    println()
    println("  ╭─ Workspace Export: $(ws.name) ───────────────────────────────")
    println("  │ Created: $(ws.created)")
    println("  │")
    println("  │ Datasets ($(length(ws.datasets))):")
    for (i, d) in enumerate(ws.datasets)
        title = get(d, "title", "Record #$i")
        println("  │   $i. $title")
    end
    println("  │")
    println("  │ Analyses ($(length(ws.analyses))):")
    for (i, a) in enumerate(ws.analyses)
        atype = get(a, "type", "unknown")
        println("  │   $i. $atype")
    end
    println("  │")
    println("  │ Notes:")
    for note in ws.notes
        println("  │   • $note")
    end
    println("  ╰──────────────────────────────────────────────────────────────")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Popular Datasets (Quick Access)
# ═══════════════════════════════════════════════════════════════════════════════

function popular_datasets_menu()
    println()
    println("  ⭐ Curated High-Impact Spectral Datasets on Zenodo")
    println()
    print_divider()

    for (i, ds) in enumerate(POPULAR_SPECTRAL_DATASETS)
        println("  [$i] $(ds["title"])")
        println("      ID: $(ds["record_id"]) | Views: ~$(ds["estimated_views"]) | Downloads: ~$(ds["estimated_downloads"])")
        println("      Use: $(ds["use_case"])")
        println("      Tags: $(join(ds["keywords"], ", "))")
        println()
    end

    print_divider()
    println()
    choice = prompt("  Fetch details for record # (or 'b' to go back): ")
    if choice == "b" || isempty(choice)
        return
    end

    idx = tryparse(Int, choice)
    if idx !== nothing && 1 <= idx <= length(POPULAR_SPECTRAL_DATASETS)
        record_id = POPULAR_SPECTRAL_DATASETS[idx]["record_id"]
        println()
        println("  📥 Fetching live details for record #$record_id...")
        client = ZenodoClient()
        record = fetch_record(client, record_id)
        if !haskey(record, "error")
            println("  ✅ Record retrieved successfully.")
            println("  Title: $(get(record, "title", ""))")
            println("  DOI: $(get(record, "doi", ""))")
            files = get(record, "files", Dict[])
            println("  Files: $(length(files))")
        else
            println("  ⚠️  Could not reach Zenodo: $(get(record, "error", "unknown"))")
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Quick Analysis
# ═══════════════════════════════════════════════════════════════════════════════

function quick_analysis_menu()
    println()
    println("  ⚡ Quick Analysis — Analyze spectral data from:")
    println("    [1] JSON file path")
    println("    [2] Manual input (frequency & amplitude arrays)")
    println("    [3] Generated sample data")
    println("    [b] Back")
    println()

    choice = prompt("quick> ")

    if choice == "b"
        return
    elseif choice == "1"
        quick_from_file()
    elseif choice == "2"
        quick_from_input()
    elseif choice == "3"
        generate_sample_data()
    end
end

function quick_from_file()
    println()
    path = prompt("  JSON file path: ")
    if isempty(path) || !isfile(path)
        println("  ⚠️  File not found: $path")
        return
    end

    try
        content = read(path, String)
        record = JSON.parse(content)
        println("  ✅ Loaded JSON record.")
        _analyze_record(record)
    catch e
        println("  ❌ Error reading file: $e")
    end
end

function quick_from_input()
    println()
    println("  Enter frequency values (comma-separated):")
    freq_str = prompt("  freq> ")
    println("  Enter amplitude values (comma-separated):")
    amp_str = prompt("  amp> ")

    try
        freq = Float64[parse(Float64, strip(s)) for s in split(freq_str, ",")]
        amp = Float64[parse(Float64, strip(s)) for s in split(amp_str, ",")]

        if length(freq) != length(amp)
            println("  ⚠️  Frequency and amplitude arrays must have same length.")
            return
        end

        record = Dict("frequency" => freq, "amplitude" => amp)
        _analyze_record(record)
    catch e
        println("  ❌ Parse error: $e")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 7. System
# ═══════════════════════════════════════════════════════════════════════════════

function system_menu()
    while true
        print_menu("🔧 System", [
            ("1", "Health Check — Verify SDK status"),
            ("2", "About — Version and capabilities"),
            ("3", "Julia Info — Runtime details"),
            ("b", "← Back to main menu"),
        ])

        choice = prompt("system> ")

        if choice == "b" || choice == "back"
            return
        elseif choice == "1"
            run_health_check()
        elseif choice == "2"
            show_about()
        elseif choice == "3"
            show_julia_info()
        end
    end
end

function run_health_check()
    println()
    println("  ⏳ Running health check...")
    result = health_check()

    println()
    println("  ╭─ Health Status ─────────────────────────────────────────────")
    println("  │ Status: $(result["status"]) ✅")
    println("  │ SDK: $(result["sdk"]) v$(result["version"])")
    println("  │ Runtime: Julia $(result["julia_version"])")
    println("  │ Threads: $(result["threads"])")
    println("  │ Capabilities:")
    for cap in result["capabilities"]
        println("  │   • $cap")
    end
    println("  ╰──────────────────────────────────────────────────────────────")
end

function show_about()
    println()
    println("  ╭──────────────────────────────────────────────────────────────╮")
    println("  │  MESIE — Multi-Element Spectral Intelligence Engine          │")
    println("  │  ZenodoSpectralSDK Research OS v0.1.0                        │")
    println("  │                                                              │")
    println("  │  A research platform for spectral intelligence, providing:   │")
    println("  │  • Scientific dataset discovery via Zenodo                   │")
    println("  │  • Transformer-based spectral embeddings                     │")
    println("  │  • Intelligence protocols for autonomous analysis            │")
    println("  │  • Research workspace management                             │")
    println("  │                                                              │")
    println("  │  Author: Alfredo Medina (ITSNOTAILabs)                       │")
    println("  │  License: Apache-2.0                                         │")
    println("  │  Repository: github.com/FreddyCreates/MESIE                  │")
    println("  ╰──────────────────────────────────────────────────────────────╯")
end

function show_julia_info()
    println()
    println("  ╭─ Julia Runtime ─────────────────────────────────────────────")
    println("  │ Version: $(VERSION)")
    println("  │ Threads: $(Threads.nthreads())")
    println("  │ Word Size: $(Sys.WORD_SIZE)-bit")
    println("  │ OS: $(Sys.KERNEL)")
    println("  │ CPU: $(Sys.CPU_THREADS) logical cores")
    println("  │ Free Memory: $(round(Sys.free_memory() / 1024^3, digits=2)) GB")
    println("  │ Total Memory: $(round(Sys.total_memory() / 1024^3, digits=2)) GB")
    println("  ╰──────────────────────────────────────────────────────────────")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════════

function _get_or_create_record()
    println()
    println("  Data source:")
    println("    [1] Generate sample data")
    println("    [2] From workspace (if available)")
    println("    [3] Enter manually")
    println()

    choice = prompt("  Source: ")

    if choice == "1"
        n = 200
        freq = collect(range(0.1, stop=100.0, length=n))
        amp = sin.(freq .* 0.3) .+ randn(n) .* 0.2
        println("  ✅ Generated 200-point sample record")
        return Dict("frequency" => freq, "amplitude" => amp)

    elseif choice == "2"
        if isempty(CURRENT_WORKSPACE[]) || isempty(WORKSPACES[CURRENT_WORKSPACE[]].datasets)
            println("  ⚠️  No datasets in current workspace. Generating sample instead.")
            n = 200
            freq = collect(range(0.1, stop=100.0, length=n))
            amp = sin.(freq .* 0.5) .+ randn(n) .* 0.1
            return Dict("frequency" => freq, "amplitude" => amp)
        end
        ws = WORKSPACES[CURRENT_WORKSPACE[]]
        println("  Available datasets: $(length(ws.datasets))")
        idx_str = prompt("  Dataset index (default 1): ")
        idx = isempty(idx_str) ? 1 : parse(Int, idx_str)
        idx = clamp(idx, 1, length(ws.datasets))
        return ws.datasets[idx]

    elseif choice == "3"
        println("  Enter frequency values (comma-separated):")
        freq_str = prompt("  freq> ")
        println("  Enter amplitude values (comma-separated):")
        amp_str = prompt("  amp> ")
        try
            freq = Float64[parse(Float64, strip(s)) for s in split(freq_str, ",")]
            amp = Float64[parse(Float64, strip(s)) for s in split(amp_str, ",")]
            return Dict("frequency" => freq, "amplitude" => amp)
        catch
            println("  ❌ Parse error. Using sample data.")
            n = 200
            freq = collect(range(0.1, stop=100.0, length=n))
            amp = sin.(freq .* 0.3) .+ randn(n) .* 0.2
            return Dict("frequency" => freq, "amplitude" => amp)
        end
    else
        # Default: generate sample
        n = 200
        freq = collect(range(0.1, stop=100.0, length=n))
        amp = sin.(freq .* 0.3) .+ randn(n) .* 0.2
        return Dict("frequency" => freq, "amplitude" => amp)
    end
end

function _analyze_record(record::Dict)
    println()
    println("  🧠 Running full spectral intelligence analysis...")
    println()

    intel = SpectralIntelligence(protocol_level=3)

    # Classification
    classify = intelligence_protocol(intel, record; action="classify")
    println("  📋 Classification: $(classify["class"]) ($(round(classify["confidence"] * 100, digits=1))% confidence)")

    # Summary
    summary = intelligence_protocol(intel, record; action="summarize")
    println("  📊 Points: $(summary["n_points"]) | Energy: $(round(summary["total_energy"], digits=3))")
    println("     Freq: $(round.(summary["frequency_range"], digits=2)) Hz")
    println("     Amp: $(round.(summary["amplitude_range"], digits=4))")

    # Embedding
    emb_result = transformer_pipeline(record)
    println("  🧠 Embedding: $(emb_result["d_model"])D vector generated")

    # Anomalies
    amp = Float64.(get(record, "amplitude", Float64[]))
    if !isempty(amp)
        anomalies = detect_anomalies(intel, amp)
        println("  🔴 Anomalies: $(anomalies["n_anomalies"]) detected")
    end

    println()
    println("  ✅ Analysis complete.")
end

function _save_analysis(analysis_type::String, result::Dict)
    if !isempty(CURRENT_WORKSPACE[])
        ws = WORKSPACES[CURRENT_WORKSPACE[]]
        push!(ws.analyses, merge(result, Dict("type" => analysis_type, "timestamp" => string(now()))))
        println("  📂 Saved to workspace '$(ws.name)'")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

function launch_research_os()
    SESSION_START[] = now()
    print_header()

    println("  Welcome to the MESIE Research OS!")
    println("  A spectral intelligence platform powered by Zenodo + Transformers.")
    println()
    println("  💡 Tip: Create a workspace first to save your research progress.")
    println()

    # Auto-create a default workspace
    default_ws = "research-$(Dates.format(now(), "yyyymmdd"))"
    WORKSPACES[default_ws] = ResearchWorkspace(default_ws)
    CURRENT_WORKSPACE[] = default_ws
    println("  📂 Auto-created workspace: '$default_ws'")

    while true
        main_menu()
        choice = prompt("mesie> ")

        if choice == "q" || choice == "quit" || choice == "exit"
            println()
            println("  👋 Exiting MESIE Research OS. Your workspace data is in memory.")
            println("     Thank you for using MESIE!")
            println()
            break
        elseif choice == "1"
            zenodo_browser()
        elseif choice == "2"
            spectral_analysis_menu()
        elseif choice == "3"
            transformer_lab_menu()
        elseif choice == "4"
            workspace_menu()
        elseif choice == "5"
            popular_datasets_menu()
        elseif choice == "6"
            quick_analysis_menu()
        elseif choice == "7"
            system_menu()
        else
            println("  ⚠️  Unknown option '$choice'. Enter a number 1-7 or 'q' to quit.")
        end
    end
end
