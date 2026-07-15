#!/usr/bin/env julia
"""
MESIE ZenodoSpectralSDK — One-Shot Launcher

Run this single file to automatically:
1. Verify system dependencies (bash, git)
2. Install all required Julia dependencies
3. Activate the SDK environment
4. Launch the interactive Research OS terminal

Usage:
    julia launch.jl

That's it. Everything else is handled automatically.

If system dependencies are missing, run the installer first:
    chmod +x install.sh && ./install.sh
"""

println()
println("╔══════════════════════════════════════════════════════════════════╗")
println("║   MESIE — Multi-Element Spectral Intelligence Engine            ║")
println("║   ZenodoSpectralSDK Research OS                                 ║")
println("║   v0.1.0                                                        ║")
println("╚══════════════════════════════════════════════════════════════════╝")
println()

# --- Step 0: Verify system dependencies ---
println("🔍 Checking system dependencies...")

function check_command(cmd::String, name::String)
    try
        result = read(`which $cmd`, String)
        version_str = try
            strip(read(`$cmd --version`, String))
        catch
            "installed"
        end
        println("  ✅ $name: found at $(strip(result))")
        return true
    catch
        return false
    end
end

sys_ok = true

# Check bash
if !check_command("bash", "bash")
    println("  ❌ bash: NOT FOUND")
    sys_ok = false
end

# Check git
if !check_command("git", "git")
    println("  ❌ git: NOT FOUND")
    sys_ok = false
end

if !sys_ok
    println()
    println("⚠️  Missing system dependencies!")
    println("   Run the full installer to fix this:")
    println()
    println("     chmod +x install.sh && ./install.sh")
    println()
    println("   Or install manually:")
    println("     • bash: apt-get install bash / brew install bash")
    println("     • git:  apt-get install git  / brew install git")
    println()

    # Attempt auto-install if running on a system with apt/brew
    print("   Attempt automatic installation? (y/N): ")
    answer = strip(readline())
    if lowercase(answer) == "y"
        println()
        println("   🔧 Attempting automatic installation...")
        try
            if Sys.islinux()
                if isfile("/usr/bin/apt-get") || success(`which apt-get`)
                    run(`sudo apt-get update -qq`)
                    run(`sudo apt-get install -y -qq bash git`)
                elseif success(`which dnf`)
                    run(`sudo dnf install -y bash git`)
                elseif success(`which apk`)
                    run(`sudo apk add --no-cache bash git`)
                else
                    error("No supported package manager found")
                end
            elseif Sys.isapple()
                if success(`which brew`)
                    run(`brew install bash git`)
                else
                    error("Homebrew not found. Install from https://brew.sh")
                end
            end
            println("   ✅ System dependencies installed successfully!")
        catch e
            println("   ❌ Auto-install failed: $e")
            println("   Please install bash and git manually, then re-run this script.")
            exit(1)
        end
    else
        println("   Exiting. Please install dependencies and try again.")
        exit(1)
    end
end

println("✅ System dependencies verified.")
println()

# --- Step 1: Auto-install Julia package dependencies ---
println("⏳ Initializing Julia environment...")

import Pkg

# Activate the project
project_dir = @__DIR__
Pkg.activate(project_dir)

# Install all dependencies automatically
println("📦 Installing Julia packages (first run may take a minute)...")
try
    Pkg.instantiate()
    Pkg.resolve()
catch e
    println("⚠️  Dependency resolution needed, adding packages...")
    Pkg.add([
        Pkg.PackageSpec(name="HTTP"),
        Pkg.PackageSpec(name="JSON"),
        Pkg.PackageSpec(name="Downloads"),
    ])
    Pkg.resolve()
    Pkg.instantiate()
end

println("✅ All Julia packages installed.")
println()

# --- Step 2: Load the SDK ---
println("🔬 Loading ZenodoSpectralSDK modules...")
include(joinpath(project_dir, "src", "ZenodoSpectralSDK.jl"))
using .ZenodoSpectralSDK

println("✅ SDK loaded successfully.")
println()

# --- Step 3: Launch interactive REPL ---
include(joinpath(project_dir, "src", "repl.jl"))
launch_research_os()
