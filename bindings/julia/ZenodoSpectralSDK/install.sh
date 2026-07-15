#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# MESIE ZenodoSpectralSDK — Full Installation Script
#
# This script ensures all system-level dependencies are installed:
#   1. bash (shell)
#   2. git (version control)
#   3. Julia (language runtime)
#   4. Julia package dependencies (HTTP, JSON, etc.)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# After installation, launch the Research OS terminal with:
#   julia launch.jl
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

JULIA_MIN_VERSION="1.6"
JULIA_INSTALL_VERSION="1.10.4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║   MESIE — Multi-Element Spectral Intelligence Engine            ║"
    echo "║   ZenodoSpectralSDK Full Installer                              ║"
    echo "║   v0.1.0                                                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ─── Detect OS and Package Manager ────────────────────────────────────────────
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &>/dev/null; then
            PKG_MANAGER="apt"
        elif command -v dnf &>/dev/null; then
            PKG_MANAGER="dnf"
        elif command -v yum &>/dev/null; then
            PKG_MANAGER="yum"
        elif command -v pacman &>/dev/null; then
            PKG_MANAGER="pacman"
        elif command -v apk &>/dev/null; then
            PKG_MANAGER="apk"
        else
            PKG_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    else
        OS="unknown"
        PKG_MANAGER="unknown"
    fi

    info "Detected OS: $OS (package manager: $PKG_MANAGER)"
}

# ─── Install bash ─────────────────────────────────────────────────────────────
install_bash() {
    if command -v bash &>/dev/null; then
        local bash_version
        bash_version=$(bash --version | head -n1 | grep -oP '\d+\.\d+' | head -1)
        success "bash is installed (version $bash_version)"
        return 0
    fi

    warn "bash not found — installing..."
    case "$PKG_MANAGER" in
        apt)
            sudo apt-get update -qq && sudo apt-get install -y -qq bash
            ;;
        dnf)
            sudo dnf install -y bash
            ;;
        yum)
            sudo yum install -y bash
            ;;
        pacman)
            sudo pacman -Sy --noconfirm bash
            ;;
        apk)
            sudo apk add --no-cache bash
            ;;
        brew)
            brew install bash
            ;;
        *)
            error "Cannot auto-install bash. Please install it manually."
            exit 1
            ;;
    esac
    success "bash installed successfully"
}

# ─── Install git ──────────────────────────────────────────────────────────────
install_git() {
    if command -v git &>/dev/null; then
        local git_version
        git_version=$(git --version | grep -oP '\d+\.\d+\.\d+' | head -1)
        success "git is installed (version $git_version)"
        return 0
    fi

    warn "git not found — installing..."
    case "$PKG_MANAGER" in
        apt)
            sudo apt-get update -qq && sudo apt-get install -y -qq git
            ;;
        dnf)
            sudo dnf install -y git
            ;;
        yum)
            sudo yum install -y git
            ;;
        pacman)
            sudo pacman -Sy --noconfirm git
            ;;
        apk)
            sudo apk add --no-cache git
            ;;
        brew)
            brew install git
            ;;
        *)
            error "Cannot auto-install git. Please install it manually."
            exit 1
            ;;
    esac
    success "git installed successfully"
}

# ─── Install Julia ────────────────────────────────────────────────────────────
install_julia() {
    if command -v julia &>/dev/null; then
        local julia_version
        julia_version=$(julia --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)
        success "Julia is installed (version $julia_version)"
        return 0
    fi

    warn "Julia not found — installing Julia $JULIA_INSTALL_VERSION..."

    case "$OS" in
        linux)
            local arch
            arch=$(uname -m)
            if [[ "$arch" == "x86_64" ]]; then
                arch="x64"
            elif [[ "$arch" == "aarch64" ]]; then
                arch="aarch64"
            fi

            local julia_url="https://julialang-s3.julialang.org/bin/linux/${arch}/${JULIA_INSTALL_VERSION%.*}/julia-${JULIA_INSTALL_VERSION}-linux-${arch}.tar.gz"
            info "Downloading Julia from: $julia_url"

            local tmp_dir
            tmp_dir=$(mktemp -d)
            curl -fsSL "$julia_url" -o "$tmp_dir/julia.tar.gz"
            tar -xzf "$tmp_dir/julia.tar.gz" -C "$tmp_dir"

            # Install to /usr/local or ~/.local
            if [[ -w /usr/local/bin ]]; then
                sudo cp -r "$tmp_dir"/julia-*/. /usr/local/julia/
                sudo ln -sf /usr/local/julia/bin/julia /usr/local/bin/julia
            else
                mkdir -p "$HOME/.local/julia"
                cp -r "$tmp_dir"/julia-*/. "$HOME/.local/julia/"
                mkdir -p "$HOME/.local/bin"
                ln -sf "$HOME/.local/julia/bin/julia" "$HOME/.local/bin/julia"
                export PATH="$HOME/.local/bin:$PATH"
                info "Added Julia to PATH ($HOME/.local/bin)"
                info "Add this to your shell profile: export PATH=\"\$HOME/.local/bin:\$PATH\""
            fi
            rm -rf "$tmp_dir"
            ;;
        macos)
            if command -v brew &>/dev/null; then
                brew install julia
            else
                error "Please install Julia from https://julialang.org/downloads/"
                exit 1
            fi
            ;;
        *)
            error "Cannot auto-install Julia. Please install from https://julialang.org/downloads/"
            exit 1
            ;;
    esac

    # Verify installation
    if command -v julia &>/dev/null; then
        success "Julia installed successfully ($(julia --version))"
    else
        error "Julia installation failed. Please install manually from https://julialang.org/downloads/"
        exit 1
    fi
}

# ─── Install Julia packages ──────────────────────────────────────────────────
install_julia_packages() {
    info "Installing Julia package dependencies..."

    julia --project="$SCRIPT_DIR" -e '
        import Pkg
        println("  Resolving package dependencies...")
        try
            Pkg.instantiate()
            Pkg.resolve()
            println("  ✅ All packages resolved from Project.toml")
        catch e
            println("  ⚠️  Adding packages manually...")
            Pkg.add([
                Pkg.PackageSpec(name="HTTP"),
                Pkg.PackageSpec(name="JSON"),
                Pkg.PackageSpec(name="Downloads"),
            ])
            Pkg.resolve()
            Pkg.instantiate()
            println("  ✅ Packages installed manually")
        end
        println("  Precompiling...")
        Pkg.precompile()
        println("  ✅ Precompilation complete")
    '

    success "Julia packages installed and precompiled"
}

# ─── Verify everything works ──────────────────────────────────────────────────
verify_installation() {
    info "Running verification..."

    julia --project="$SCRIPT_DIR" -e '
        include(joinpath("'"$SCRIPT_DIR"'", "src", "ZenodoSpectralSDK.jl"))
        using .ZenodoSpectralSDK
        result = dispatch_action(Dict("action" => "health"))
        if get(result, "ok", false)
            println("  ✅ SDK health check PASSED")
        else
            println("  ❌ SDK health check FAILED")
            exit(1)
        end
    '

    success "All verification checks passed!"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
    header
    detect_os

    echo ""
    info "Step 1/5: Checking bash..."
    install_bash

    echo ""
    info "Step 2/5: Checking git..."
    install_git

    echo ""
    info "Step 3/5: Checking Julia..."
    install_julia

    echo ""
    info "Step 4/5: Installing Julia packages..."
    install_julia_packages

    echo ""
    info "Step 5/5: Verifying installation..."
    verify_installation

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║   ✅ Installation Complete!                                     ║"
    echo "║                                                                 ║"
    echo "║   Launch the Research OS terminal:                              ║"
    echo "║     cd $(basename "$SCRIPT_DIR")                                              ║"
    echo "║     julia launch.jl                                             ║"
    echo "║                                                                 ║"
    echo "║   Or use the CLI directly:                                      ║"
    echo "║     julia cli.jl health                                         ║"
    echo "║     julia cli.jl popular                                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

main "$@"
