<#
.SYNOPSIS
    MESIE — Multi-Element Spectral Intelligence Engine PowerShell Module.

.DESCRIPTION
    Cross-platform PowerShell wrapper for the MESIE SDK. Provides cmdlet-style
    functions for spectral validation, matching, generation, embedding, and
    enterprise AI workflows.

    Works on Windows PowerShell 5.1+ and PowerShell Core 7+ (Linux/macOS).

.EXAMPLE
    Import-Module ./scripts/MESIE.psm1
    Test-MESIEInstall
    Invoke-MESIEValidate -RecordPath "data/reference/vibration_monitoring_reference.json"
    Invoke-MESIEMonteCarlo -Trials 500
#>

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

$script:MESIEPython = if ($env:MESIE_PYTHON) { $env:MESIE_PYTHON } else { "python" }
$script:MESIERoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Get-MESIERoot {
    <#
    .SYNOPSIS
        Returns the MESIE repository root path.
    #>
    if ($script:MESIERoot -and (Test-Path $script:MESIERoot)) {
        return $script:MESIERoot
    }
    # Fallback: walk up from this script
    $current = $PSScriptRoot
    while ($current -and !(Test-Path (Join-Path $current "pyproject.toml"))) {
        $current = Split-Path -Parent $current
    }
    return $current
}

# ---------------------------------------------------------------------------
# Installation & Environment
# ---------------------------------------------------------------------------

function Test-MESIEInstall {
    <#
    .SYNOPSIS
        Verify MESIE SDK is installed and importable.
    .OUTPUTS
        [PSCustomObject] with Version, Status, and Python path.
    #>
    [CmdletBinding()]
    param()

    $result = & $script:MESIEPython -c "
import sys
try:
    import mesie
    print(f'OK|{mesie.__version__}|{sys.executable}')
except ImportError as e:
    print(f'FAIL|{e}|{sys.executable}')
" 2>&1

    $parts = ($result -join "").Split("|")
    [PSCustomObject]@{
        Status  = $parts[0]
        Version = $parts[1]
        Python  = $parts[2]
    }
}

function Install-MESIE {
    <#
    .SYNOPSIS
        Install MESIE SDK in development mode.
    .PARAMETER Full
        Include full scientific stack (scipy, pandas, scikit-learn, networkx).
    .PARAMETER Extras
        Additional extras to install (e.g., "ml", "ai", "intelligence").
    #>
    [CmdletBinding()]
    param(
        [switch]$Full,
        [string[]]$Extras = @()
    )

    $root = Get-MESIERoot
    $extraStr = "dev"
    if ($Full) { $extraStr += ",full" }
    foreach ($e in $Extras) { $extraStr += ",$e" }

    Write-Host "Installing MESIE [$extraStr] from $root ..." -ForegroundColor Cyan
    & $script:MESIEPython -m pip install -e "$root[$extraStr]"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ MESIE installed successfully" -ForegroundColor Green
    } else {
        Write-Error "MESIE installation failed with exit code $LASTEXITCODE"
    }
}

# ---------------------------------------------------------------------------
# Core SDK Operations
# ---------------------------------------------------------------------------

function Invoke-MESIEValidate {
    <#
    .SYNOPSIS
        Validate a spectral record file.
    .PARAMETER RecordPath
        Path to JSON spectral record file.
    .OUTPUTS
        [PSCustomObject] with IsValid, Level, Errors, Warnings.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$RecordPath
    )

    $absPath = Resolve-Path $RecordPath -ErrorAction Stop
    $json = & $script:MESIEPython -c "
import json, sys
sys.path.insert(0, '$(Get-MESIERoot)')
from mesie import load_record, validate_record
rec = load_record('$absPath')
v = validate_record(rec)
print(json.dumps({'is_valid': v.is_valid, 'level': v.level, 'errors': v.errors, 'warnings': v.warnings}))
" 2>&1

    $data = $json | ConvertFrom-Json
    [PSCustomObject]@{
        IsValid  = $data.is_valid
        Level    = $data.level
        Errors   = $data.errors
        Warnings = $data.warnings
        File     = $absPath
    }
}

function Invoke-MESIEMatch {
    <#
    .SYNOPSIS
        Match two spectral record files.
    .PARAMETER ReferencePath
        Path to reference record.
    .PARAMETER CandidatePath
        Path to candidate record.
    .OUTPUTS
        [PSCustomObject] with CompositeScore and MetricBreakdown.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$ReferencePath,
        [Parameter(Mandatory)]
        [string]$CandidatePath
    )

    $refAbs = Resolve-Path $ReferencePath -ErrorAction Stop
    $candAbs = Resolve-Path $CandidatePath -ErrorAction Stop

    $json = & $script:MESIEPython -c "
import json, sys
sys.path.insert(0, '$(Get-MESIERoot)')
from mesie import load_record, match_records
ref = load_record('$refAbs')
cand = load_record('$candAbs')
result = match_records(ref, cand)
print(json.dumps({'composite_score': result.composite_score, 'metrics': result.metric_breakdown}))
" 2>&1

    $data = $json | ConvertFrom-Json
    [PSCustomObject]@{
        CompositeScore  = $data.composite_score
        MetricBreakdown = $data.metrics
        Reference       = $refAbs
        Candidate       = $candAbs
    }
}

function Invoke-MESIEGenerate {
    <#
    .SYNOPSIS
        Generate a spectral record (PSD or FAS).
    .PARAMETER Type
        Type of spectrum: "psd" or "fas".
    .PARAMETER Seed
        Random seed for reproducibility.
    .PARAMETER OutputPath
        Optional path to save the generated record.
    .OUTPUTS
        [PSCustomObject] with RecordId, Components, and optional FilePath.
    #>
    [CmdletBinding()]
    param(
        [ValidateSet("psd", "fas")]
        [string]$Type = "psd",
        [int]$Seed = 42,
        [string]$OutputPath
    )

    $saveCmd = ""
    if ($OutputPath) {
        $saveCmd = "
import json
data = {'record_id': rec.record_id, 'frequency': rec.components[0].frequency.tolist(), 'amplitude': rec.components[0].amplitude.tolist()}
with open('$OutputPath', 'w') as f: json.dump(data, f, indent=2)
print(json.dumps({'record_id': rec.record_id, 'n_points': len(rec.components[0].frequency), 'saved': '$OutputPath'}))
"
    } else {
        $saveCmd = "
import json
print(json.dumps({'record_id': rec.record_id, 'n_points': len(rec.components[0].frequency), 'saved': None}))
"
    }

    $json = & $script:MESIEPython -c "
import sys
sys.path.insert(0, '$(Get-MESIERoot)')
from mesie import generate_psd, generate_fas
from mesie.core.config import GenerationConfig
cfg = GenerationConfig(seed=$Seed)
rec = generate_$Type(config=cfg)
$saveCmd
" 2>&1

    $data = $json | ConvertFrom-Json
    [PSCustomObject]@{
        RecordId = $data.record_id
        Points   = $data.n_points
        Type     = $Type
        Saved    = $data.saved
    }
}

function Invoke-MESIEEmbed {
    <#
    .SYNOPSIS
        Compute spectral embedding for a record.
    .PARAMETER RecordPath
        Path to spectral record file.
    .OUTPUTS
        [PSCustomObject] with Dimension and Embedding vector.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$RecordPath
    )

    $absPath = Resolve-Path $RecordPath -ErrorAction Stop
    $json = & $script:MESIEPython -c "
import json, sys
sys.path.insert(0, '$(Get-MESIERoot)')
from mesie import load_record
from mesie.embeddings import SpectralVectorizer
rec = load_record('$absPath')
v = SpectralVectorizer()
emb = v.transform(rec)
print(json.dumps({'dimension': len(emb), 'embedding': emb.tolist()}))
" 2>&1

    $data = $json | ConvertFrom-Json
    [PSCustomObject]@{
        Dimension = $data.dimension
        Embedding = $data.embedding
        File      = $absPath
    }
}

# ---------------------------------------------------------------------------
# Enterprise AI Operations
# ---------------------------------------------------------------------------

function Invoke-MESIEMonteCarlo {
    <#
    .SYNOPSIS
        Run the MESIE Monte Carlo enterprise benchmark.
    .PARAMETER Trials
        Number of trials per use case (default 200, for 2000 total).
    .OUTPUTS
        [PSCustomObject] with overall results.
    #>
    [CmdletBinding()]
    param(
        [int]$Trials = 200
    )

    $root = Get-MESIERoot
    Write-Host "Running Monte Carlo benchmark ($Trials trials × 10 use cases = $($Trials * 10) total)..." -ForegroundColor Cyan

    $output = & $script:MESIEPython "$root/scripts/monte_carlo_enterprise_benchmark.py" --trials $Trials 2>&1
    $output | ForEach-Object { Write-Host $_ }

    # Parse report
    $reportPath = Join-Path $root "deliverables/MESIE_Monte_Carlo_Enterprise_Report.json"
    if (Test-Path $reportPath) {
        $report = Get-Content $reportPath -Raw | ConvertFrom-Json
        [PSCustomObject]@{
            TotalTrials    = $report.total_trials
            SuccessRate    = $report.overall_success_rate
            EnterpriseGrade = $report.enterprise_grade
            ElapsedSeconds = $report.elapsed_s
            ReportPath     = $reportPath
        }
    }
}

function Invoke-MESIETest {
    <#
    .SYNOPSIS
        Run MESIE pytest suite.
    .PARAMETER Filter
        Optional pytest -k filter expression.
    .PARAMETER Verbose
        Show verbose output.
    .PARAMETER File
        Specific test file to run.
    #>
    [CmdletBinding()]
    param(
        [string]$Filter,
        [switch]$Verbose,
        [string]$File
    )

    $root = Get-MESIERoot
    $args = @("$root/tests/")
    if ($File) { $args = @($File) }
    if ($Verbose) { $args += "-v" }
    if ($Filter) { $args += "-k"; $args += $Filter }

    & $script:MESIEPython -m pytest @args
}

function Get-MESIEKnowledge {
    <#
    .SYNOPSIS
        Display MAESI knowledge base statistics.
    .OUTPUTS
        [PSCustomObject] with counts for each knowledge domain.
    #>
    [CmdletBinding()]
    param()

    $json = & $script:MESIEPython -c "
import json, sys
sys.path.insert(0, '$(Get-MESIERoot)')
from mesie.sdk import MAESIClient
client = MAESIClient(fast=False, use_fingerprint=False)
stats = client.knowledge_stats()
print(json.dumps({
    'physical_laws': stats.physical_laws,
    'chemical_elements': stats.chemical_elements,
    'biological_systems': stats.biological_systems,
    'technical_concepts': stats.technical_concepts,
    'research_entries': stats.research_entries,
}))
" 2>&1

    $data = $json | ConvertFrom-Json
    [PSCustomObject]@{
        PhysicalLaws      = $data.physical_laws
        ChemicalElements  = $data.chemical_elements
        BiologicalSystems = $data.biological_systems
        TechnicalConcepts = $data.technical_concepts
        ResearchEntries   = $data.research_entries
    }
}

function Search-MESIEResearch {
    <#
    .SYNOPSIS
        Search the MAESI research catalog.
    .PARAMETER Query
        Search query string.
    .PARAMETER TopK
        Number of results to return.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Query,
        [int]$TopK = 5
    )

    $json = & $script:MESIEPython -c "
import json, sys
sys.path.insert(0, '$(Get-MESIERoot)')
from mesie.sdk import search_research
hits = search_research('$Query', top_k=$TopK)
print(json.dumps([{'title': h.title, 'field': h.field.value if hasattr(h.field, 'value') else str(h.field)} for h in hits]))
" 2>&1

    $results = $json | ConvertFrom-Json
    $results | ForEach-Object {
        [PSCustomObject]@{
            Title = $_.title
            Field = $_.field
        }
    }
}

# ---------------------------------------------------------------------------
# Electron Desktop UX Launcher
# ---------------------------------------------------------------------------

function Start-MESIEDesktop {
    <#
    .SYNOPSIS
        Launch the MESIE Electron desktop application.
    .PARAMETER Dev
        Run in development mode with hot reload.
    #>
    [CmdletBinding()]
    param(
        [switch]$Dev
    )

    $desktopPath = Join-Path (Get-MESIERoot) "mesie-desktop"
    if (!(Test-Path $desktopPath)) {
        Write-Error "Electron app not found at $desktopPath. Run 'npm install' in mesie-desktop/ first."
        return
    }

    Push-Location $desktopPath
    try {
        if (!(Test-Path "node_modules")) {
            Write-Host "Installing dependencies..." -ForegroundColor Yellow
            npm install
        }
        if ($Dev) {
            Write-Host "Starting MESIE Desktop (dev mode)..." -ForegroundColor Cyan
            npm run dev
        } else {
            Write-Host "Starting MESIE Desktop..." -ForegroundColor Cyan
            npm start
        }
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# Module Export
# ---------------------------------------------------------------------------

Export-ModuleMember -Function @(
    'Get-MESIERoot',
    'Test-MESIEInstall',
    'Install-MESIE',
    'Invoke-MESIEValidate',
    'Invoke-MESIEMatch',
    'Invoke-MESIEGenerate',
    'Invoke-MESIEEmbed',
    'Invoke-MESIEMonteCarlo',
    'Invoke-MESIETest',
    'Get-MESIEKnowledge',
    'Search-MESIEResearch',
    'Start-MESIEDesktop'
)
