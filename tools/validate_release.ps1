[CmdletBinding()]
param(
    [string]$ExpectedVersion = "0.9.0"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$env:PYGAME_HIDE_SUPPORT_PROMPT = "1"

function Confirm-Gate {
    param([Parameter(Mandatory)][string]$Name)

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}


function Get-CandidateFingerprint {
    $statusLines = @(git status --porcelain=v1 --untracked-files=all)
    Confirm-Gate "Read candidate Git status"

    $records = foreach ($line in $statusLines) {
        if ($line.Length -lt 4) {
            throw "Git returned an invalid porcelain status line."
        }

        $path = $line.Substring(3)
        $hash = if (Test-Path -LiteralPath $path -PathType Leaf) {
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        }
        else {
            "<missing>"
        }
        "$line`t$hash"
    }

    return @($records | Sort-Object)
}

function Remove-ReleaseArtefacts {
    Remove-Item -Recurse -Force `
        .release-runtime, `
        .pytest-v1-release, `
        .pytest-v1-order-a, `
        .pytest-v1-order-b `
        -ErrorAction SilentlyContinue
    Remove-Item -Force .coverage-v1.json, .pytest-v1.xml -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    throw "Run this script from the repository root."
}

$initialFingerprint = @(Get-CandidateFingerprint)

Remove-ReleaseArtefacts

try {
    python --version
    Confirm-Gate "Python version"

    python -m pip install -e ".[dev]"
    Confirm-Gate "Editable installation"

    python -m ruff check .
    Confirm-Gate "Ruff lint"

    python -m ruff format --check .
    Confirm-Gate "Ruff formatting"

    python -m mypy
    Confirm-Gate "Mypy"

    python -m pytest `
        --basetemp .pytest-v1-release `
        --junitxml .pytest-v1.xml
    Confirm-Gate "Complete test suite"

    [xml]$junit = Get-Content -LiteralPath .pytest-v1.xml
    $suites = @($junit.testsuites.testsuite)
    $skipped = ($suites | Measure-Object -Property skipped -Sum).Sum
    if ($null -eq $skipped) {
        $skipped = 0
    }
    if ([int]$skipped -ne 0) {
        throw "Release validation requires zero skipped tests; found $skipped."
    }

    python -m coverage json -o .coverage-v1.json
    Confirm-Gate "Coverage JSON"

    python -c "import json, pathlib, sys; totals = json.loads(pathlib.Path('.coverage-v1.json').read_text(encoding='utf-8'))['totals']; missing_lines = int(totals['missing_lines']); missing_branches = int(totals['missing_branches']); print(f'Coverage gaps: {missing_lines} lines, {missing_branches} branches'); sys.exit(1 if missing_lines or missing_branches else 0)"
    Confirm-Gate "Coverage completeness"

    python -m pytest `
        tests/integration/test_terrain_demo.py `
        tests/integration/test_voxel_demo.py `
        --no-cov `
        --basetemp .pytest-v1-order-a
    Confirm-Gate "Graphical order A"

    python -m pytest `
        tests/integration/test_voxel_demo.py `
        tests/integration/test_terrain_demo.py `
        --no-cov `
        --basetemp .pytest-v1-order-b
    Confirm-Gate "Graphical order B"

    python -m pip check
    Confirm-Gate "Dependency validation"

    $versionOutput = python -m open_world_rpg --version
    Confirm-Gate "Module version"
    if ($versionOutput -ne "Open World RPG $ExpectedVersion") {
        throw "Unexpected module version output: $versionOutput"
    }

    $installedVersionOutput = open-world-rpg --version
    Confirm-Gate "Installed version"
    if ($installedVersionOutput -ne "Open World RPG $ExpectedVersion") {
        throw "Unexpected installed version output: $installedVersionOutput"
    }

    python -m open_world_rpg --runtime-check
    Confirm-Gate "Rendering-free runtime check"

    python -m open_world_rpg `
        --smoke-test `
        --smoke-frames 3 `
        --data-dir .release-runtime
    Confirm-Gate "Primary launcher smoke test"

    open-world-rpg-voxel-demo `
        --smoke-test `
        --smoke-frames 3 `
        --data-dir .release-runtime
    Confirm-Gate "Voxel launcher smoke test"

    python -m open_world_rpg.ui.terrain_demo --smoke-test
    Confirm-Gate "Terrain module smoke test"

    open-world-rpg-terrain-demo --smoke-test
    Confirm-Gate "Terrain installed smoke test"

    git diff --check
    Confirm-Gate "Git whitespace validation"
}
finally {
    Remove-ReleaseArtefacts
}

$finalFingerprint = @(Get-CandidateFingerprint)
$unexpectedMutations = @(
    Compare-Object -ReferenceObject $initialFingerprint -DifferenceObject $finalFingerprint
)
if ($unexpectedMutations.Count -ne 0) {
    git status --short
    throw "Release validation changed the working tree."
}

Write-Host "`nRelease validation passed for Open World RPG $ExpectedVersion without mutating the candidate changes."
