[CmdletBinding()]
param(
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Confirm-Gate {
    param([Parameter(Mandatory)][string]$Name)

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "The Windows executable must be built on Windows."
}
if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    throw "Run this script from the repository root."
}
$workingTreeChanges = @(git status --porcelain)
if ($workingTreeChanges.Count -ne 0) {
    git status --short
    throw "The working tree must be clean before packaging."
}

$version = python -c "import sys; sys.path.insert(0, 'src'); from open_world_rpg import __version__; print(__version__)"
Confirm-Gate "Read package version"

if (-not $SkipValidation) {
    & "$PSScriptRoot\validate_release.ps1" -ExpectedVersion $version
}

python -m pip install -e ".[release]"
Confirm-Gate "Release dependencies"

$buildRoot = Join-Path (Get-Location) "build\windows-release"
$distRoot = Join-Path $buildRoot "dist"
$packageRoot = Join-Path $buildRoot "OpenWorldRPG-$version-windows-x64"
$archivePath = Join-Path (Get-Location) "dist\OpenWorldRPG-$version-windows-x64.zip"

Remove-Item -Recurse -Force $buildRoot -ErrorAction SilentlyContinue
Remove-Item -Force $archivePath -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $distRoot, $packageRoot, (Split-Path $archivePath) | Out-Null

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name OpenWorldRPG `
    --paths src `
    --collect-all pygame `
    --collect-all moderngl `
    --hidden-import glcontext `
    --distpath $distRoot `
    --workpath (Join-Path $buildRoot "work") `
    --specpath (Join-Path $buildRoot "spec") `
    src/open_world_rpg/__main__.py
Confirm-Gate "PyInstaller"

$executable = Join-Path $distRoot "OpenWorldRPG.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "PyInstaller did not produce OpenWorldRPG.exe."
}

Copy-Item -LiteralPath $executable -Destination $packageRoot
Copy-Item -LiteralPath README.md, LICENSE -Destination $packageRoot
@"
Open World RPG $version

Launch OpenWorldRPG.exe from a writable folder. The game creates saves/, logs/,
and crash-reports/ beside the executable when needed.
"@ | Set-Content -LiteralPath (Join-Path $packageRoot "START-HERE.txt") -Encoding utf8

Compress-Archive -Path "$packageRoot\*" -DestinationPath $archivePath -CompressionLevel Optimal

Write-Host "`nWindows release archive: $archivePath"
