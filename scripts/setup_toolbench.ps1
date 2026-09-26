$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ExpectedCommit = "aa4ed9f4737ad98bd706663f01d63623c3427812"
$Submodule = Join-Path $Root "external/StableToolBenchQoS"
$Venv = Join-Path $Root ".venv-toolbench-server"
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

git -C $Root submodule update --init --recursive external/StableToolBenchQoS
$ActualCommit = (git -C $Submodule rev-parse HEAD).Trim()
if ($ActualCommit -ne $ExpectedCommit) {
    throw "StableToolBench revision mismatch: expected $ExpectedCommit, got $ActualCommit"
}

& $PythonBin -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
& $PythonBin -m venv $Venv
& (Join-Path $Venv "Scripts/python.exe") -m pip install --upgrade pip
& (Join-Path $Venv "Scripts/python.exe") -m pip install -r (Join-Path $Root "requirements-toolbench-server.lock.txt")
& (Join-Path $Venv "Scripts/python.exe") (Join-Path $Root "scripts/check_toolbench_setup.py")

Write-Host "StableToolBench server environment is ready: $Venv"
