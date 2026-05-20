param(
    [string]$EnvName = "..\g310",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot $EnvName

Write-Host "Creating GPU environment at $envPath"
py -$PythonVersion -m venv $envPath

$pythonExe = Join-Path $envPath "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found in created environment: $pythonExe"
}

& $pythonExe -m pip install --upgrade pip setuptools wheel
& $pythonExe -m pip install -e "$repoRoot[gpu]" --no-build-isolation

Write-Host ""
Write-Host "Verifying TensorFlow device visibility..."
& $pythonExe -c "import tensorflow as tf; print('TF', tf.__version__); print('Built with CUDA', tf.test.is_built_with_cuda()); print('GPUs', tf.config.list_physical_devices('GPU'))"

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with: $envPath\Scripts\Activate.ps1"
