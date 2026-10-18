param(
    [Parameter(Mandatory = $true)]
    [string] $ConfigPath
)

$ErrorActionPreference = "Stop"
trap {
    Write-Host "Qwen3.6 launcher: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$model = "D:\Qwen3.6-35B\Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
$expectedBytes = 21388319008L
$tokenizer = "D:\polystrata-maxwell\PolyStrata-data\packs\qwen36-ud-q4_k_s\tokenizer"
$engine = "D:\polystrata-maxwell\PolyStrata-0.1.44.1\engine-poly-maxwell\strata-poly.exe"

if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw "The model file was not found: $model" }
$length = (Get-Item -LiteralPath $model).Length
if ($length -ne $expectedBytes) { throw "The model file is $length bytes; expected $expectedBytes. The download may be incomplete." }
if (-not (Test-Path -LiteralPath (Join-Path $tokenizer "vocab.json") -PathType Leaf)) { throw "The extracted tokenizer is missing: $tokenizer" }
if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) { throw "The Maxwell PolyStrata engine is missing: $engine" }

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ($config.args[0] -ne "--model" -or $config.args[1] -ne $model) { throw "The JSON does not point to the verified Qwen3.6 file." }

Write-Host "Qwen3.6 model verified: $model"
Write-Host "GPU budget: Strata may use 21888 MiB (95% of the Tesla M40); 1152 MiB stays free."
