param([string] $ConfigPath)
$ErrorActionPreference = 'Stop'
$modelDir = 'D:\models\deep4-1-1017-rco-2bxs'
$pack = Join-Path $modelDir 'maya-ds4'
$engram = Join-Path $modelDir 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'
$meta = Join-Path $modelDir 'UD-Q8_K_XL\DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf'
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root 'engine-dsv41-maxwell\strata-dsv41.exe'
$hf = Join-Path $env:USERPROFILE '.local\bin\hf.exe'
$shards = 1..4 | ForEach-Object { Join-Path $modelDir ('DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-{0:D5}-of-00004.gguf' -f $_) }

if (-not (Test-Path $engine -PathType Leaf)) { throw "The SM 5.2 DSV4.1 engine is missing: $engine" }
foreach ($s in $shards) { if (-not (Test-Path $s -PathType Leaf)) { throw "Missing backbone shard: $s" } }
if (-not (Test-Path $engram -PathType Leaf)) {
    Write-Host 'The required 110.6 GB EnGram lookup is missing. Resume it with:' -ForegroundColor Yellow
    Write-Host "& '$hf' download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf --local-dir '$modelDir'" -ForegroundColor Cyan
    exit 2
}
if ((Get-Item $engram).Length -ne 110595290816L) { throw "The EnGram file is incomplete: $engram" }
if (-not (Test-Path $meta -PathType Leaf)) {
    Write-Host 'This ISTA quant also needs the first Unsloth shard for EnGram token-map metadata (its weights are not used):' -ForegroundColor Yellow
    Write-Host "& '$hf' download unsloth/DeepSeek-V4.1-Flash-1017-GGUF UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf --local-dir '$modelDir'" -ForegroundColor Cyan
    exit 3
}
if (-not (Test-Path (Join-Path $pack 'tokenizer\vocab.json'))) {
    $python = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    Write-Host 'Building the zero-copy DSV4.1 pack (weights stay in their existing GGUF files) ...' -ForegroundColor Cyan
    & $python (Join-Path $root 'tools\dsv41\pack_gguf.py') --model $modelDir --out $pack --shards @shards --engram $modelDir --engram-meta $meta --dspark-noise-token 128799
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host 'DeepSeek V4.1 pack verified.' -ForegroundColor Green
Write-Host 'GPU ceiling: 21,888 MiB (95% of the Tesla M40); 1,152 MiB reserved.'

