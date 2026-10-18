param(
    [Parameter(Mandatory = $true)]
    [string] $ConfigPath,
    [string[]] $ModelsRoot = @("D:\Nemotron3.5-Bolt", "D:\models")
)

$ErrorActionPreference = "Stop"
trap {
    Write-Host "Nemotron launcher: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
$firstName = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00001-of-00003.gguf"
$expected = [ordered]@{
    "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00001-of-00003.gguf" = 7876160L
    "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00002-of-00003.gguf" = 49524981280L
    "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00003-of-00003.gguf" = 686620928L
}

$roots = @($ModelsRoot | Where-Object { Test-Path -LiteralPath $_ -PathType Container })
if ($roots.Count -eq 0) { throw "No configured Nemotron models folder exists: $($ModelsRoot -join ', ')" }
$matches = @($roots | ForEach-Object { Get-ChildItem -LiteralPath $_ -File -Recurse -Filter $firstName -ErrorAction SilentlyContinue })
if ($matches.Count -eq 0) {
    throw "Nemotron shard 1 was not found below $($roots -join ', ')."
}
if ($matches.Count -gt 1) {
    throw "More than one Nemotron UD-Q4_K_XL shard set was found below $ModelsRoot. Keep one complete set so the launcher cannot select the wrong files."
}

$model = $matches[0]
foreach ($item in $expected.GetEnumerator()) {
    $path = Join-Path $model.DirectoryName $item.Key
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "The model download is incomplete: missing $($item.Key)"
    }
    $length = (Get-Item -LiteralPath $path).Length
    if ($length -ne $item.Value) {
        throw "The model download is incomplete: $($item.Key) is $length bytes; expected $($item.Value)."
    }
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$modelArg = [Array]::IndexOf([object[]] $config.args, "--model")
if ($modelArg -lt 0 -or $modelArg + 1 -ge $config.args.Count) {
    throw "The Nemotron config has no --model argument."
}
$config.args[$modelArg + 1] = $model.FullName
$json = $config | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($ConfigPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

Write-Host "Nemotron model verified: $($model.DirectoryName)"
Write-Host "GPU budget: Strata may use 21888 MiB (95% of the Tesla M40); 1152 MiB stays free."
