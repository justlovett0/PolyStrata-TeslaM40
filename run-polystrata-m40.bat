@echo off
setlocal
title PolyStrata M40 Launcher
set "M40_SELF=%~f0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$p=$env:M40_SELF; $s=Get-Content -LiteralPath $p; $m=[Array]::IndexOf($s,'#__POLYSTRATA_M40_POWERSHELL__'); if($m -lt 0){throw 'Embedded launcher code is missing'}; & ([scriptblock]::Create(($s[($m+1)..($s.Length-1)] -join [Environment]::NewLine))) -Root '%~dp0'"
set "M40_EXIT=%ERRORLEVEL%"
if not "%M40_EXIT%"=="0" (
  echo.
  echo PolyStrata stopped with exit code %M40_EXIT%.
  pause
)
exit /b %M40_EXIT%

#__POLYSTRATA_M40_POWERSHELL__
param([string]$Root)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root)
$LastConfig = Join-Path $Root 'strata-m40-last-run.json'
$DataRoot = 'D:\polystrata-maxwell\PolyStrata-data'

function Banner {
    Clear-Host
    Write-Host '===============================================================' -ForegroundColor DarkCyan
    Write-Host '  PolyStrata Maxwell Edition  |  Tesla M40 smart launcher' -ForegroundColor Cyan
    Write-Host '===============================================================' -ForegroundColor DarkCyan
    Write-Host '  CUDA 12.0 / native SM 5.2 / 23,040 MiB M40 profile'
    Write-Host '  A generated copy is saved as strata-m40-last-run.json.'
    Write-Host
}

function Read-Menu([string]$Prompt, [int]$Min, [int]$Max, [int]$Default = 0) {
    while ($true) {
        $suffix = if ($Default -ge $Min) { " [$Default]" } else { '' }
        $answer = Read-Host "$Prompt$suffix"
        $raw = if ($null -eq $answer) { '' } else { $answer.Trim() }
        if (-not $raw -and $Default -ge $Min) { return $Default }
        $n = 0
        if ([int]::TryParse($raw, [ref]$n) -and $n -ge $Min -and $n -le $Max) { return $n }
        Write-Host "Enter a number from $Min to $Max." -ForegroundColor Yellow
    }
}

function Read-YesNo([string]$Prompt, [bool]$Default) {
    $hint = if ($Default) { 'Y/n' } else { 'y/N' }
    while ($true) {
        $answer = Read-Host "$Prompt [$hint]"
        $raw = if ($null -eq $answer) { '' } else { $answer.Trim().ToLowerInvariant() }
        if (-not $raw) { return $Default }
        if ($raw -in @('y','yes')) { return $true }
        if ($raw -in @('n','no')) { return $false }
        Write-Host 'Enter Y or N.' -ForegroundColor Yellow
    }
}

function Get-ArgIndex($List, [string]$Name) {
    for ($i = 0; $i -lt $List.Count; $i++) { if ($List[$i] -eq $Name) { return $i } }
    return -1
}

function Set-EngineArg($List, [string]$Name, [string]$Value) {
    $i = Get-ArgIndex $List $Name
    if ($i -ge 0) {
        if ($i + 1 -ge $List.Count) { throw "$Name has no value in the base config" }
        $List[$i + 1] = $Value
    } else {
        [void]$List.Add($Name); [void]$List.Add($Value)
    }
}

function Remove-EngineArg($List, [string]$Name) {
    $i = Get-ArgIndex $List $Name
    if ($i -ge 0) {
        $List.RemoveAt($i)
        if ($i -lt $List.Count -and -not $List[$i].StartsWith('--')) { $List.RemoveAt($i) }
    }
}

function Add-Flag($List, [string]$Name) {
    if ((Get-ArgIndex $List $Name) -lt 0) { [void]$List.Add($Name) }
}

function Remove-Flag($List, [string]$Name) {
    $i = Get-ArgIndex $List $Name
    if ($i -ge 0) { $List.RemoveAt($i) }
}

function Select-Gguf([string]$Given, [string]$PreferredLeaf = '') {
    $p = $Given.Trim().Trim('"')
    if (Test-Path -LiteralPath $p -PathType Leaf) {
        if ([IO.Path]::GetExtension($p) -ine '.gguf') { throw "Not a GGUF file: $p" }
        return [IO.Path]::GetFullPath($p)
    }
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { throw "Folder does not exist: $p" }
    if ($PreferredLeaf) {
        $exact = Get-ChildItem -LiteralPath $p -Filter $PreferredLeaf -File -Recurse -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if ($exact) { return $exact.FullName }
    }
    $files = @(Get-ChildItem -LiteralPath $p -Filter '*.gguf' -File -Recurse -ErrorAction SilentlyContinue |
               Sort-Object @{Expression={if($_.Name -match '-00001-of-'){0}else{1}}}, FullName)
    if ($files.Count -eq 0) { throw "No .gguf files were found under $p" }
    if ($files.Count -eq 1) { return $files[0].FullName }
    $first = @($files | Where-Object Name -Match '-00001-of-')
    $choices = if ($first.Count) { $first } else { $files }
    Write-Host 'GGUF candidates:' -ForegroundColor Cyan
    for ($i=0; $i -lt $choices.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i+1), $choices[$i].FullName) }
    $pick = Read-Menu 'Choose the model or first shard' 1 $choices.Count 1
    return $choices[$pick-1].FullName
}

function Get-GgufBytes([string]$Path) {
    $f = Get-Item -LiteralPath $Path
    if ($f.Name -match '^(.*)-\d{5}-of-(\d{5})\.gguf$') {
        $prefix = $Matches[1]; $count = $Matches[2]
        $set = @(Get-ChildItem -LiteralPath $f.DirectoryName -File -Filter "$prefix-*-of-$count.gguf")
        if ($set.Count) { return [int64](($set | Measure-Object Length -Sum).Sum) }
    }
    return [int64]$f.Length
}

function Slug([string]$Text) {
    $s = ($Text.ToLowerInvariant() -replace '[^a-z0-9]+','-').Trim('-')
    if (-not $s) { return 'custom-model' }
    return $s
}

function Find-Python {
    # regex includes a CPython native extension. A Python executable can answer
    # --version successfully and still be unable to load this fork's dependencies.
    $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $venv = Join-Path $Root '.venv\Scripts\python.exe'
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    $choices = @($bundled,$venv)
    if ($python) { $choices += $python.Source }
    $oldPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $Root '.tools\pydeps'
    $probe = "import sys; sys.path.insert(0, r'$($Root.Replace("'", "''"))tools'); import regex, strata_tokenizer"
    $chosen = $null
    try {
        foreach ($p in ($choices | Select-Object -Unique)) {
            if (-not $p -or -not (Test-Path -LiteralPath $p -PathType Leaf)) { continue }
            & $p -c $probe *> $null
            if ($LASTEXITCODE -eq 0) { $chosen = $p; break }
        }
    } finally {
        $env:PYTHONPATH = $oldPythonPath
    }
    if ($chosen) { return $chosen }
    throw 'No compatible Python was found. This build needs Python 3.12 x64 and its bundled native dependencies. Run SETUP.bat, then try again.'
}

Banner
$presets = @(
    [pscustomobject]@{Key='swift'; Name='Swift Qwen3.8 Flash Next IQ3_S'; Config='strata-swift-iq3_s.json'; Port=8080; Rec=131072; Near=196608; Draft=$true},
    [pscustomobject]@{Key='glm'; Name='GLM 5.3 Flash UD-IQ3_XXS'; Config='strata-glm-ud-iq3_xxs.json'; Port=8081; Rec=32768; Near=98304; Draft=$false},
    [pscustomobject]@{Key='nemotron'; Name='NVIDIA Nemotron 3.5 Bolt 74B-A9B UD-Q4_K_XL'; Config='strata-nemotron-ud-q4_k_xl.json'; Port=8082; Rec=32768; Near=65536; Draft=$false},
    [pscustomobject]@{Key='qwen36'; Name='Qwen3.6 35B-A3B UD-Q4_K_S'; Config='strata-qwen36-ud-q4_k_s.json'; Port=8083; Rec=131072; Near=196608; Draft=$true}
)
Write-Host 'Optimized M40 presets:' -ForegroundColor Cyan
for ($i=0; $i -lt $presets.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i+1), $presets[$i].Name) }
Write-Host '  5) Try an unsupported GGUF model (experimental strata-poly path)' -ForegroundColor Yellow
$selection = Read-Menu 'Model' 1 5 4
$custom = $selection -eq 5
$Python = Find-Python
$env:PYTHONPATH = (Join-Path $Root '.tools\pydeps') + ';' + (Join-Path $Root '.venv\Lib\site-packages') + ';' + $env:PYTHONPATH
$PythonVersion = (& $Python -c "import platform; print(platform.python_version())").Trim()

if (-not $custom) {
    $preset = $presets[$selection-1]
    $basePath = Join-Path $Root $preset.Config
    if (-not (Test-Path -LiteralPath $basePath)) { throw "Missing optimized config: $basePath" }
    $config = Get-Content -LiteralPath $basePath -Raw | ConvertFrom-Json
    $argsList = New-Object 'System.Collections.Generic.List[string]'
    foreach ($a in $config.args) { [void]$argsList.Add([string]$a) }
    $modelArg = if ((Get-ArgIndex $argsList '--model') -ge 0) { '--model' } else { '--native' }
    $modelPath = $argsList[(Get-ArgIndex $argsList $modelArg)+1]
    if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
        Write-Host "The preset model path is missing: $modelPath" -ForegroundColor Yellow
        $folder = Read-Host 'Enter the folder containing its GGUF file(s), or the full GGUF path'
        $modelPath = Select-Gguf $folder ([IO.Path]::GetFileName($modelPath))
        Set-EngineArg $argsList $modelArg $modelPath
        if ((Get-ArgIndex $argsList '--ple-gguf') -ge 0) { Set-EngineArg $argsList '--ple-gguf' $modelPath }
    }
    $displayName = $preset.Name
    $portDefault = $preset.Port
    $recommended = $preset.Rec
    $nearLimit = $preset.Near
    $draftDefault = $preset.Draft
} else {
    Write-Host
    Write-Host 'Unsupported mode accepts a GGUF architecture that strata-poly may understand.' -ForegroundColor Yellow
    Write-Host 'The launcher can extract its tokenizer, but it cannot add a missing model backend.' -ForegroundColor Yellow
    $folder = Read-Host 'Enter the folder containing the GGUF file(s), or the full GGUF path'
    $modelPath = Select-Gguf $folder
    $displayName = Read-Host 'Display name (blank = GGUF filename)'
    if ($null -eq $displayName) { $displayName = '' } else { $displayName = $displayName.Trim() }
    if (-not $displayName) { $displayName = [IO.Path]::GetFileNameWithoutExtension($modelPath) }
    $slug = Slug $displayName
    $pack = Join-Path $DataRoot (Join-Path 'packs' ("m40-custom-" + $slug))
    New-Item -ItemType Directory -Path $pack -Force | Out-Null
    Write-Host 'Extracting the tokenizer embedded in the GGUF ...' -ForegroundColor Cyan
    & $Python (Join-Path $Root 'tools\strata_tokenizer.py') --gguf $modelPath --out $pack
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $pack 'tokenizer\vocab.json'))) {
        throw 'Tokenizer extraction failed. This GGUF cannot be launched through the web server yet.'
    }
    $profile = Join-Path $pack 'expert-profile.txt'
    $config = [pscustomobject][ordered]@{
        exe = Join-Path $Root 'engine-poly-maxwell\strata-poly.exe'
        args = @('--model',$modelPath,'--max-context','32768','--prompt-batch','1024','--expert-profile',$profile,
                 '--vram-reserve-mib','1664')
        cwd = $Root
        tokenizer = Join-Path $pack 'tokenizer'
        model_name = $slug
        display_name = $displayName
        log = Join-Path $Root 'strata-m40-custom.log'
        lib_dirs = @('C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin')
        port = 8084
        cuda = 12
        gpu = 0
        gpus_asked = $true
    }
    $argsList = New-Object 'System.Collections.Generic.List[string]'
    foreach ($a in $config.args) { [void]$argsList.Add([string]$a) }
    $portDefault = 8084
    $bytes = Get-GgufBytes $modelPath
    if ($bytes -le 25GB) { $recommended=65536; $nearLimit=131072 }
    elseif ($bytes -le 55GB) { $recommended=32768; $nearLimit=98304 }
    else { $recommended=32768; $nearLimit=65536 }
    $draftDefault = $false
}

$bytes = Get-GgufBytes $modelPath
$gb = [math]::Round($bytes / 1GB, 1)
$gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
Write-Host
Write-Host "Model: $displayName" -ForegroundColor Green
Write-Host "GGUF set: $gb GiB  |  GPU: $gpu"
Write-Host ('Context estimate: recommended {0:N0} tokens; near-limit {1:N0} tokens.' -f $recommended,$nearLimit) -ForegroundColor Cyan
Write-Host 'The GGUF size mainly affects system RAM and expert offload; attention/KV architecture controls context VRAM.'
$contexts = @(32768,65536,98304,131072,196608,262144)
for ($i=0; $i -lt $contexts.Count; $i++) {
    $tag = if ($contexts[$i] -eq $recommended) {' [recommended]'} elseif ($contexts[$i] -eq $nearLimit) {' [near limit]'} elseif ($contexts[$i] -gt $nearLimit) {' [experimental / may OOM]'} else {''}
    Write-Host ("  {0}) {1}K{2}" -f ($i+1), [int]($contexts[$i]/1024), $tag)
}
$ctxPick = Read-Menu 'Context window' 1 6 ([Array]::IndexOf($contexts,$recommended)+1)
$context = $contexts[$ctxPick-1]
Set-EngineArg $argsList '--max-context' ([string]$context)

Write-Host
$draft = Read-YesNo 'Enable speculative drafting / MTP where this model supports it?' $draftDefault
$isPoly = [IO.Path]::GetFileName($config.exe).ToLowerInvariant() -eq 'strata-poly.exe'
if ($isPoly) {
    if ($draft) { Remove-Flag $argsList '--no-draft' } else { Add-Flag $argsList '--no-draft' }
} else {
    if ($draft) { Set-EngineArg $argsList '--spec' '4' } else { Set-EngineArg $argsList '--spec' '1' }
}

$vision = Read-YesNo 'Enable vision / image input?' $false
if ($vision -and -not $config.PSObject.Properties['vision']) {
    Write-Host 'No vision encoder is installed in this preset.' -ForegroundColor Yellow
    $visionJson = Read-Host 'Path to a prepared PolyStrata vision-section JSON, or blank to keep vision disabled'
    if ($null -eq $visionJson) { $visionJson = '' } else { $visionJson = $visionJson.Trim().Trim('"') }
    if ($visionJson) {
        if (-not (Test-Path -LiteralPath $visionJson -PathType Leaf)) { throw "Vision JSON not found: $visionJson" }
        $config | Add-Member -NotePropertyName vision -NotePropertyValue (Get-Content $visionJson -Raw | ConvertFrom-Json) -Force
    } else { $vision = $false }
} elseif (-not $vision -and $config.PSObject.Properties['vision']) {
    $config.PSObject.Properties.Remove('vision')
}

$compressedKv = Read-YesNo 'Use compressed Q8/int8 KV cache for more long-context headroom?' ($context -ge 131072)
if ($compressedKv) {
    Set-EngineArg $argsList '--kv' $(if ($isPoly) {'q8_0'} else {'int8'})
} else { Remove-EngineArg $argsList '--kv' }

$apiMonitor = Read-YesNo 'Enable the in-memory API request monitor?' $false
$openBrowser = $true
$lazy = $false

while ($true) {
    $portRaw = Read-Host "Server port [$portDefault]"
    if ($null -eq $portRaw) { $portRaw = '' } else { $portRaw = $portRaw.Trim() }
    if (-not $portRaw) { $port = $portDefault; break }
    $port = 0
    if ([int]::TryParse($portRaw,[ref]$port) -and $port -ge 1 -and $port -le 65535) { break }
    Write-Host 'Enter a port from 1 to 65535.' -ForegroundColor Yellow
}

Write-Host '  1) Localhost only (127.0.0.1)' -ForegroundColor Cyan
Write-Host '  2) LAN access (0.0.0.0, API key required)' -ForegroundColor Yellow
$network = Read-Menu 'Network access' 1 2 1
$hostAddress = if ($network -eq 2) { '0.0.0.0' } else { '127.0.0.1' }
$apiKey = ''
if ($network -eq 2) {
    $apiKey = Read-Host 'API key (blank = generate one)'
    if ($null -eq $apiKey) { $apiKey = '' } else { $apiKey = $apiKey.Trim() }
    if (-not $apiKey) { $apiKey = [guid]::NewGuid().ToString('N') }
    Write-Host "LAN API key: $apiKey" -ForegroundColor Yellow
    Write-Host 'Use this as Authorization: Bearer <key>. It is not written to the JSON.'
}

$config.args = @($argsList)
$config.port = $port
$config | Add-Member -NotePropertyName display_name -NotePropertyValue $displayName -Force
$config | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $LastConfig -Encoding UTF8

Write-Host
Write-Host 'Launch summary' -ForegroundColor Cyan
Write-Host "  Model:    $displayName"
Write-Host "  Context:  $context"
Write-Host "  Drafting: $draft"
Write-Host "  Vision:   $vision"
Write-Host "  KV:       $(if($compressedKv){'compressed'}else{'default'})"
Write-Host "  Python:   $PythonVersion ($Python)"
Write-Host "  URL:      http://$(if($network -eq 2){'<this-PC-IP>'}else{'127.0.0.1'}):$port/"
Write-Host "  Config:   $LastConfig"
Write-Host

if ($env:STRATA_M40_DRY_RUN -eq '1') {
    Write-Host 'Dry run requested; server was not started.' -ForegroundColor Yellow
    exit 0
}

if ($apiKey) { $env:STRATA_API_KEY = $apiKey } else { Remove-Item Env:STRATA_API_KEY -ErrorAction SilentlyContinue }
$serverArgs = @((Join-Path $Root 'serve\server.py'),'--engine','strata','--config',$LastConfig,'--port',[string]$port,'--host',$hostAddress)
if ($openBrowser) { $serverArgs += '--open' }
if ($apiMonitor) { $serverArgs += '--api-monitor' }
if ($lazy) { $serverArgs += '--lazy' }
& $Python @serverArgs
exit $LASTEXITCODE
