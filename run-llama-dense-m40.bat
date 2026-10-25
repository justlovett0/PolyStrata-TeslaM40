@echo off
setlocal EnableExtensions
title llama.cpp Dense M40 Launcher
set "DENSE_SELF=%~f0"
set "DENSE_MODEL_INPUT=%~1"
if not defined LLAMA_CPP_DIR set "LLAMA_CPP_DIR=%~dp0third_party\llama.cpp"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$p=$env:DENSE_SELF; $s=Get-Content -LiteralPath $p; $m=[Array]::IndexOf($s,'#__LLAMA_DENSE_M40_POWERSHELL__'); if($m -lt 0){throw 'Embedded launcher code is missing'}; & ([scriptblock]::Create(($s[($m+1)..($s.Length-1)] -join [Environment]::NewLine))) -Root '%~dp0'"
set "DENSE_EXIT=%ERRORLEVEL%"
if not "%DENSE_EXIT%"=="0" (
  echo.
  echo llama.cpp stopped with exit code %DENSE_EXIT%.
  pause
)
exit /b %DENSE_EXIT%

#__LLAMA_DENSE_M40_POWERSHELL__
param([string]$Root)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root)
$ModelInput = $env:DENSE_MODEL_INPUT
$LastConfig = Join-Path $Root 'llama-dense-m40-last-run.json'

function Banner {
    Clear-Host
    Write-Host '===============================================================' -ForegroundColor DarkCyan
    Write-Host '  llama.cpp Dense M40 Launcher  |  Tesla M40 SM 5.2' -ForegroundColor Cyan
    Write-Host '===============================================================' -ForegroundColor DarkCyan
    Write-Host '  CUDA 12.0 / native SM 5.2 / 24 GB class Maxwell GPU'
    Write-Host '  Uses a separately built llama-server.exe, not strata-poly.'
    Write-Host '  The bundled SM52 build is headless API mode; its optional Web UI is not included.' -ForegroundColor DarkGray
    Write-Host '  A reviewable copy is saved as llama-dense-m40-last-run.json.'
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

function Read-PathOrDefault([string]$Prompt, [string]$Default = '') {
    $promptText = if ($Default) { "$Prompt [$Default]" } else { $Prompt }
    $answer = Read-Host $promptText
    $raw = if ($null -eq $answer) { '' } else { $answer.Trim().Trim('"') }
    if (-not $raw) { return $Default }
    return $raw
}

function Read-GgufString([IO.BinaryReader]$Reader) {
    $length = [int64]$Reader.ReadUInt64()
    if ($length -lt 0 -or $length -gt 64MB) { throw "Invalid GGUF string length: $length" }
    $bytes = $Reader.ReadBytes([int]$length)
    if ($bytes.Length -ne $length) { throw 'Unexpected end of GGUF while reading a string' }
    return [Text.Encoding]::UTF8.GetString($bytes)
}

function Skip-GgufValue([IO.BinaryReader]$Reader, [uint32]$Type) {
    switch ($Type) {
        0 { [void]$Reader.ReadByte() }
        1 { [void]$Reader.ReadSByte() }
        2 { [void]$Reader.ReadUInt16() }
        3 { [void]$Reader.ReadInt16() }
        4 { [void]$Reader.ReadUInt32() }
        5 { [void]$Reader.ReadInt32() }
        6 { [void]$Reader.ReadSingle() }
        7 { [void]$Reader.ReadByte() }
        8 { [void](Read-GgufString $Reader) }
        9 {
            $elementType = $Reader.ReadUInt32()
            $count = [int64]$Reader.ReadUInt64()
            if ($count -lt 0 -or $count -gt 100000000) { throw "Invalid GGUF array length: $count" }
            for ($i = 0; $i -lt $count; $i++) { Skip-GgufValue $Reader $elementType }
        }
        10 { [void]$Reader.ReadUInt64() }
        11 { [void]$Reader.ReadInt64() }
        12 { [void]$Reader.ReadDouble() }
        default { throw "Unknown GGUF metadata type: $Type" }
    }
}

function Read-GgufScalar([IO.BinaryReader]$Reader, [uint32]$Type) {
    switch ($Type) {
        0 { return [uint64]$Reader.ReadByte() }
        1 { return [int64]$Reader.ReadSByte() }
        2 { return [uint64]$Reader.ReadUInt16() }
        3 { return [int64]$Reader.ReadInt16() }
        4 { return [uint64]$Reader.ReadUInt32() }
        5 { return [int64]$Reader.ReadInt32() }
        6 { return [double]$Reader.ReadSingle() }
        7 { return [bool]($Reader.ReadByte() -ne 0) }
        8 { return Read-GgufString $Reader }
        9 { Skip-GgufValue $Reader $Type; return $null }
        10 { return [uint64]$Reader.ReadUInt64() }
        11 { return [int64]$Reader.ReadInt64() }
        12 { return [double]$Reader.ReadDouble() }
        default { throw "Unknown GGUF metadata type: $Type" }
    }
}

function Read-GgufMetadata([string]$Path) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $reader = New-Object IO.BinaryReader($stream)
    try {
        $magic = $reader.ReadUInt32()
        if ($magic -ne [uint32]0x46554747) { throw 'Not a GGUF file' }
        $version = $reader.ReadUInt32()
        if ($version -lt 2 -or $version -gt 3) { throw "Unsupported GGUF version: $version" }
        [void]$reader.ReadUInt64() # tensor count
        $kvCount = [int64]$reader.ReadUInt64()
        $meta = @{}
        for ($i = 0; $i -lt $kvCount; $i++) {
            $key = Read-GgufString $reader
            # Tokenizer arrays can contain hundreds of thousands of strings. They
            # are not needed for sizing or launching, and parsing them through
            # PowerShell makes a several-GB GGUF appear to hang. GGUF writers
            # conventionally place tokenizer metadata after model metadata.
            if ($key -like 'tokenizer.*') { break }
            $type = $reader.ReadUInt32()
            $value = Read-GgufScalar $reader $type
            if ($null -ne $value) { $meta[$key] = $value }
        }
        return $meta
    } finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function Get-MetaValue([hashtable]$Metadata, [string]$Architecture, [string]$Suffix) {
    $key = "$Architecture.$Suffix"
    if ($Metadata.ContainsKey($key)) { return $Metadata[$key] }
    return $null
}

function Get-FileTypeName([object]$Value) {
    if ($null -eq $Value) { return 'unknown' }
    switch ([int64]$Value) {
        0 { 'F32' }; 1 { 'F16' }; 2 { 'Q4_0' }; 3 { 'Q4_1' }
        6 { 'Q5_0' }; 7 { 'Q5_1' }; 8 { 'Q8_0' }; 9 { 'Q8_1' }
        10 { 'Q2_K' }; 11 { 'Q3_K_S' }; 12 { 'Q3_K_M' }; 13 { 'Q3_K_L' }
        14 { 'Q4_K_S' }; 15 { 'Q4_K_M' }; 16 { 'Q5_K_S' }; 17 { 'Q5_K_M' }
        18 { 'Q6_K' }; 19 { 'IQ2_XXS' }; 20 { 'IQ2_XS' }; 21 { 'IQ3_XXS' }
        22 { 'IQ1_S' }; 23 { 'IQ4_NL' }; 24 { 'IQ3_S' }; 25 { 'IQ2_S' }
        26 { 'IQ4_XS' }; 27 { 'IQ1_M' }; 28 { 'BF16' }
        default { "file_type_$Value" }
    }
}

function Get-GgufSetBytes([object[]]$Files) {
    return [int64](($Files | Measure-Object -Property Length -Sum).Sum)
}

function Get-GgufInfo([object]$First, [object[]]$Files) {
    $metadata = Read-GgufMetadata $First.FullName
    $architecture = if ($metadata.ContainsKey('general.architecture')) { [string]$metadata['general.architecture'] } else { 'unknown' }
    $name = if ($metadata.ContainsKey('general.name')) { [string]$metadata['general.name'] } else { [IO.Path]::GetFileNameWithoutExtension($First.Name) }
    $trainedContext = Get-MetaValue $metadata $architecture 'context_length'
    $blocks = Get-MetaValue $metadata $architecture 'block_count'
    $embedding = Get-MetaValue $metadata $architecture 'embedding_length'
    $headKv = Get-MetaValue $metadata $architecture 'attention.head_count_kv'
    $headCount = Get-MetaValue $metadata $architecture 'attention.head_count'
    $keyLength = Get-MetaValue $metadata $architecture 'attention.key_length'
    $valueLength = Get-MetaValue $metadata $architecture 'attention.value_length'
    if ($null -eq $headKv) { $headKv = $headCount }
    if ($null -eq $keyLength -and $headCount -and $embedding) { $keyLength = [int64]$embedding / [int64]$headCount }
    if ($null -eq $valueLength) { $valueLength = $keyLength }
    $fileTypeValue = $null
    if ($metadata.ContainsKey('general.file_type')) { $fileTypeValue = $metadata['general.file_type'] }
    [pscustomobject]@{
        First = $First.FullName
        Files = @($Files | Sort-Object Name | ForEach-Object FullName)
        Name = $name
        Architecture = $architecture
        Quantization = Get-FileTypeName $fileTypeValue
        Bytes = Get-GgufSetBytes $Files
        Parts = $Files.Count
        TrainedContext = if ($trainedContext) { [int64]$trainedContext } else { 0 }
        Blocks = if ($blocks) { [int64]$blocks } else { 0 }
        HeadKv = if ($headKv) { [int64]$headKv } else { 0 }
        KeyLength = if ($keyLength) { [int64]$keyLength } else { 0 }
        ValueLength = if ($valueLength) { [int64]$valueLength } else { 0 }
        Metadata = $metadata
    }
}

function Get-GgufSets([string]$InputPath) {
    $p = $InputPath.Trim().Trim('"')
    if (-not $p) { $p = Read-Host 'Folder containing dense GGUF file(s), or a full GGUF path' }
    if (-not (Test-Path -LiteralPath $p)) { throw "Path does not exist: $p" }
    if (Test-Path -LiteralPath $p -PathType Leaf) {
        $file = Get-Item -LiteralPath $p
        if ($file.Extension -ine '.gguf') { throw "Not a GGUF file: $p" }
        return ,@([pscustomobject]@{ First=$file; Files=@($file) })
    }
    $files = @(Get-ChildItem -LiteralPath $p -Filter '*.gguf' -File -Recurse -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -notmatch '^ggml-vocab-' })
    if ($files.Count -eq 0) { throw "No model .gguf files were found under $p" }
    $groups = @{}
    foreach ($file in $files) {
        if ($file.Name -match '^(.*)-(\d{5})-of-(\d{5})\.gguf$') {
            $key = "$($file.DirectoryName)|$($Matches[1])|$($Matches[3])"
        } else {
            $key = $file.FullName
        }
        if (-not $groups.ContainsKey($key)) { $groups[$key] = New-Object System.Collections.ArrayList }
        [void]$groups[$key].Add($file)
    }
    return @($groups.Values | ForEach-Object {
        $setFiles = @($_ | Sort-Object Name)
        $first = $setFiles | Where-Object { $_.Name -match '-00001-of-' } | Select-Object -First 1
        if (-not $first) { $first = $setFiles[0] }
        [pscustomobject]@{ First=$first; Files=$setFiles }
    } | Sort-Object { $_.First.FullName })
}

function Choose-Gguf([object[]]$Sets) {
    $infos = @()
    foreach ($set in $Sets) {
        try { $infos += Get-GgufInfo $set.First $set.Files }
        catch { Write-Host "Skipping unreadable GGUF: $($set.First.FullName) ($($_.Exception.Message))" -ForegroundColor Yellow }
    }
    if ($infos.Count -eq 0) { throw 'No readable model GGUF was found.' }
    if ($infos.Count -eq 1) { return $infos[0] }
    Write-Host 'Dense GGUF candidates:' -ForegroundColor Cyan
    for ($i = 0; $i -lt $infos.Count; $i++) {
        $size = [math]::Round($infos[$i].Bytes / 1GB, 2)
        Write-Host ("  {0}) {1} | {2} | {3} | {4:N2} GiB | {5} part(s)" -f ($i+1),$infos[$i].Name,$infos[$i].Architecture,$infos[$i].Quantization,$size,$infos[$i].Parts)
        Write-Host ("     {0}" -f $infos[$i].First) -ForegroundColor DarkGray
    }
    return $infos[(Read-Menu 'Choose the model or first shard' 1 $infos.Count 1)-1]
}

function Get-ContextPlan([object]$Info) {
    $choices = @(4096,8192,16384,32768,65536)
    $weightGiB = $Info.Bytes / 1GB
    $cap = if ($weightGiB -gt 17) { 8192 } elseif ($weightGiB -gt 12) { 16384 } elseif ($weightGiB -gt 7) { 32768 } else { 65536 }
    $kvPerToken = 0.0
    if ($Info.Blocks -gt 0 -and $Info.HeadKv -gt 0 -and $Info.KeyLength -gt 0 -and $Info.ValueLength -gt 0) {
        # q8_0 KV is approximately one byte per K/V value, plus scale overhead.
        $kvPerToken = [double]$Info.Blocks * [double]$Info.HeadKv * ([double]$Info.KeyLength + [double]$Info.ValueLength) * 1.15
        $available = (23040MB) - $Info.Bytes - (1536MB)
        if ($available -gt 0) { $cap = [math]::Min($cap, [math]::Floor($available / $kvPerToken / 4096) * 4096) }
    }
    if ($Info.TrainedContext -gt 0) { $cap = [math]::Min($cap, $Info.TrainedContext) }
    $valid = @($choices | Where-Object { $_ -le $cap })
    if ($valid.Count -eq 0) { $valid = @(4096) }
    $recommended = $valid[-1]
    if ($weightGiB -gt 12 -and $recommended -gt 16384) { $recommended = 16384 }
    if ($weightGiB -gt 17 -and $recommended -gt 8192) { $recommended = 8192 }
    $near = ($valid | Where-Object { $_ -gt $recommended } | Select-Object -First 1)
    if (-not $near) { $near = $recommended }
    [pscustomobject]@{ Choices=$valid; Recommended=[int]$recommended; NearLimit=[int]$near; KvBytes=[double]$kvPerToken }
}

function Find-LlamaServer([string]$Base) {
    $candidates = @()
    if ($env:LLAMA_SERVER_EXE) { $candidates += $env:LLAMA_SERVER_EXE }
    $llamaRoots = @()
    if ($env:LLAMA_CPP_DIR) { $llamaRoots += $env:LLAMA_CPP_DIR }
    $llamaRoots += @(
        (Join-Path $Base 'third_party\llama.cpp'),
        (Join-Path $Base 'llama.cpp'),
        (Join-Path $Base 'llama-build')
    )
    foreach ($llamaRoot in ($llamaRoots | Select-Object -Unique)) {
        $candidates += @(
            (Join-Path $llamaRoot 'build\bin\llama-server.exe'),
            (Join-Path $llamaRoot 'build\bin\Release\llama-server.exe'),
            (Join-Path $llamaRoot 'build\bin\RelWithDebInfo\llama-server.exe'),
            (Join-Path $llamaRoot 'bin\llama-server.exe'),
            (Join-Path $llamaRoot 'bin\Release\llama-server.exe'),
            (Join-Path $llamaRoot 'bin\RelWithDebInfo\llama-server.exe')
        )
    }
    $command = Get-Command llama-server.exe -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return [IO.Path]::GetFullPath($candidate) }
    }
    $sourceTree = $llamaRoots | Where-Object { Test-Path -LiteralPath $_ -PathType Container } | Select-Object -First 1
    if ($sourceTree) {
        Write-Host "Found llama.cpp source tree: $([IO.Path]::GetFullPath($sourceTree))" -ForegroundColor Cyan
        Write-Host 'No built llama-server.exe exists in that source tree yet.' -ForegroundColor Yellow
        Write-Host 'Build the SM52 server into build\bin (or set LLAMA_SERVER_EXE to an existing executable).'
    } else {
        Write-Host 'No llama.cpp source tree or llama-server.exe was found.' -ForegroundColor Yellow
        Write-Host 'Set LLAMA_CPP_DIR to your llama.cpp checkout or LLAMA_SERVER_EXE to the executable.'
    }
    $manual = Read-PathOrDefault 'llama-server.exe path'
    if (-not $manual) { throw 'No llama-server.exe path was provided.' }
    if (-not (Test-Path -LiteralPath $manual -PathType Leaf)) { throw "llama-server.exe not found: $manual" }
    return [IO.Path]::GetFullPath($manual)
}

function Get-GpuText {
    try {
        $text = & nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($text) { return $text.Trim() }
    } catch {}
    return 'nvidia-smi information unavailable'
}

function Add-Arg([System.Collections.Generic.List[string]]$List, [string]$Name, [string]$Value) {
    [void]$List.Add($Name); [void]$List.Add($Value)
}

Banner
$server = $null
Write-Host 'Scanning GGUF metadata (model fields only)...' -ForegroundColor DarkGray
$info = Choose-Gguf (Get-GgufSets $ModelInput)
$plan = Get-ContextPlan $info
$server = Find-LlamaServer $Root
$serverDir = Split-Path -Parent $server
$sizeGiB = [math]::Round($info.Bytes / 1GB, 2)
$gpuText = Get-GpuText

Write-Host
Write-Host "Model:        $($info.Name)" -ForegroundColor Green
Write-Host "Architecture: $($info.Architecture)"
Write-Host "Quantization: $($info.Quantization)"
Write-Host ("GGUF set:     {0:N2} GiB in {1} part(s)" -f $sizeGiB,$info.Parts)
Write-Host "GPU:          $gpuText"
if ($info.TrainedContext -gt 0) { Write-Host ("Trained context: {0:N0} tokens" -f $info.TrainedContext) }
Write-Host ("Context estimate: recommended {0:N0}; near-limit {1:N0}." -f $plan.Recommended,$plan.NearLimit) -ForegroundColor Cyan
Write-Host 'This estimate reserves VRAM for CUDA, scratch buffers, and a q8_0 KV cache.'

for ($i = 0; $i -lt $plan.Choices.Count; $i++) {
    $ctx = $plan.Choices[$i]
    $tag = if ($ctx -eq $plan.Recommended) { ' [recommended]' } elseif ($ctx -eq $plan.NearLimit) { ' [near limit]' } else { '' }
    Write-Host ("  {0}) {1}K{2}" -f ($i+1),[int]($ctx/1024),$tag)
}
$ctxPick = Read-Menu 'Context window' 1 $plan.Choices.Count ([Array]::IndexOf($plan.Choices,$plan.Recommended)+1)
$context = [int]$plan.Choices[$ctxPick-1]

Write-Host
Write-Host 'GPU offload:' -ForegroundColor Cyan
Write-Host '  1) All possible layers (recommended)'
Write-Host '  2) CPU only'
Write-Host '  3) Enter a custom GPU layer count'
$offloadPick = Read-Menu 'GPU offload' 1 3 1
if ($offloadPick -eq 1) { $gpuLayers = '99' }
elseif ($offloadPick -eq 2) { $gpuLayers = '0' }
else { $gpuLayers = Read-PathOrDefault 'GPU layers' '99'; if ($gpuLayers -notmatch '^\d+$') { throw 'GPU layers must be a non-negative integer.' } }

Write-Host
Write-Host 'KV cache:' -ForegroundColor Cyan
Write-Host '  1) q8_0 K/V (recommended on a 24 GB M40)'
Write-Host '  2) f16 K/V (more VRAM, sometimes slightly better quality)'
$kvPick = Read-Menu 'KV cache type' 1 2 1
$kvType = if ($kvPick -eq 1) { 'q8_0' } else { 'f16' }

Write-Host
$flash = Read-YesNo 'Enable flash attention? (SM52 default is No)' $false
$batchDefault = if ($sizeGiB -gt 12) { 512 } else { 1024 }
Write-Host
Write-Host 'Prompt batch size:' -ForegroundColor Cyan
Write-Host '  1) 256  - lowest VRAM / safest'
Write-Host '  2) 512  - recommended for larger dense models'
Write-Host '  3) 1024 - recommended for smaller models'
Write-Host '  4) 2048 - experimental on this GPU'
$batchPickDefault = if ($batchDefault -eq 512) { 2 } else { 3 }
$batchPick = Read-Menu 'Prompt batch size' 1 4 $batchPickDefault
$batch = @(256,512,1024,2048)[$batchPick-1]
$ubatch = [math]::Min($batch, 512)

Write-Host
$parallel = Read-Menu 'Parallel server slots' 1 2 1
$portDefault = 8090
while ($true) {
    $portRaw = Read-Host "Server port [$portDefault]"
    $portRaw = if ($null -eq $portRaw) { '' } else { $portRaw.Trim() }
    if (-not $portRaw) { $port = $portDefault; break }
    $port = 0
    if ([int]::TryParse($portRaw,[ref]$port) -and $port -ge 1 -and $port -le 65535) { break }
    Write-Host 'Enter a port from 1 to 65535.' -ForegroundColor Yellow
}

Write-Host
Write-Host '  1) Localhost only (127.0.0.1)' -ForegroundColor Cyan
Write-Host '  2) LAN access (0.0.0.0, API key required)' -ForegroundColor Yellow
$network = Read-Menu 'Network access' 1 2 1
$hostAddress = if ($network -eq 2) { '0.0.0.0' } else { '127.0.0.1' }
$apiKey = ''
if ($network -eq 2) {
    $apiKey = Read-Host 'API key (blank = generate one)'
    $apiKey = if ($null -eq $apiKey) { '' } else { $apiKey.Trim() }
    if (-not $apiKey) { $apiKey = [guid]::NewGuid().ToString('N') }
    Write-Host "LAN API key: $apiKey" -ForegroundColor Yellow
    Write-Host 'Use this as Authorization: Bearer <key>. It is not written to the JSON.'
}
$openBrowser = Read-YesNo 'Open the server URL in a browser? (headless build has no embedded UI)' $false

$launchArgs = New-Object 'System.Collections.Generic.List[string]'
Add-Arg $launchArgs '-m' $info.First
Add-Arg $launchArgs '-c' ([string]$context)
Add-Arg $launchArgs '-ngl' ([string]$gpuLayers)
Add-Arg $launchArgs '-b' ([string]$batch)
Add-Arg $launchArgs '-ub' ([string]$ubatch)
Add-Arg $launchArgs '-ctk' $kvType
Add-Arg $launchArgs '-ctv' $kvType
$flashMode = if ($flash) { 'on' } else { 'off' }
Add-Arg $launchArgs '-fa' $flashMode
Add-Arg $launchArgs '-np' ([string]$parallel)
Add-Arg $launchArgs '--host' $hostAddress
Add-Arg $launchArgs '--port' ([string]$port)
[void]$launchArgs.Add('--jinja')
[void]$launchArgs.Add('--no-ui')
if ($apiKey) { Add-Arg $launchArgs '--api-key' $apiKey }

$savedArgs = @($launchArgs | ForEach-Object { [string]$_ })
if ($apiKey) {
    $savedArgs = @($savedArgs | Where-Object { $_ -ne '--api-key' -and $_ -ne $apiKey })
}
$savedConfig = [ordered]@{
    executable = $server
    model = $info.First
    gguf_parts = @($info.Files)
    model_name = $info.Name
    architecture = $info.Architecture
    quantization = $info.Quantization
    gguf_gib = $sizeGiB
    context = $context
    gpu_layers = [int]$gpuLayers
    kv_cache = $kvType
    flash_attention = $flash
    batch = $batch
    ubatch = $ubatch
    parallel = $parallel
    host = $hostAddress
    port = $port
    args = $savedArgs
    api_key_saved = $false
}
[ordered]@{ settings=$savedConfig; metadata=$info.Metadata } |
    ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $LastConfig -Encoding UTF8

Write-Host
Write-Host 'Launch summary' -ForegroundColor Cyan
Write-Host "  Model:      $($info.Name)"
Write-Host "  Architecture: $($info.Architecture)"
Write-Host "  Context:    $context"
Write-Host "  GPU layers: $gpuLayers"
Write-Host "  KV cache:   $kvType"
Write-Host "  Flash attn: $(if ($flash) {'on'} else {'off'})"
Write-Host "  Batch:      $batch / ubatch $ubatch"
Write-Host "  Slots:      $parallel"
Write-Host "  URL:        http://$(if ($network -eq 2) {'<this-PC-IP>'} else {'127.0.0.1'}):$port/"
Write-Host "  Server:     $server"
Write-Host "  Config:     $LastConfig"
Write-Host

if ($env:LLAMA_DENSE_M40_DRY_RUN -eq '1') {
    Write-Host 'Dry run requested; llama-server was not started.' -ForegroundColor Yellow
    exit 0
}

$browserJob = $null
if ($openBrowser) {
    $browserHost = if ($network -eq 2) { '127.0.0.1' } else { $hostAddress }
    $browserUrl = "http://$browserHost`:$port/"
    $browserJob = Start-Job -ArgumentList $browserHost,$port,$browserUrl -ScriptBlock {
        param($HostName,$PortNumber,$Url)
        for ($try = 0; $try -lt 120; $try++) {
            $client = $null
            try {
                $client = New-Object Net.Sockets.TcpClient
                $task = $client.ConnectAsync($HostName,[int]$PortNumber)
                if ($task.Wait(500) -and $client.Connected) {
                    $client.Close()
                    Start-Process $Url
                    break
                }
            } catch {} finally { if ($client) { $client.Dispose() } }
            Start-Sleep -Milliseconds 250
        }
    }
}

Push-Location $serverDir
try {
    & $server @launchArgs
    $serverExit = $LASTEXITCODE
} finally {
    Pop-Location
    if ($browserJob) { Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue }
}
exit $serverExit
