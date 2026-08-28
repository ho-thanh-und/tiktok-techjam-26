[CmdletBinding()]
param(
    [switch]$KeepArchive
)

$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$archive = Join-Path $workspace 'KuaiRand-Pure.tar.gz'
$datasetRoot = Join-Path $workspace 'KuaiRand-Pure'
$dataTarget = Join-Path $datasetRoot 'data'
$manifestTarget = Join-Path $workspace 'data_manifests\kuairand-pure-public.json'
$downloadUrl = 'https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz'
$expectedMd5 = '0820331067a3784d9691136f772b35a7'
$staging = Join-Path $workspace ('.dataset-staging-' + [Guid]::NewGuid().ToString('N'))

function Assert-SafeStagingPath([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    $prefix = $workspace.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Staging path escapes workspace: $resolved"
    }
    if ([System.IO.Path]::GetFileName($resolved) -notlike '.dataset-staging-*') {
        throw "Unexpected staging directory name: $resolved"
    }
}

if (-not (Test-Path -LiteralPath $dataTarget -PathType Container)) {
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        & curl.exe -L --fail --retry 3 --output $archive $downloadUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed with exit code $LASTEXITCODE"
        }
    }

    $actualMd5 = (Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actualMd5 -ne $expectedMd5) {
        throw "Archive checksum mismatch: expected $expectedMd5, got $actualMd5"
    }

    $entries = @(& tar -tf $archive)
    if ($LASTEXITCODE -ne 0 -or $entries.Count -eq 0) {
        throw 'Unable to list archive contents'
    }
    $unsafe = $entries | Where-Object {
        [System.IO.Path]::IsPathRooted($_) -or $_ -match '(^|[\\/])\.\.([\\/]|$)'
    }
    if ($unsafe) {
        throw "Archive contains unsafe paths: $($unsafe -join ', ')"
    }

    Assert-SafeStagingPath $staging
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        & tar -xf $archive -C $staging
        if ($LASTEXITCODE -ne 0) {
            throw "Extraction failed with exit code $LASTEXITCODE"
        }
        $stagedRoot = Join-Path $staging 'KuaiRand-Pure'
        $stagedData = Join-Path $stagedRoot 'data'
        if (-not (Test-Path -LiteralPath $stagedData -PathType Container)) {
            throw 'Extracted archive does not contain KuaiRand-Pure/data'
        }
        New-Item -ItemType Directory -Path $datasetRoot -Force | Out-Null
        Move-Item -LiteralPath $stagedData -Destination $dataTarget
        $stagedLicense = Join-Path $stagedRoot 'LICENSE'
        if (Test-Path -LiteralPath $stagedLicense -PathType Leaf) {
            Move-Item -LiteralPath $stagedLicense -Destination (Join-Path $datasetRoot 'LICENSE') -Force
        }
    }
    finally {
        if (Test-Path -LiteralPath $staging) {
            Assert-SafeStagingPath $staging
            Remove-Item -LiteralPath $staging -Recurse -Force
        }
    }
}

if (-not $KeepArchive -and (Test-Path -LiteralPath $archive -PathType Leaf)) {
    $actualMd5 = (Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actualMd5 -eq $expectedMd5) {
        Remove-Item -LiteralPath $archive -Force
    }
}

& python -m automl_agent.kuairand_manifest `
    --data-dir $dataTarget `
    --archive-md5 $expectedMd5 `
    --output $manifestTarget
if ($LASTEXITCODE -ne 0) {
    throw "Manifest generation failed with exit code $LASTEXITCODE"
}

Write-Host "KuaiRand-Pure is ready at $dataTarget"
Write-Host "Manifest: $manifestTarget"

