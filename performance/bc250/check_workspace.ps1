#requires -Version 7.0
[CmdletBinding()]
param(
    [switch]$AllowDirtyEngineeringSystem,
    [string]$CandidateBranch
)

$ErrorActionPreference = "Stop"

function Invoke-GitChecked {
    param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    # Keep harmless Git warnings (notably autocrlf diagnostics) out of the
    # machine-readable stdout used by the checks below.
    $output = & git @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed"
    }
    return @($output)
}

function Get-LfNormalizedSha256 {
    param([Parameter(Mandatory)][string]$LiteralPath)
    $utf8 = [Text.UTF8Encoding]::new($false, $true)
    $text = $utf8.GetString([IO.File]::ReadAllBytes($LiteralPath))
    $normalized = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    $bytes = $utf8.GetBytes($normalized)
    return [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($bytes)
    ).ToLowerInvariant()
}

$repoPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$manifestPath = Join-Path $PSScriptRoot "lkg.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$baselineCommit = [string]$manifest.runtime_lkg.commit

Push-Location $repoPath
try {
    $branch = (Invoke-GitChecked branch --show-current | Select-Object -First 1).Trim()
    $isBaselineBranch = $branch -eq $manifest.runtime_lkg.branch
    $isCandidateBranch = $CandidateBranch -and $branch -eq $CandidateBranch
    if (-not $isBaselineBranch -and -not $isCandidateBranch) {
        throw "Expected branch $($manifest.runtime_lkg.branch) or explicit -CandidateBranch; found $branch"
    }

    & git cat-file -e "$baselineCommit^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing baseline commit $baselineCommit"
    }
    $baselineTree = (Invoke-GitChecked rev-parse "$baselineCommit^{tree}" |
        Select-Object -First 1).Trim()
    if ($baselineTree -ne [string]$manifest.runtime_lkg.tree) {
        throw "Runtime LKG tree mismatch: $baselineTree"
    }
    & git merge-base --is-ancestor $baselineCommit HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "HEAD does not descend from runtime LKG $baselineCommit"
    }

    $worktreeCount = @(Invoke-GitChecked worktree list --porcelain |
        Where-Object { $_ -like "worktree *" }).Count
    if ($worktreeCount -ne 1) {
        throw "Expected one worktree; found $worktreeCount"
    }

    $allowedBranches = @("main", [string]$manifest.runtime_lkg.branch)
    if ($CandidateBranch) {
        $allowedBranches += $CandidateBranch
    }
    $localBranches = @(Invoke-GitChecked for-each-ref --format="%(refname:short)" refs/heads/)
    $unexpectedBranches = @($localBranches | Where-Object { $_ -notin $allowedBranches })
    if ($unexpectedBranches.Count -ne 0) {
        throw "Unexpected local branches: $($unexpectedBranches -join ', ')"
    }

    $allowedPattern = '^(?:AGENTS\.md|\.gitattributes|\.gitignore|\.githooks/|performance/bc250/)'
    $changedFromLkg = @(
        Invoke-GitChecked diff --name-only $baselineCommit --
        Invoke-GitChecked ls-files --others --exclude-standard
    ) | Sort-Object -Unique
    $runtimeChanges = @($changedFromLkg | Where-Object { $_ -notmatch $allowedPattern })
    if ($runtimeChanges.Count -ne 0 -and $isBaselineBranch) {
        throw "Runtime tree differs from LKG outside the engineering system: $($runtimeChanges -join ', ')"
    }

    $status = @(Invoke-GitChecked status --porcelain --untracked-files=all)
    if ($status.Count -ne 0 -and -not $AllowDirtyEngineeringSystem) {
        throw "Working tree is dirty:`n$($status -join "`n")"
    }
    if ($status.Count -ne 0) {
        $dirtyOutsideSystem = @($status | ForEach-Object {
            if ($_.Length -ge 4) { $_.Substring(3).Replace('\\', '/') }
        } | Where-Object { $_ -and $_ -notmatch $allowedPattern })
        if ($dirtyOutsideSystem.Count -ne 0) {
            throw "Dirty paths outside engineering system: $($dirtyOutsideSystem -join ', ')"
        }
    }

    $promptPath = Join-Path $repoPath ([string]$manifest.benchmark.prompt_path)
    if (-not (Test-Path -LiteralPath $promptPath -PathType Leaf)) {
        throw "Missing canonical prompt: $promptPath"
    }
    # The prompt is declared `eol=lf`. Hash that canonical form so a Windows
    # CRLF working copy and the Linux runtime checkout prove the same bytes.
    $promptHash = Get-LfNormalizedSha256 -LiteralPath $promptPath
    if ($promptHash -ne ([string]$manifest.benchmark.prompt_sha256).ToLowerInvariant()) {
        throw "Canonical prompt hash mismatch: $promptHash"
    }

    $archiveChecks = @($manifest.archives | ForEach-Object {
        $archivePath = [string]$_.path
        if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
            throw "Missing recovery archive: $archivePath"
        }
        $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
        if ($archiveHash -ne ([string]$_.sha256).ToLowerInvariant()) {
            throw "Recovery archive hash mismatch: $archivePath"
        }
        [pscustomobject]@{
            path = $archivePath
            sha256 = $archiveHash
            refs = [int]$_.refs
        }
    })

    [pscustomobject]@{
        verdict = "PASS"
        branch = $branch
        mode = $(if ($isBaselineBranch) { "baseline" } else { "candidate" })
        head = (Invoke-GitChecked rev-parse HEAD | Select-Object -First 1).Trim()
        runtime_lkg = $baselineCommit
        runtime_tree = $baselineTree
        worktrees = $worktreeCount
        local_branches = $localBranches
        runtime_changes = $runtimeChanges
        archives = $archiveChecks
    } | ConvertTo-Json -Depth 4
}
finally {
    Pop-Location
}
