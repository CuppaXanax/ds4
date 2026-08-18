#requires -Version 7.0
[CmdletBinding()]
param(
    [switch]$AllowDirtyEngineeringSystem
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
    if (-not $isBaselineBranch) {
        throw "Expected runtime branch $($manifest.runtime_lkg.branch); found $branch"
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
    $localBranches = @(Invoke-GitChecked for-each-ref --format="%(refname:short)" refs/heads/)
    $unexpectedBranches = @($localBranches | Where-Object { $_ -notin $allowedBranches })
    if ($unexpectedBranches.Count -ne 0) {
        throw "Unexpected local branches: $($unexpectedBranches -join ', ')"
    }

    $allowedPattern = '^(?:AGENTS\.md|\.gitattributes|\.gitignore|\.githooks/|performance/bc250/|vulkan/shaders/(?:compile\.py|test_compile\.py)$)'
    $changedFromLkg = @(
        Invoke-GitChecked diff --name-only $baselineCommit --
        Invoke-GitChecked ls-files --others --exclude-standard
    ) | Sort-Object -Unique
    $runtimeChanges = @($changedFromLkg | Where-Object { $_ -notmatch $allowedPattern })
    if ($runtimeChanges.Count -ne 0) {
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

    $coordinatorProfile = $manifest.runtime_lkg.runtime_shader_profiles.coordinator
    $workerProfile = $manifest.runtime_lkg.runtime_shader_profiles.worker
    if (-not [bool]$manifest.runtime_lkg.uniform_shader_profile_required -or
        [int]$coordinatorProfile.shader_count -ne [int]$workerProfile.shader_count -or
        [string]$coordinatorProfile.shader_manifest_sha256 -ne
            [string]$workerProfile.shader_manifest_sha256) {
        throw "Runtime LKG does not declare one uniform shader profile"
    }

    $hex64 = '^[0-9a-f]{64}$'
    if ([long]$manifest.model.size_bytes -ne 86720111488 -or
        ([string]$manifest.model.sha256).ToLowerInvariant() -notmatch $hex64 -or
        ([string]$manifest.model.sample_sha256).ToLowerInvariant() -notmatch $hex64 -or
        [string]$manifest.model.sample_scheme -ne
            'sha256(first_4mib || centered_4mib || last_4mib)') {
        throw "Runtime LKG model identity is incomplete or invalid"
    }

    $semantic = $manifest.runtime_lkg.promotion_evidence.semantic_qualification
    if ([string]$semantic.status -ne 'pass') {
        throw "Runtime LKG lacks a passing semantic qualification"
    }
    $fixturePath = Join-Path $repoPath ([string]$semantic.fixture)
    if (-not (Test-Path -LiteralPath $fixturePath -PathType Leaf) -or
        (Get-LfNormalizedSha256 -LiteralPath $fixturePath) -ne
            ([string]$semantic.fixture_sha256).ToLowerInvariant()) {
        throw "Official semantic fixture is missing or changed"
    }
    $semanticCases = @(
        @('short_reasoning_plain', '16', @(926)),
        @('short_italian_fact', 'Ada Lovelace', @(108149, 47121, 317, 805))
    )
    foreach ($caseSpec in $semanticCases) {
        $caseName = [string]$caseSpec[0]
        $expectedText = [string]$caseSpec[1]
        $expectedTokens = @($caseSpec[2])
        $case = $semantic.$caseName
        $casePromptPath = Join-Path $repoPath ([string]$case.prompt)
        if ([string]$case.expected -ne $expectedText -or
            [string]$case.actual -ne $expectedText -or
            (@($case.token_ids) -join ',') -ne ($expectedTokens -join ',') -or
            -not (Test-Path -LiteralPath $casePromptPath -PathType Leaf) -or
            (Get-LfNormalizedSha256 -LiteralPath $casePromptPath) -ne
                ([string]$case.prompt_sha256).ToLowerInvariant()) {
            throw "Official semantic case $caseName is missing, changed, or failing"
        }
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
        mode = "baseline"
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
