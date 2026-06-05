param(
    [string]$Remote = "wout.decrop@MOC-GPU-3.vliz.be",
    [string]$RemoteProject = "/data/woutdecrop/projects/planktonclass",
    [string]$RemoteEnv = "/data/woutdecrop/envs/planktonclass-gpu",
    [string]$QarchiveRoot = "/mnt/qarchive_data_sensors",
    [string]$QarchiveCheckDir = "/mnt/qarchive_data_sensors/plankton-imager-10",
    [string]$QarchiveUser = "wout.decrop",
    [string]$QarchiveDomain = "vliz.be",
    [string]$SshKeyComment = "",
    [string]$ConfigPath = "",
    [int[]]$Gpus = @(0, 1),
    [switch]$SetupSshKey,
    [switch]$SyncOnly,
    [switch]$StartOnly,
    [switch]$SkipQarchiveRepair
)

$ErrorActionPreference = "Stop"

function Test-CommandAvailable {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Escape-SingleQuotedRemoteValue {
    param([string]$Value)
    return $Value.Replace("'", "'\''")
}

function Test-SshKeyLogin {
    param([string]$RemoteHost)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        $null = & ssh -o BatchMode=yes $RemoteHost "true" 2>&1
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Get-JsonValue {
    param(
        [object]$Object,
        [string[]]$Path
    )

    $current = $Object
    foreach ($name in $Path) {
        if ($null -eq $current) {
            return $null
        }

        $property = $current.PSObject.Properties[$name]
        if ($null -eq $property) {
            return $null
        }

        $current = $property.Value
    }

    return $current
}

function Import-PrivatePredictConfig {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Could not parse private config JSON: $Path"
    }
}

function Get-RemoteConfigValue {
    param(
        [object]$RemoteConfig,
        [object]$ProfileConfig,
        [string]$Name
    )

    $profileValue = Get-JsonValue -Object $ProfileConfig -Path @($Name)
    if ($null -ne $profileValue -and $profileValue -ne "") {
        return $profileValue
    }

    $remoteValue = Get-JsonValue -Object $RemoteConfig -Path @($Name)
    if ($null -ne $remoteValue -and $remoteValue -ne "") {
        return $remoteValue
    }

    return $null
}

function Test-RemoteQarchiveAccess {
    param(
        [string]$RemoteHost,
        [string]$CheckDir
    )

    $escapedCheckDir = Escape-SingleQuotedRemoteValue $CheckDir
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        $null = & ssh -o BatchMode=yes $RemoteHost "test -r '$escapedCheckDir' -a -x '$escapedCheckDir'" 2>&1
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Invoke-RemoteQarchiveRepair {
    param(
        [string]$RemoteHost,
        [string]$Root,
        [string]$CheckDir,
        [string]$User,
        [string]$Domain
    )

    $escapedRoot = Escape-SingleQuotedRemoteValue $Root
    $escapedCheckDir = Escape-SingleQuotedRemoteValue $CheckDir
    $escapedUser = Escape-SingleQuotedRemoteValue $User
    $escapedDomain = Escape-SingleQuotedRemoteValue $Domain

    $remoteCommand = @"
QARCHIVE_ROOT='$escapedRoot'
QARCHIVE_CHECK_DIR='$escapedCheckDir'
QARCHIVE_USER='$escapedUser'
QARCHIVE_DOMAIN='$escapedDomain'

if [ -r "`$QARCHIVE_CHECK_DIR" ] && [ -x "`$QARCHIVE_CHECK_DIR" ]; then
  echo "qarchive access OK: `$QARCHIVE_CHECK_DIR"
  exit 0
fi

echo "qarchive is not accessible for this Linux session."
echo "Adding CIFS credentials for `$QARCHIVE_USER@`$QARCHIVE_DOMAIN..."
cifscreds add -u "`$QARCHIVE_USER" -d "`$QARCHIVE_DOMAIN"

if ! mountpoint -q "`$QARCHIVE_ROOT"; then
  sudo mount "`$QARCHIVE_ROOT"
else
  sudo mount "`$QARCHIVE_ROOT" 2>/dev/null || true
fi

if [ -r "`$QARCHIVE_CHECK_DIR" ] && [ -x "`$QARCHIVE_CHECK_DIR" ]; then
  echo "qarchive access OK: `$QARCHIVE_CHECK_DIR"
else
  echo "qarchive is still not accessible: `$QARCHIVE_CHECK_DIR"
  exit 1
fi
"@

    & ssh -tt $RemoteHost $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "qarchive credential/mount repair failed."
    }
}

function Ensure-RemoteQarchiveAccess {
    param(
        [string]$RemoteHost,
        [string]$Root,
        [string]$CheckDir,
        [string]$User,
        [string]$Domain
    )

    if (Test-RemoteQarchiveAccess -RemoteHost $RemoteHost -CheckDir $CheckDir) {
        Write-Host "qarchive access OK: $CheckDir"
        return
    }

    Write-Host "qarchive is not accessible; opening an interactive credential repair step."
    Invoke-RemoteQarchiveRepair `
        -RemoteHost $RemoteHost `
        -Root $Root `
        -CheckDir $CheckDir `
        -User $User `
        -Domain $Domain
}

function Invoke-RsyncSync {
    param(
        [string]$SourceDir,
        [string]$RemoteHost,
        [string]$RemoteTarget
    )

    $rsyncSource = ($SourceDir -replace "\\", "/")
    if (-not $rsyncSource.EndsWith("/")) {
        $rsyncSource = "$rsyncSource/"
    }

    & rsync -az --delete `
        --exclude ".git/" `
        --exclude "__pycache__/" `
        --exclude "*.pyc" `
        --exclude "PI10/predict_gpu_config.json" `
        --exclude "**/.env" `
        --exclude "PI10/run_logs/" `
        $rsyncSource "${RemoteHost}:$RemoteTarget/"

    if ($LASTEXITCODE -ne 0) {
        throw "rsync failed."
    }
}

function Invoke-TarScpSync {
    param(
        [string]$SourceDir,
        [string]$RemoteHost,
        [string]$RemoteTarget
    )

    if (-not (Test-CommandAvailable "tar")) {
        throw "Neither rsync nor tar is available. Install rsync, or make sure Windows tar.exe is on PATH."
    }

    $archiveName = "planktonclass-sync-$([guid]::NewGuid().ToString('N')).tar.gz"
    $localArchive = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
    $remoteArchive = "/tmp/$archiveName"
    $escapedRemoteTarget = Escape-SingleQuotedRemoteValue $RemoteTarget

    try {
        Push-Location $SourceDir
        $tarArgs = @(
            "-czf", $localArchive,
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=PI10/predict_gpu_config.json",
            "--exclude=PI10/run_logs",
            "--exclude=.env",
            "--exclude=*/.env",
            "."
        )
        & tar @tarArgs
        if ($LASTEXITCODE -ne 0) {
            throw "tar archive creation failed."
        }
    }
    finally {
        Pop-Location
    }

    try {
        & ssh $RemoteHost "mkdir -p '$escapedRemoteTarget'"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create remote project directory."
        }

        & scp $localArchive "${RemoteHost}:$remoteArchive"
        if ($LASTEXITCODE -ne 0) {
            throw "scp upload failed."
        }

        & ssh $RemoteHost "tar -xzf '$remoteArchive' -C '$escapedRemoteTarget' && rm -f '$remoteArchive'"
        if ($LASTEXITCODE -ne 0) {
            throw "remote archive extraction failed."
        }
    }
    finally {
        if (Test-Path $localArchive) {
            Remove-Item -LiteralPath $localArchive -Force
        }
    }
}

function Invoke-ProjectSync {
    param(
        [string]$SourceDir,
        [string]$RemoteHost,
        [string]$RemoteTarget
    )

    if (Test-CommandAvailable "rsync") {
        Invoke-RsyncSync -SourceDir $SourceDir -RemoteHost $RemoteHost -RemoteTarget $RemoteTarget
    }
    else {
        Write-Host "rsync not found; using tar + scp fallback sync."
        Invoke-TarScpSync -SourceDir $SourceDir -RemoteHost $RemoteHost -RemoteTarget $RemoteTarget
    }
}

$LocalProject = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $PSScriptRoot "predict_gpu_config.json"
}

$PrivateConfig = Import-PrivatePredictConfig -Path $ConfigPath
$RemoteConfig = Get-JsonValue -Object $PrivateConfig -Path @("remote")
$ActiveProfileName = Get-JsonValue -Object $RemoteConfig -Path @("active_profile")
$ProfileConfig = $null

if ($ActiveProfileName) {
    $ProfileConfig = Get-JsonValue -Object $RemoteConfig -Path @("profiles", [string]$ActiveProfileName)
    if ($null -eq $ProfileConfig) {
        throw "Remote profile '$ActiveProfileName' was requested in $ConfigPath, but remote.profiles.$ActiveProfileName does not exist."
    }
}

if ($RemoteConfig) {
    if (-not $PSBoundParameters.ContainsKey("Remote")) {
        $configuredRemote = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "remote"
        $configuredSshUser = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "ssh_user"
        $configuredSshHost = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "ssh_host"

        if ($configuredRemote) {
            $Remote = [string]$configuredRemote
        }
        elseif ($configuredSshUser -and $configuredSshHost) {
            $Remote = "$configuredSshUser@$configuredSshHost"
        }
    }

    if (-not $PSBoundParameters.ContainsKey("RemoteProject")) {
        $configuredRemoteProject = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "remote_project"
        if ($configuredRemoteProject) {
            $RemoteProject = [string]$configuredRemoteProject
        }
    }

    if (-not $PSBoundParameters.ContainsKey("RemoteEnv")) {
        $configuredRemoteEnv = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "remote_env"
        if ($configuredRemoteEnv) {
            $RemoteEnv = [string]$configuredRemoteEnv
        }
    }

    if (-not $PSBoundParameters.ContainsKey("QarchiveRoot")) {
        $configuredQarchiveRoot = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "qarchive_root"
        if ($configuredQarchiveRoot) {
            $QarchiveRoot = [string]$configuredQarchiveRoot
        }
    }

    if (-not $PSBoundParameters.ContainsKey("QarchiveCheckDir")) {
        $configuredQarchiveCheckDir = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "qarchive_check_dir"
        if ($configuredQarchiveCheckDir) {
            $QarchiveCheckDir = [string]$configuredQarchiveCheckDir
        }
    }

    if (-not $PSBoundParameters.ContainsKey("QarchiveUser")) {
        $configuredQarchiveUser = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "qarchive_user"
        if ($configuredQarchiveUser) {
            $QarchiveUser = [string]$configuredQarchiveUser
        }
        elseif ($configuredSshUser) {
            $QarchiveUser = [string]$configuredSshUser
        }
    }

    if (-not $PSBoundParameters.ContainsKey("QarchiveDomain")) {
        $configuredQarchiveDomain = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "qarchive_domain"
        if ($configuredQarchiveDomain) {
            $QarchiveDomain = [string]$configuredQarchiveDomain
        }
    }

    if (-not $PSBoundParameters.ContainsKey("SshKeyComment")) {
        $configuredEmail = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "email"
        if ($configuredEmail) {
            $SshKeyComment = [string]$configuredEmail
        }
    }

    if (-not $PSBoundParameters.ContainsKey("Gpus")) {
        $configuredGpus = Get-RemoteConfigValue -RemoteConfig $RemoteConfig -ProfileConfig $ProfileConfig -Name "gpus"
        if ($configuredGpus) {
            $Gpus = @($configuredGpus | ForEach-Object { [int]$_ })
        }
    }
}

if (-not $SshKeyComment) {
    $SshKeyComment = "$env:USERNAME@$env:COMPUTERNAME"
}

$RemotePi10 = "$RemoteProject/PI10"

if ($SetupSshKey) {
    $sshDir = Join-Path $HOME ".ssh"
    $keyPath = Join-Path $sshDir "id_ed25519"
    $pubKeyPath = "$keyPath.pub"

    if (-not (Test-Path $pubKeyPath)) {
        New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
        & ssh-keygen -t ed25519 -f $keyPath -N "" -C $SshKeyComment
    }

    $publicKey = (Get-Content -LiteralPath $pubKeyPath -Raw).Trim()
    $escapedPublicKey = Escape-SingleQuotedRemoteValue $publicKey
    & ssh $Remote "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$escapedPublicKey' ~/.ssh/authorized_keys || printf '%s\n' '$escapedPublicKey' >> ~/.ssh/authorized_keys"

    if (Test-SshKeyLogin $Remote) {
        Write-Host "SSH key login is ready. Now run: .\sync_and_start_remote.ps1"
        exit 0
    }

    throw "SSH key setup ran, but passwordless login still failed. Check the password, server account, or ~/.ssh/authorized_keys permissions."
}

if (-not (Test-SshKeyLogin $Remote)) {
    throw @"
SSH key login is not ready yet.

Run this once from this folder:
  .\sync_and_start_remote.ps1 -SetupSshKey

Or from D:\USERS\wout.decrop\environments\PI10:
  .\planktonclass\PI10\sync_and_start_remote.ps1 -SetupSshKey
"@
}

if (-not $StartOnly) {
    Invoke-ProjectSync -SourceDir $LocalProject -RemoteHost $Remote -RemoteTarget $RemoteProject
}

if (-not $SyncOnly) {
    if (-not $SkipQarchiveRepair) {
        Ensure-RemoteQarchiveAccess `
            -RemoteHost $Remote `
            -Root $QarchiveRoot `
            -CheckDir $QarchiveCheckDir `
            -User $QarchiveUser `
            -Domain $QarchiveDomain
    }

    $gpuList = $Gpus -join " "
    $escapedRemotePi10 = Escape-SingleQuotedRemoteValue $RemotePi10
    $escapedRemoteEnv = Escape-SingleQuotedRemoteValue $RemoteEnv
    $escapedGpuList = Escape-SingleQuotedRemoteValue $gpuList
    $escapedQarchiveRoot = Escape-SingleQuotedRemoteValue $QarchiveRoot
    $escapedQarchiveCheckDir = Escape-SingleQuotedRemoteValue $QarchiveCheckDir
    $remoteCommand = "cd '$escapedRemotePi10' && PI10_REMOTE_ENV_DIR='$escapedRemoteEnv' PI10_GPUS='$escapedGpuList' PI10_QARCHIVE_ROOT='$escapedQarchiveRoot' PI10_QARCHIVE_CHECK_DIR='$escapedQarchiveCheckDir' bash remote_start_gpu_workers.sh"

    & ssh $Remote $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote worker start failed."
    }
}
