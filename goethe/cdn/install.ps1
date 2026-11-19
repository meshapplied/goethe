# Goethe on Windows: enable WSL2, install Ubuntu, run the Linux installer, open a terminal.
# Usage:  irm https://goethe.sh/install.ps1 | iex
# Or:    powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [--authkey ...]
# Safe to re-run. Never installs a Linux NVIDIA driver inside WSL.

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Script:Distro = 'Ubuntu-24.04'
$Script:DefaultPkgUrl = 'https://pkg.goethe.sh'
$Script:StateDir = Join-Path $env:LOCALAPPDATA 'Goethe'
$Script:StatePath = Join-Path $Script:StateDir 'install-state.json'
$Script:LogPath = Join-Path $Script:StateDir 'install.log'
$Script:DriverMin = 580

function Write-Goethe([string]$Message) {
    Write-Host "goethe install: $Message"
}

function Write-GoetheErr([string]$Message) {
    Write-Host "goethe install: $Message" -ForegroundColor Red
}

function Die([string]$Message) {
    Write-GoetheErr $Message
    throw $Message
}

function Ensure-StateDir {
    if (-not (Test-Path -LiteralPath $Script:StateDir)) {
        New-Item -ItemType Directory -Path $Script:StateDir -Force | Out-Null
    }
}

function Write-Log([string]$Message) {
    Ensure-StateDir
    $line = '{0} {1}' -f (Get-Date -Format 'o'), $Message
    Add-Content -LiteralPath $Script:LogPath -Value $line -Encoding UTF8
}

function Save-State([hashtable]$State) {
    Ensure-StateDir
    ($State | ConvertTo-Json -Compress) | Set-Content -LiteralPath $Script:StatePath -Encoding UTF8
}

function Read-State {
    if (-not (Test-Path -LiteralPath $Script:StatePath)) {
        return $null
    }
    return Get-Content -LiteralPath $Script:StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Clear-State {
    if (Test-Path -LiteralPath $Script:StatePath) {
        Remove-Item -LiteralPath $Script:StatePath -Force
    }
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-Admin {
    if (Test-IsAdmin) { return }
    Die 'this installer must run as Administrator (right-click PowerShell → Run as administrator, then run it again)'
}

function Assert-SupportedWindows {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        Die "unsupported CPU architecture '$($env:PROCESSOR_ARCHITECTURE)'; Goethe on Windows needs x64 with WSL2"
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $cap = [version]$os.Version
    # Windows 10 1903+ / Windows 11: WSL2 requires build 18362+.
    if ($cap.Build -lt 18362) {
        Die "Windows build $($cap.Build) is too old for WSL2; use Windows 10 version 1903 or newer, or Windows 11"
    }
}

function Get-PassedLinuxArgs {
    # Forward only known flags to install.sh; everything else is refused.
    param([string[]]$ArgList)
    $forward = New-Object System.Collections.Generic.List[string]
    $i = 0
    while ($i -lt $ArgList.Count) {
        $a = $ArgList[$i]
        switch -Regex ($a) {
            '^--authkey$' {
                if ($i + 1 -ge $ArgList.Count) { Die '--authkey needs a value' }
                $forward.Add('--authkey'); $forward.Add($ArgList[$i + 1]); $i += 2; continue
            }
            '^--authkey=' {
                $forward.Add($a); $i++; continue
            }
            '^--platform-url$' {
                if ($i + 1 -ge $ArgList.Count) { Die '--platform-url needs a value' }
                $forward.Add('--platform-url'); $forward.Add($ArgList[$i + 1]); $i += 2; continue
            }
            '^--platform-url=' {
                $forward.Add($a); $i++; continue
            }
            '^--version$' {
                if ($i + 1 -ge $ArgList.Count) { Die '--version needs a value' }
                $forward.Add('--version'); $forward.Add($ArgList[$i + 1]); $i += 2; continue
            }
            '^--version=' {
                $forward.Add($a); $i++; continue
            }
            '^--resume$' { $i++; continue }
            default { Die "unknown argument: $a (supported: --authkey, --platform-url, --version)" }
        }
    }
    return ,$forward.ToArray()
}

function Test-WslFeatureEnabled {
    $wsl = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -ErrorAction SilentlyContinue
    $vm = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -ErrorAction SilentlyContinue
    return ($wsl -and $wsl.State -eq 'Enabled' -and $vm -and $vm.State -eq 'Enabled')
}

function Enable-WslFeatures {
    Write-Goethe 'enabling WSL and the Virtual Machine Platform (may need a reboot)'
    Write-Log 'enabling Windows features'
    $r1 = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
    $r2 = Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All -NoRestart
    $needReboot = ($r1.RestartNeeded -or $r2.RestartNeeded -or -not (Test-WslFeatureEnabled))
    try {
        & wsl --set-default-version 2 2>$null | Out-Null
    } catch {
        # wsl.exe may be missing until reboot; resume handles it.
    }
    return [bool]$needReboot
}

function Register-ResumeAfterReboot {
    param([string[]]$LinuxArgs)
    Ensure-StateDir
    $self = if ($PSCommandPath) { $PSCommandPath } else { Join-Path $Script:StateDir 'install.ps1' }
    if ($PSCommandPath -and (Test-Path -LiteralPath $PSCommandPath)) {
        Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $Script:StateDir 'install.ps1') -Force
        $self = Join-Path $Script:StateDir 'install.ps1'
    }
    $argLine = (@('--resume') + $LinuxArgs) -join ' '
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$self`" $argLine"
    Save-State @{
        stage = 'post-reboot'
        linuxArgs = $LinuxArgs
        startedAt = (Get-Date).ToString('o')
    }
    $runOnce = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
    if (-not (Test-Path -LiteralPath $runOnce)) {
        New-Item -Path $runOnce -Force | Out-Null
    }
    New-ItemProperty -Path $runOnce -Name 'GoetheInstall' -Value $cmd -PropertyType String -Force | Out-Null
    Write-Log "registered RunOnce: $cmd"
    Write-Goethe 'a reboot is required to finish enabling WSL2'
    Write-Goethe 'after you sign back in, this installer continues on its own'
    Write-Goethe "log: $Script:LogPath"
    Restart-Computer -Force
}

function Assert-WslReady {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Die 'wsl.exe is missing after enabling WSL; install "Windows Subsystem for Linux" from Optional features, reboot, and run this installer again'
    }
    & wsl --status 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Die 'WSL is not ready yet; reboot once, then run this installer again'
    }
    try {
        & wsl --set-default-version 2 | Out-Null
    } catch {
        Die 'could not set WSL default version to 2'
    }
}

function Get-WslDistroNames {
    # wsl.exe often prints UTF-16; strip NULs so names compare as plain text.
    $raw = & wsl -l -q 2>$null
    if ($LASTEXITCODE -ne 0) { return @() }
    $text = ($raw | Out-String) -replace [char]0, ''
    return @($text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Test-DistroInstalled([string]$Name) {
    return (Get-WslDistroNames) -contains $Name
}

function Install-UbuntuDistro {
    if (Test-DistroInstalled $Script:Distro) {
        Write-Goethe "WSL distro $Script:Distro is already installed"
        & wsl --set-default $Script:Distro | Out-Null
        return
    }
    Write-Goethe "installing $Script:Distro (WSL2)"
    Write-Log "wsl --install -d $Script:Distro"
    # --no-launch avoids an interactive first-boot prompt in this process.
    & wsl --install -d $Script:Distro --no-launch
    if ($LASTEXITCODE -ne 0) {
        Die "could not install $Script:Distro (wsl exit $LASTEXITCODE); fix WSL, then run this installer again"
    }
    & wsl --set-default $Script:Distro | Out-Null
    $deadline = (Get-Date).AddMinutes(5)
    do {
        & wsl -d $Script:Distro -- bash -lc 'echo ok' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    Die "$Script:Distro did not become ready within 5 minutes; open 'Ubuntu 24.04' once from the Start menu to finish first-time setup, then run this installer again"
}

function Assert-GlibcDistro {
    $out = & wsl -d $Script:Distro -- bash -lc 'getconf GNU_LIBC_VERSION 2>/dev/null | awk "{print \$2}"' 2>$null
    $ver = ($out | Out-String).Trim()
    if (-not $ver) {
        Die "$Script:Distro does not look like a glibc system; Goethe needs Ubuntu or another glibc distro (not Alpine/musl)"
    }
    Write-Log "glibc $ver"
}

function Enable-SystemdInWsl {
    Write-Goethe 'enabling systemd in WSL (so goethe up can install a service)'
    $script = @'
set -euo pipefail
conf=/etc/wsl.conf
if [[ -f "$conf" ]] && grep -q '^\[boot\]' "$conf" && grep -Eq '^[[:space:]]*systemd[[:space:]]*=[[:space:]]*true' "$conf"; then
  exit 0
fi
tmp="$(mktemp)"
if [[ -f "$conf" ]]; then
  awk 'BEGIN{skip=0} /^\[/{skip=($0=="[boot]")} !skip{print}' "$conf" >"$tmp"
else
  : >"$tmp"
fi
printf '\n[boot]\nsystemd=true\n' >>"$tmp"
cp "$tmp" "$conf"
rm -f "$tmp"
'@
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
    & wsl -d $Script:Distro -u root -- bash -lc "echo $b64 | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) {
        Die "could not enable systemd in /etc/wsl.conf as root; run: wsl -d $Script:Distro -u root -- bash -lc `"printf '\\n[boot]\\nsystemd=true\\n' >> /etc/wsl.conf`" then wsl --shutdown"
    }
    Write-Goethe 'restarting WSL so systemd starts'
    & wsl --shutdown
    Start-Sleep -Seconds 3
    $deadline = (Get-Date).AddMinutes(2)
    do {
        & wsl -d $Script:Distro -- bash -lc 'echo ok' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    $comm = (& wsl -d $Script:Distro -- bash -lc 'tr -d "\0" </proc/1/comm' 2>$null | Out-String).Trim()
    if ($comm -ne 'systemd') {
        Write-Goethe "note: PID 1 is '$comm' (wanted systemd); goethe up will still work, using crontab if needed"
        Write-Log "pid1=$comm"
    }
}

function Show-NvidiaDriverHint {
    $gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'NVIDIA' } |
        Select-Object -First 1
    if (-not $gpu) {
        Write-Goethe 'no NVIDIA GPU detected on Windows; Goethe installs anyway (CPU). For GPU inference, install a Windows NVIDIA driver 580 or newer, then run: wsl --shutdown'
        return
    }
    $driver = $null
    if ($gpu.DriverVersion) {
        # Windows driver versions look like 32.0.15.8092; NVIDIA branch is not always parseable here.
        $driver = [string]$gpu.DriverVersion
    }
    Write-Goethe "NVIDIA GPU on Windows: $($gpu.Name) (driver $driver)"
    Write-Goethe "for GPU inference use Windows NVIDIA driver $Script:DriverMin or newer, then: wsl --shutdown"
    Write-Goethe 'never install a Linux NVIDIA driver inside WSL'
}

function Invoke-LinuxInstaller {
    param([string[]]$LinuxArgs)
    $pkg = $env:GOETHE_PKG_URL
    if (-not $pkg) { $pkg = $Script:DefaultPkgUrl }
    $pkg = $pkg.TrimEnd('/')
    Write-Goethe "running the Linux installer inside $Script:Distro"
    Write-Log "GOETHE_PKG_URL=$pkg args=$($LinuxArgs -join ' ')"

    $export = New-Object System.Collections.Generic.List[string]
    $export.Add("export GOETHE_PKG_URL='$pkg'")
    # Non-interactive: do not block on browser login unless an auth key was provided.
    $hasAuth = $false
    foreach ($a in $LinuxArgs) {
        if ($a -eq '--authkey' -or $a.StartsWith('--authkey=')) { $hasAuth = $true }
    }
    if ($env:GOETHE_AUTHKEY) { $hasAuth = $true; $export.Add("export GOETHE_AUTHKEY='$($env:GOETHE_AUTHKEY)'") }
    if ($env:GOETHE_PLATFORM_URL) { $export.Add("export GOETHE_PLATFORM_URL='$($env:GOETHE_PLATFORM_URL)'") }
    if ($env:GOETHE_HOME) { $export.Add("export GOETHE_HOME='$($env:GOETHE_HOME)'") }
    if (-not $hasAuth -and -not $env:GOETHE_LOGIN) {
        $export.Add('export GOETHE_LOGIN=0')
    } elseif ($env:GOETHE_LOGIN) {
        $export.Add("export GOETHE_LOGIN='$($env:GOETHE_LOGIN)'")
    }

    $argStr = ($LinuxArgs | ForEach-Object {
        if ($_ -match "[\s`"]") { "'" + ($_ -replace "'", "'\''") + "'" } else { $_ }
    }) -join ' '

    $bash = @"
set -euo pipefail
$($export -join "`n")
curl -fsSL "$pkg/install.sh" | bash -s -- $argStr
command -v goethe >/dev/null
goethe --version
"@
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bash))
    & wsl -d $Script:Distro -- bash -lc "echo $b64 | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) {
        Die "Linux installer failed inside WSL (exit $LASTEXITCODE); see messages above, then run this installer again"
    }
}

function Open-GoetheTerminal {
    Write-Goethe 'opening a terminal in WSL with Goethe'
    $inner = @'
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
if ! command -v goethe >/dev/null 2>&1; then
  echo "goethe install: goethe is not on PATH in this shell; open a new Ubuntu terminal or run: hash -r" >&2
  exec bash -l
fi
echo "goethe $(goethe --version 2>/dev/null | head -n1)"
echo "From Windows, after: goethe up   use http://127.0.0.1:<port> (localhost reaches WSL)."
echo "GPU driver stays on Windows (580+). Never install a Linux NVIDIA driver in WSL."
echo
exec goethe
'@
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($inner))
    $wslArgs = @('-d', $Script:Distro, '--', 'bash', '-lc', "echo $b64 | base64 -d | bash")

    $wt = Get-Command wt.exe -ErrorAction SilentlyContinue
    if ($wt) {
        Start-Process -FilePath $wt.Source -ArgumentList (@('-w', '0', 'nt', 'wsl.exe') + $wslArgs) | Out-Null
        return
    }
    Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs | Out-Null
}

function Main {
    param([string[]]$ArgList)

    Ensure-StateDir
    Write-Log "start args=$($ArgList -join ' ')"
    Assert-SupportedWindows

    $resume = $false
    $filtered = New-Object System.Collections.Generic.List[string]
    foreach ($a in $ArgList) {
        if ($a -eq '--resume') { $resume = $true; continue }
        $filtered.Add($a)
    }
    $linuxArgs = Get-PassedLinuxArgs -ArgList $filtered.ToArray()

    if ($resume) {
        Write-Goethe 'resuming after reboot'
        Write-Log 'resume'
    }

    Assert-Admin

    if (-not (Test-WslFeatureEnabled)) {
        $needReboot = Enable-WslFeatures
        if ($needReboot -or -not (Test-WslFeatureEnabled)) {
            Register-ResumeAfterReboot -LinuxArgs $linuxArgs
            return
        }
    } else {
        try { & wsl --set-default-version 2 | Out-Null } catch { }
    }

    Assert-WslReady
    Install-UbuntuDistro
    Assert-GlibcDistro
    Enable-SystemdInWsl
    Show-NvidiaDriverHint
    Invoke-LinuxInstaller -LinuxArgs $linuxArgs
    Clear-State
    Write-Goethe 'install finished'
    Write-Goethe 'tip: from Windows, localhost ports reach this WSL distro after goethe up'
    try {
        Open-GoetheTerminal
    } catch {
        Write-Goethe "could not open a terminal automatically; run:  wsl -d $Script:Distro"
        Write-Log "terminal open failed: $_"
    }
}

# When piped via `irm … | iex`, there is no $PSCommandPath; persist a local copy for RunOnce.
if (-not $PSCommandPath) {
    Ensure-StateDir
    $dest = Join-Path $Script:StateDir 'install.ps1'
    try {
        $src = $MyInvocation.MyCommand.Path
        if ($src -and (Test-Path -LiteralPath $src)) {
            Copy-Item -LiteralPath $src -Destination $dest -Force
        }
    } catch { }
}

$argv = @()
if ($args) { $argv = @($args) }
Main -ArgList $argv
