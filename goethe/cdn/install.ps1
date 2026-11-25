# Goethe on Windows: enable WSL2, install Ubuntu, then run the same install.sh as Linux/macOS.
# Usage:  irm https://goethe.sh/install | iex
# Or:    powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [--authkey ...]
# Safe to re-run. Never installs a Linux NVIDIA driver inside WSL.
# ASCII only: Windows PowerShell 5.1 mis-parses UTF-8 without a BOM.

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Script:Distro = 'Ubuntu-24.04'
$Script:DefaultPkgUrl = 'https://pkg.goethe.sh'
$Script:StateDir = Join-Path $env:LOCALAPPDATA 'Goethe'
$Script:StatePath = Join-Path $Script:StateDir 'install-state.json'
$Script:LogPath = Join-Path $Script:StateDir 'install.log'
$Script:PersistedPath = Join-Path $Script:StateDir 'install.ps1'
$Script:DriverMin = 580

# Capture source at script scope so irm|iex can persist a copy for RunOnce (functions see their own MyInvocation).
$Script:SelfSource = $null
if ($PSCommandPath -and (Test-Path -LiteralPath $PSCommandPath)) {
    $Script:SelfSource = [System.IO.File]::ReadAllText($PSCommandPath)
} elseif ($MyInvocation.MyCommand.ScriptBlock) {
    $Script:SelfSource = $MyInvocation.MyCommand.ScriptBlock.ToString()
}

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

function Clear-State {
    if (Test-Path -LiteralPath $Script:StatePath) {
        Remove-Item -LiteralPath $Script:StatePath -Force
    }
}

function Persist-Self {
    # Always keep a disk copy for reboot resume (irm|iex has no $PSCommandPath).
    Ensure-StateDir
    if (-not $Script:SelfSource) {
        return $Script:PersistedPath
    }
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Script:PersistedPath, $Script:SelfSource, $utf8)
    return $Script:PersistedPath
}

function ConvertTo-BashBase64([string]$Text) {
    # Git on Windows checkouts this file as CRLF; bash then rejects `set -euo pipefail\r`.
    $lf = $Text -replace "`r`n", "`n" -replace "`r", "`n"
    return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($lf))
}

function Invoke-Wsl {
    # Native wsl.exe stderr must not abort under $ErrorActionPreference Stop (PS 5.1 / 7).
    # Exit code is in $Script:LastWslExit. -Capture returns output; otherwise it streams to the host.
    param(
        [Parameter(Mandatory = $true)][string[]]$WslArgs,
        [switch]$IgnoreExit,
        [switch]$Capture
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Capture) {
            $out = & wsl.exe @WslArgs 2>&1
            $Script:LastWslExit = $LASTEXITCODE
        } else {
            & wsl.exe @WslArgs
            $Script:LastWslExit = $LASTEXITCODE
            $out = $null
        }
    } finally {
        $ErrorActionPreference = $prev
    }
    if (-not $IgnoreExit -and $Script:LastWslExit -ne 0) {
        $msg = if ($Capture) { (($out | Out-String) -replace [char]0, '').Trim() } else { '' }
        if (-not $msg) { $msg = "wsl exit $($Script:LastWslExit)" }
        Die $msg
    }
    return $out
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-Admin {
    if (Test-IsAdmin) { return }
    Die 'this installer must run as Administrator (right-click PowerShell -> Run as administrator, then run it again)'
}

function Assert-SupportedWindows {
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64' -and $env:PROCESSOR_ARCHITEW6432 -ne 'AMD64') {
        Die "unsupported CPU architecture '$($env:PROCESSOR_ARCHITECTURE)'; Goethe on Windows needs x64 with WSL2"
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $cap = [version]$os.Version
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
            '^--authkey=' { $forward.Add($a); $i++; continue }
            '^--platform-url$' {
                if ($i + 1 -ge $ArgList.Count) { Die '--platform-url needs a value' }
                $forward.Add('--platform-url'); $forward.Add($ArgList[$i + 1]); $i += 2; continue
            }
            '^--platform-url=' { $forward.Add($a); $i++; continue }
            '^--version$' {
                if ($i + 1 -ge $ArgList.Count) { Die '--version needs a value' }
                $forward.Add('--version'); $forward.Add($ArgList[$i + 1]); $i += 2; continue
            }
            '^--version=' { $forward.Add($a); $i++; continue }
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
        [void](Invoke-Wsl -WslArgs @('--set-default-version', '2') -IgnoreExit)
    } catch { }
    return [bool]$needReboot
}

function Register-ResumeAfterReboot {
    param([string[]]$LinuxArgs)
    $self = Persist-Self
    if (-not (Test-Path -LiteralPath $self)) {
        Die "could not save installer to $self for reboot resume; re-run after reboot with: irm https://goethe.sh/install | iex"
    }
    # Quote each arg for RunOnce (auth keys must survive the reboot command line).
    $parts = New-Object System.Collections.Generic.List[string]
    $parts.Add('--resume')
    foreach ($a in $LinuxArgs) {
        if ($a -match "[\s`"']") {
            $parts.Add("'" + ($a -replace "'", "''") + "'")
        } else {
            $parts.Add($a)
        }
    }
    $argLine = $parts -join ' '
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$self`" $argLine"
    Save-State @{
        stage = 'post-reboot'
        startedAt = (Get-Date).ToString('o')
    }
    $runOnce = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
    if (-not (Test-Path -LiteralPath $runOnce)) {
        New-Item -Path $runOnce -Force | Out-Null
    }
    New-ItemProperty -Path $runOnce -Name 'GoetheInstall' -Value $cmd -PropertyType String -Force | Out-Null
    Write-Log 'registered RunOnce (args omitted from log)'
    Write-Goethe 'a reboot is required to finish enabling WSL2'
    Write-Goethe 'after you sign back in, this installer continues on its own'
    Write-Goethe "log: $Script:LogPath"
    Restart-Computer -Force
}

function Assert-WslReady {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Die 'wsl.exe is missing after enabling WSL; install "Windows Subsystem for Linux" from Optional features, reboot, and run this installer again'
    }
    [void](Invoke-Wsl -WslArgs @('--status') -IgnoreExit)
    if ($Script:LastWslExit -ne 0) {
        Die 'WSL is not ready yet; reboot once, then run this installer again'
    }
    [void](Invoke-Wsl -WslArgs @('--set-default-version', '2') -IgnoreExit)
    if ($Script:LastWslExit -ne 0) {
        Die 'could not set WSL default version to 2'
    }
}

function Get-WslDistroNames {
    $raw = Invoke-Wsl -WslArgs @('-l', '-q') -IgnoreExit -Capture
    if ($Script:LastWslExit -ne 0) { return @() }
    $text = ($raw | Out-String) -replace [char]0, ''
    return @($text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Test-DistroInstalled([string]$Name) {
    return (Get-WslDistroNames) -contains $Name
}

function Install-UbuntuDistro {
    if (Test-DistroInstalled $Script:Distro) {
        Write-Goethe "WSL distro $Script:Distro is already installed"
        [void](Invoke-Wsl -WslArgs @('--set-default', $Script:Distro) -IgnoreExit)
        return
    }
    Write-Goethe "installing $Script:Distro (WSL2)"
    Write-Log "wsl --install -d $Script:Distro"
    [void](Invoke-Wsl -WslArgs @('--install', '-d', $Script:Distro, '--no-launch') -IgnoreExit)
    if ($Script:LastWslExit -ne 0) {
        Die "could not install $Script:Distro (wsl exit $($Script:LastWslExit)); fix WSL, then run this installer again"
    }
    [void](Invoke-Wsl -WslArgs @('--set-default', $Script:Distro) -IgnoreExit)
    $deadline = (Get-Date).AddMinutes(5)
    do {
        [void](Invoke-Wsl -WslArgs @('-d', $Script:Distro, '--', 'bash', '-lc', 'echo ok') -IgnoreExit)
        if ($Script:LastWslExit -eq 0) { return }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    Die "$Script:Distro did not become ready within 5 minutes; open 'Ubuntu 24.04' once from the Start menu to finish first-time setup, then run this installer again"
}

function Assert-GlibcDistro {
    $raw = Invoke-Wsl -WslArgs @('-d', $Script:Distro, '--', 'getconf', 'GNU_LIBC_VERSION') -IgnoreExit -Capture
    $text = (($raw | Out-String) -replace [char]0, '').Trim()
    $ver = $null
    if ($text -match '(\d+(?:\.\d+)+)') { $ver = $Matches[1] }
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
touch "$conf"
if awk '/^\[/{b=($0=="[boot]")} b && /^[[:space:]]*systemd[[:space:]]*=[[:space:]]*true[[:space:]]*$/{f=1} END{exit !f}' "$conf"; then
  exit 0
fi
tmp="$(mktemp)"
awk '/^\[/{b=($0=="[boot]")} b && /^[[:space:]]*systemd[[:space:]]*=/{next} {print} $0=="[boot]" && !d{print "systemd=true"; d=1} END{if(!d) printf "\n[boot]\nsystemd=true\n"}' "$conf" >"$tmp"
cp "$tmp" "$conf"
rm -f "$tmp"
'@
    $b64 = ConvertTo-BashBase64 $script
    [void](Invoke-Wsl -WslArgs @('-d', $Script:Distro, '-u', 'root', '--', 'bash', '-lc', "echo $b64 | base64 -d | bash") -IgnoreExit)
    if ($Script:LastWslExit -ne 0) {
        Die "could not enable systemd in /etc/wsl.conf as root; run: wsl -d $Script:Distro -u root -- bash -lc `"printf '\\n[boot]\\nsystemd=true\\n' >> /etc/wsl.conf`" then wsl --terminate $Script:Distro"
    }
    Write-Goethe "restarting $Script:Distro so systemd starts"
    # Terminate only this distro; do not wsl --shutdown (that kills every distro).
    [void](Invoke-Wsl -WslArgs @('--terminate', $Script:Distro) -IgnoreExit)
    Start-Sleep -Seconds 3
    $deadline = (Get-Date).AddMinutes(2)
    do {
        [void](Invoke-Wsl -WslArgs @('-d', $Script:Distro, '--', 'bash', '-lc', 'echo ok') -IgnoreExit)
        if ($Script:LastWslExit -eq 0) { break }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    $commRaw = Invoke-Wsl -WslArgs @('-d', $Script:Distro, '--', 'bash', '-lc', 'tr -d "\0" </proc/1/comm') -IgnoreExit -Capture
    $comm = (($commRaw | Out-String) -replace [char]0, '').Trim()
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
        Write-Goethe 'no NVIDIA GPU detected on Windows; Goethe installs anyway (CPU). For GPU inference, install a Windows NVIDIA driver 580 or newer, then: wsl --terminate Ubuntu-24.04'
        return
    }
    $driver = $null
    if ($gpu.DriverVersion) { $driver = [string]$gpu.DriverVersion }
    Write-Goethe "NVIDIA GPU on Windows: $($gpu.Name) (driver $driver)"
    Write-Goethe "for GPU inference use Windows NVIDIA driver $Script:DriverMin or newer, then: wsl --terminate $Script:Distro"
    Write-Goethe 'never install a Linux NVIDIA driver inside WSL'
}

function Test-PkgUrl([string]$Url) {
    # The same rule as install.sh: https, or plain http only for this machine or with GOETHE_ALLOW_HTTP=1, since install.sh itself comes from this URL.
    if ($Url -match '^https://') { return }
    if ($Url -match '^http://(localhost|127\.\d+\.\d+\.\d+|\[::1\])([:/]|$)' -or $env:GOETHE_ALLOW_HTTP -eq '1') { return }
    Die "GOETHE_PKG_URL must be an https:// URL, got $Url"
}

function Invoke-LinuxInstaller {
    # Same install.sh as Linux and macOS; Windows only bootstraps WSL. Settings and secrets reach Ubuntu through WSLENV, never on a command line.
    param([string[]]$LinuxArgs)
    $pkg = $env:GOETHE_PKG_URL
    if (-not $pkg) { $pkg = $Script:DefaultPkgUrl }
    $pkg = $pkg.TrimEnd('/')
    Test-PkgUrl $pkg
    $env:GOETHE_PKG_URL = $pkg
    Write-Goethe "running install.sh inside $Script:Distro (same installer as Linux/macOS)"
    Write-Log "GOETHE_PKG_URL=$pkg"

    $forward = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $LinuxArgs.Count; $i++) {
        $a = $LinuxArgs[$i]
        if ($a -eq '--authkey') { $env:GOETHE_AUTHKEY = $LinuxArgs[$i + 1]; $i++ }
        elseif ($a.StartsWith('--authkey=')) { $env:GOETHE_AUTHKEY = $a.Substring(10) }
        else { $forward.Add($a) }
    }
    if (-not $env:GOETHE_AUTHKEY -and -not $env:GOETHE_LOGIN) { $env:GOETHE_LOGIN = '0' }
    $names = @('GOETHE_PKG_URL', 'GOETHE_AUTHKEY', 'GOETHE_PKG_TOKEN', 'GOETHE_PLATFORM_URL', 'GOETHE_HOME', 'GOETHE_LOGIN', 'GOETHE_VERSION', 'GOETHE_ALLOW_HTTP')
    $pass = @($names | Where-Object { [Environment]::GetEnvironmentVariable($_) } | ForEach-Object { "$_/u" })
    if ($env:WSLENV) { $pass = @($env:WSLENV) + $pass }
    $env:WSLENV = $pass -join ':'

    $argStr = ($forward | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join ' '

    $bash = @"
set -euo pipefail
curl -fsSL "`${GOETHE_PKG_URL}/install.sh" | bash -s -- $argStr
command -v goethe >/dev/null
goethe --version
"@
    $b64 = ConvertTo-BashBase64 $bash
    [void](Invoke-Wsl -WslArgs @('-d', $Script:Distro, '--', 'bash', '-lc', "echo $b64 | base64 -d | bash") -IgnoreExit)
    if ($Script:LastWslExit -ne 0) {
        Die "install.sh failed inside WSL (exit $($Script:LastWslExit)); see messages above, then run this installer again"
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
echo "Next:  goethe up && goethe api expose local"
echo "From Windows:  curl.exe -k https://127.0.0.1:8443/  (HTTPS; allowlist via expose local)"
echo "GPU driver stays on Windows (580+). Never install a Linux NVIDIA driver in WSL."
echo
exec goethe
'@
    $b64 = ConvertTo-BashBase64 $inner
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
    [void](Persist-Self)
    Write-Log "start (arg count=$($ArgList.Count))"
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
        [void](Invoke-Wsl -WslArgs @('--set-default-version', '2') -IgnoreExit)
    }

    Assert-WslReady
    Install-UbuntuDistro
    Assert-GlibcDistro
    Enable-SystemdInWsl
    Show-NvidiaDriverHint
    Invoke-LinuxInstaller -LinuxArgs $linuxArgs
    Clear-State
    Write-Goethe 'install finished (Linux install.sh completed inside WSL)'
    Write-Goethe 'next:  wsl -d Ubuntu-24.04 -- goethe up'
    Write-Goethe 'then:  wsl -d Ubuntu-24.04 -- goethe api expose local'
    Write-Goethe 'from Windows use HTTPS:  curl.exe -k https://127.0.0.1:8443/'
    try {
        Open-GoetheTerminal
    } catch {
        Write-Goethe "could not open a terminal automatically; run:  wsl -d $Script:Distro"
        Write-Log "terminal open failed: $_"
    }
}

$argv = @()
if ($args) { $argv = @($args) }
Main -ArgList $argv
