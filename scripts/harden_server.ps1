# Hardens Windows file permissions on config/ secrets and configures Private-Network-Only Firewall for port 8085
$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path "$PSScriptRoot\..").Path
$ConfigDir = Join-Path $ProjectDir "config"
$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"

Write-Host "=== VisionDine AI Server Hardening ===" -ForegroundColor Cyan

# 1. Restrict NTFS ACLs on sensitive config files to current owner & SYSTEM only
$SecretFiles = @(
    "settings.json",
    "firebase_credentials.json",
    "telegram_config.json",
    "auth_config.json"
)

foreach ($file in $SecretFiles) {
    $fullPath = Join-Path $ConfigDir $file
    if (Test-Path $fullPath) {
        icacls $fullPath /inheritance:r | Out-Null
        icacls $fullPath /grant:r "${CurrentUser}:(F)" "SYSTEM:(F)" | Out-Null
        Write-Host "[ACL] Locked permissions on config\$file to $CurrentUser & SYSTEM" -ForegroundColor Green
    }
}

# 2. Firewall check (Restrict port 8085 to Private network profile only if running as Administrator)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($isAdmin) {
    $RuleName = "VisionDineAI_Port8085_PrivateOnly"
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName $RuleName `
        -Direction Inbound `
        -LocalPort 8085 `
        -Protocol TCP `
        -Action Allow `
        -Profile Private `
        -RemoteAddress LocalSubnet | Out-Null
    Write-Host "[Firewall] Allowed port 8085 strictly on Private LocalSubnet." -ForegroundColor Green
} else {
    Write-Host "[Firewall] Skipping Windows Firewall rule (not running as Administrator)." -ForegroundColor Yellow
    Write-Host "           Built-in server LAN-only middleware is still active and blocking public IPs!" -ForegroundColor Yellow
}

Write-Host "=== Hardening Complete ===" -ForegroundColor Cyan
