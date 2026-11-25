# Registers VisionDine AI to run automatically on Windows startup (24/7 Restaurant Server)
$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path "$PSScriptRoot\..").Path
$RunnerBat = Join-Path $ProjectDir "start_24_7.bat"
$TaskName = "VisionDineAI_24_7_Server"

if (-not (Test-Path $RunnerBat)) {
    Write-Error "Could not find $RunnerBat"
    exit 1
}

Write-Host "Setting up auto-start task '$TaskName' for: $ProjectDir" -ForegroundColor Cyan

# Run hidden/minimized on logon (no UAC needed for current user)
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c start /min `"`" `"$RunnerBat`"" -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force | Out-Null

Write-Host "SUCCESS! VisionDine AI will now start automatically whenever Windows starts." -ForegroundColor Green
Write-Host "Tip: Also disable Sleep mode in Windows Power Settings so the PC stays awake 24/7." -ForegroundColor Yellow
