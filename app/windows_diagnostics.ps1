# windows_diagnostics.ps1
# Run this on your Windows machine, inside the folder containing your WASH-IT
# project files (student_best.pt, server.py, test_continuous_sequence_local.py,
# hand_landmarker.task, norm_stats.npz, washit_v7_live.html, etc).
#
# It does two things:
#   1. Prints diagnostic info (Python/pip versions, camera devices, file sizes)
#      that's useful to paste back into chat if something doesn't match what
#      the Pi setup script expects.
#   2. Packages the files the Pi actually needs into a zip, ready to transfer
#      with scp.
#
# USAGE:
#   powershell -ExecutionPolicy Bypass -File windows_diagnostics.ps1

Write-Host "=== Python environment ===" -ForegroundColor Cyan
python --version
pip show torch mediapipe opencv-python numpy scipy websockets 2>$null |
    Select-String -Pattern "Name:|Version:"

Write-Host "`n=== Connected cameras (DirectShow devices) ===" -ForegroundColor Cyan
Get-PnpDevice -Class Camera | Select-Object FriendlyName, Status

Write-Host "`n=== Project files present ===" -ForegroundColor Cyan
$required = @(
    "server.py",
    "test_continuous_sequence_local.py",
    "model_tcn_attention.py",
    "live_webcam.py",
    "live_guided.py",
    "live_sequence_stream.py",
    "student_best.pt",
    "norm_stats.npz",
    "hand_landmarker.task",
    "washit_v7_live.html",
    "washit_live_patch.js"
)
foreach ($f in $required) {
    if (Test-Path $f) {
        $size = (Get-Item $f).Length / 1MB
        Write-Host ("  [OK]  {0,-35} {1,8:N2} MB" -f $f, $size) -ForegroundColor Green
    } else {
        Write-Host ("  [MISSING]  {0}" -f $f) -ForegroundColor Yellow
    }
}

Write-Host "`n=== norm_stats.npz check ===" -ForegroundColor Cyan
if (-not (Test-Path "norm_stats.npz")) {
    Write-Host "  ! norm_stats.npz was not in the files you sent me earlier either." -ForegroundColor Yellow
    Write-Host "    load_model() in test_continuous_sequence_local.py requires it -" -ForegroundColor Yellow
    Write-Host "    make sure it's copied to the Pi alongside student_best.pt." -ForegroundColor Yellow
}

Write-Host "`n=== Packaging for transfer to the Pi ===" -ForegroundColor Cyan
$zipName = "washit_pi_bundle.zip"
if (Test-Path $zipName) { Remove-Item $zipName }
$present = $required | Where-Object { Test-Path $_ }
Compress-Archive -Path $present -DestinationPath $zipName
Write-Host "  Created $zipName ($((Get-Item $zipName).Length / 1MB | ForEach-Object {$_.ToString('N1')}) MB)" -ForegroundColor Green

Write-Host "`n=== Transfer to the Pi ===" -ForegroundColor Cyan
Write-Host "  Find the Pi's IP address on the Pi itself with: hostname -I"
Write-Host "  Then from this Windows machine (needs OpenSSH client - Settings >"
Write-Host "  Apps > Optional Features > OpenSSH Client if 'scp' isn't found):"
Write-Host ""
Write-Host "    scp $zipName pi@<PI_IP_ADDRESS>:~/washit/" -ForegroundColor White
Write-Host ""
Write-Host "  Then on the Pi:"
Write-Host "    cd ~/washit && unzip washit_pi_bundle.zip" -ForegroundColor White
