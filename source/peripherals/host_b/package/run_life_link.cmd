@echo off
setlocal
set "INSTALL_ROOT=%~dp0"
if not exist "%INSTALL_ROOT%versions\peripheral_base_v10\host_b_tool.exe" (
    set "INSTALL_ROOT=%LOCALAPPDATA%\ArtificialLifeHostB\"
)
set "APP=%INSTALL_ROOT%versions\peripheral_base_v10\host_b_tool.exe"
set "CONFIG=%INSTALL_ROOT%host_b.toml"

if not exist "%APP%" (
    echo Host B version 10 is not installed correctly.
    pause
    exit /b 1
)

echo Keep Minecraft in the foreground. Press F12 at any time for emergency stop.
echo Full peripheral connection starts in 5 seconds.
timeout /t 5 /nobreak >nul
"%APP%" run-life-link "%CONFIG%"
if not defined HOST_B_NO_PAUSE pause
