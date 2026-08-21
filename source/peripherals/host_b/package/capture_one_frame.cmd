@echo off
setlocal
set "INSTALL_ROOT=%~dp0"
if not exist "%INSTALL_ROOT%versions\peripheral_base_v10\host_b_tool.exe" (
    set "INSTALL_ROOT=%LOCALAPPDATA%\ArtificialLifeHostB\"
)
set "APP=%INSTALL_ROOT%versions\peripheral_base_v10\host_b_tool.exe"
set "CONFIG=%INSTALL_ROOT%host_b.toml"

if not exist "%APP%" (
    echo Host B program is not installed correctly.
    pause
    exit /b 1
)

echo Switch back to Minecraft now. One-frame capture starts in 5 seconds.
timeout /t 5 /nobreak >nul
"%APP%" capture-one "%CONFIG%"
if not defined HOST_B_NO_PAUSE pause
