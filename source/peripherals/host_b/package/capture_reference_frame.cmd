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

if "%~1"=="" (
    echo Usage: capture_reference_frame.cmd reference_name
    pause
    exit /b 2
)

set "OUTPUT=%INSTALL_ROOT%visual_references"
echo Switch back to Minecraft now. Reference capture starts in 5 seconds.
timeout /t 5 /nobreak >nul
"%APP%" capture-reference "%CONFIG%" "%OUTPUT%" "%~1"
if not defined HOST_B_NO_PAUSE pause
