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

"%APP%" setup-peripherals "%CONFIG%"
if errorlevel 1 (
    echo.
    echo Peripheral configuration was not completed.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)
"%APP%" check "%CONFIG%"
if not defined HOST_B_NO_PAUSE pause
