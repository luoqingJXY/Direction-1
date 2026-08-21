@echo off
setlocal
set "PACKAGE_ROOT=%~dp0"
if defined HOST_B_INSTALL_ROOT (
    set "INSTALL_ROOT=%HOST_B_INSTALL_ROOT%"
) else (
    set "INSTALL_ROOT=%LOCALAPPDATA%\ArtificialLifeHostB"
)
set "VERSION_ROOT=%INSTALL_ROOT%\versions\peripheral_base_v10"
set "APP=%VERSION_ROOT%\host_b_tool.exe"

if not exist "%PACKAGE_ROOT%payload\host_b_tool\host_b_tool.exe" (
    echo Package payload is incomplete.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

if not exist "%VERSION_ROOT%" mkdir "%VERSION_ROOT%"
robocopy "%PACKAGE_ROOT%payload\host_b_tool" "%VERSION_ROOT%" /E /R:1 /W:1 >nul
if errorlevel 8 (
    echo Failed to copy program files.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

if not exist "%INSTALL_ROOT%\tools" mkdir "%INSTALL_ROOT%\tools"
robocopy "%PACKAGE_ROOT%hardware_probe" "%INSTALL_ROOT%\tools" /E /R:1 /W:1 >nul
if errorlevel 8 (
    echo Failed to copy hardware probe files.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

if not exist "%INSTALL_ROOT%\virtual_microphone" mkdir "%INSTALL_ROOT%\virtual_microphone"
robocopy "%PACKAGE_ROOT%virtual_microphone" "%INSTALL_ROOT%\virtual_microphone" /E /R:1 /W:1 >nul
if errorlevel 8 (
    echo Failed to copy the official virtual microphone package.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

copy /Y "%PACKAGE_ROOT%check_host_b.cmd" "%INSTALL_ROOT%\check_host_b.cmd" >nul
copy /Y "%PACKAGE_ROOT%capture_one_frame.cmd" "%INSTALL_ROOT%\capture_one_frame.cmd" >nul
copy /Y "%PACKAGE_ROOT%capture_reference_frame.cmd" "%INSTALL_ROOT%\capture_reference_frame.cmd" >nul
copy /Y "%PACKAGE_ROOT%run_visual_link.cmd" "%INSTALL_ROOT%\run_visual_link.cmd" >nul
copy /Y "%PACKAGE_ROOT%configure_peripherals.cmd" "%INSTALL_ROOT%\configure_peripherals.cmd" >nul
copy /Y "%PACKAGE_ROOT%run_life_link.cmd" "%INSTALL_ROOT%\run_life_link.cmd" >nul
copy /Y "%PACKAGE_ROOT%install_virtual_microphone.cmd" "%INSTALL_ROOT%\install_virtual_microphone.cmd" >nul

if not exist "%INSTALL_ROOT%\host_b.toml" (
    "%APP%" initialize "%INSTALL_ROOT%\host_b.toml"
    if errorlevel 1 (
        echo Failed to create host_b.toml.
        if not defined HOST_B_NO_PAUSE pause
        exit /b 1
    )
)

"%APP%" system
if errorlevel 1 (
    echo Installed files, but the program self-check failed.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

echo.
echo Installed to: %INSTALL_ROOT%
echo Configuration: %INSTALL_ROOT%\host_b.toml
echo Next steps:
echo 1. Run install_virtual_microphone.cmd and restart Windows.
echo 2. Run configure_peripherals.cmd once.
echo 3. Run check_host_b.cmd.
if not defined HOST_B_NO_PAUSE pause
exit /b 0
