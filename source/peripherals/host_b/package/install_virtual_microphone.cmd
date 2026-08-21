@echo off
setlocal
set "INSTALL_ROOT=%~dp0"
if not exist "%INSTALL_ROOT%virtual_microphone\vac471lite.zip" (
    set "INSTALL_ROOT=%LOCALAPPDATA%\ArtificialLifeHostB\"
)
set "VAC_ARCHIVE=%INSTALL_ROOT%virtual_microphone\vac471lite.zip"

if not exist "%VAC_ARCHIVE%" (
    echo The original VAC 4.71 Lite package is missing.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

echo This installs the official VAC 4.71 Lite virtual audio driver.
echo It is used only as the silent third auditory input for this experiment.
echo Accept the administrator prompt and finish the visible driver installer.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$source=$env:VAC_ARCHIVE; $target=Join-Path $env:TEMP ('ArtificialLifeVAC471Lite_' + [Guid]::NewGuid().ToString('N')); Expand-Archive -LiteralPath $source -DestinationPath $target; $setup=Join-Path $target 'setup.exe'; if (-not (Test-Path -LiteralPath $setup)) { throw 'VAC setup.exe is missing' }; $process=Start-Process -FilePath $setup -Verb RunAs -Wait -PassThru; exit $process.ExitCode"
if errorlevel 1 (
    echo.
    echo Virtual microphone installation did not complete.
    if not defined HOST_B_NO_PAUSE pause
    exit /b 1
)

echo.
echo Virtual microphone installer completed.
echo Restart Windows before running configure_peripherals.cmd.
if not defined HOST_B_NO_PAUSE pause
exit /b 0
