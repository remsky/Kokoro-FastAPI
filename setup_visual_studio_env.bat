@echo off

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VS_INSTALL_DIR=%%i
)

if exist "%VS_INSTALL_DIR%\Common7\Tools\VsDevCmd.bat" (
    call "%VS_INSTALL_DIR%\Common7\Tools\VsDevCmd.bat" -arch=x64
) else (
    echo.
    echo Visual Studio Build Tools not found. Run this command to get the installer:
    echo.
    echo winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
    echo.
    exit /b 1
)