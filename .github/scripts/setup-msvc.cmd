@echo off

set "VSINSTALLDIR="
set "NMAKE="
set "CLDIR="

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALLDIR=%%i"
if not defined VSINSTALLDIR exit /b 1

call "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

if defined VCToolsInstallDir (
  if exist "%VCToolsInstallDir%bin\Hostx64\x64\nmake.exe" set "NMAKE=%VCToolsInstallDir%bin\Hostx64\x64\nmake.exe"
  if exist "%VCToolsInstallDir%bin\Hostx64\x64\cl.exe" set "CLDIR=%VCToolsInstallDir%bin\Hostx64\x64"
)

if not defined NMAKE for /d %%d in ("%VSINSTALLDIR%\VC\Tools\MSVC\*") do if not defined NMAKE if exist "%%~fd\bin\Hostx64\x64\nmake.exe" set "NMAKE=%%~fd\bin\Hostx64\x64\nmake.exe"
if not defined CLDIR for /d %%d in ("%VSINSTALLDIR%\VC\Tools\MSVC\*") do if not defined CLDIR if exist "%%~fd\bin\Hostx64\x64\cl.exe" set "CLDIR=%%~fd\bin\Hostx64\x64"
if not defined NMAKE exit /b 1
if not defined CLDIR exit /b 1

set "PATH=%CLDIR%;%PATH%"

if defined UniversalCRTSdkDir if defined UCRTVersion if exist "%UniversalCRTSdkDir%Include\%UCRTVersion%ucrt\crtdefs.h" set "INCLUDE=%UniversalCRTSdkDir%Include\%UCRTVersion%ucrt;%INCLUDE%"
if defined UniversalCRTSdkDir if defined UCRTVersion if exist "%UniversalCRTSdkDir%Lib\%UCRTVersion%ucrt\x64" set "LIB=%UniversalCRTSdkDir%Lib\%UCRTVersion%ucrt\x64;%LIB%"

where cl
if errorlevel 1 exit /b 1
if not exist "%NMAKE%" exit /b 1

exit /b 0
