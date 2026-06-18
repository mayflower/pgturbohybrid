@echo off

set "VSINSTALLDIR="
set "NMAKE="
set "CLDIR="
set "UCRT_INCLUDE="
set "UCRT_LIB="
set "SDK_SHARED_INCLUDE="
set "SDK_UM_INCLUDE="
set "SDK_UM_LIB="
set "KITSROOT="

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALLDIR=%%i"
if not defined VSINSTALLDIR exit /b 1

call "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

for /f "tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\Microsoft\Windows Kits\Installed Roots" /v KitsRoot10 2^>nul ^| findstr /i "KitsRoot10"') do set "KITSROOT=%%b"
if not defined KITSROOT if exist "%ProgramFiles(x86)%\Windows Kits\10" set "KITSROOT=%ProgramFiles(x86)%\Windows Kits\10\"

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

if defined KITSROOT if exist "%KITSROOT%Include" for /r "%KITSROOT%Include" %%u in (crtdefs.h) do if not defined UCRT_INCLUDE set "UCRT_INCLUDE=%%~dpu"
if defined KITSROOT if exist "%KITSROOT%Include" for /r "%KITSROOT%Include" %%u in (winapifamily.h) do if not defined SDK_SHARED_INCLUDE set "SDK_SHARED_INCLUDE=%%~dpu"
if defined KITSROOT if exist "%KITSROOT%Include" for /r "%KITSROOT%Include" %%u in (windows.h) do if not defined SDK_UM_INCLUDE set "SDK_UM_INCLUDE=%%~dpu"
if defined KITSROOT if exist "%KITSROOT%Lib" for /r "%KITSROOT%Lib" %%u in (ucrt.lib) do if not defined UCRT_LIB set "UCRT_LIB=%%~dpu"
if defined KITSROOT if exist "%KITSROOT%Lib" for /r "%KITSROOT%Lib" %%u in (kernel32.lib) do if not defined SDK_UM_LIB set "SDK_UM_LIB=%%~dpu"
for /d %%d in ("%ProgramFiles(x86)%\Windows Kits\10\Include\*") do if not defined UCRT_INCLUDE if exist "%%~fd\ucrt\crtdefs.h" set "UCRT_INCLUDE=%%~fd\ucrt"
for /d %%d in ("%ProgramFiles(x86)%\Windows Kits\10\Lib\*") do if not defined UCRT_LIB if exist "%%~fd\ucrt\x64\ucrt.lib" set "UCRT_LIB=%%~fd\ucrt\x64"
if not defined UCRT_INCLUDE for /f "delims=" %%u in ('where /R "%ProgramFiles(x86)%\Windows Kits\10\Include" crtdefs.h 2^>nul') do if not defined UCRT_INCLUDE set "UCRT_INCLUDE=%%~dpu"
if not defined SDK_SHARED_INCLUDE for /f "delims=" %%u in ('where /R "%ProgramFiles(x86)%\Windows Kits\10\Include" winapifamily.h 2^>nul') do if not defined SDK_SHARED_INCLUDE set "SDK_SHARED_INCLUDE=%%~dpu"
if not defined SDK_UM_INCLUDE for /f "delims=" %%u in ('where /R "%ProgramFiles(x86)%\Windows Kits\10\Include" windows.h 2^>nul') do if not defined SDK_UM_INCLUDE set "SDK_UM_INCLUDE=%%~dpu"
if not defined UCRT_LIB for /f "delims=" %%u in ('where /R "%ProgramFiles(x86)%\Windows Kits\10\Lib" ucrt.lib 2^>nul') do if not defined UCRT_LIB set "UCRT_LIB=%%~dpu"
if not defined SDK_UM_LIB for /f "delims=" %%u in ('where /R "%ProgramFiles(x86)%\Windows Kits\10\Lib" kernel32.lib 2^>nul') do if not defined SDK_UM_LIB set "SDK_UM_LIB=%%~dpu"
if defined UCRT_INCLUDE set "INCLUDE=%UCRT_INCLUDE%;%INCLUDE%"
if defined SDK_SHARED_INCLUDE set "INCLUDE=%SDK_SHARED_INCLUDE%;%INCLUDE%"
if defined SDK_UM_INCLUDE set "INCLUDE=%SDK_UM_INCLUDE%;%INCLUDE%"
if defined UCRT_LIB set "LIB=%UCRT_LIB%;%LIB%"
if defined SDK_UM_LIB set "LIB=%SDK_UM_LIB%;%LIB%"

where cl
if errorlevel 1 exit /b 1
if not exist "%NMAKE%" exit /b 1
echo VSINSTALLDIR=%VSINSTALLDIR%
echo KITSROOT=%KITSROOT%
echo UniversalCRTSdkDir=%UniversalCRTSdkDir%
echo UCRTVersion=%UCRTVersion%
echo UCRT_INCLUDE=%UCRT_INCLUDE%
echo SDK_SHARED_INCLUDE=%SDK_SHARED_INCLUDE%
echo SDK_UM_INCLUDE=%SDK_UM_INCLUDE%
echo UCRT_LIB=%UCRT_LIB%
echo SDK_UM_LIB=%SDK_UM_LIB%
if not defined UCRT_INCLUDE exit /b 1
if not exist "%UCRT_INCLUDE%crtdefs.h" if not exist "%UCRT_INCLUDE%\crtdefs.h" exit /b 1

exit /b 0
