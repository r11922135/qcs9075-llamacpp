@echo off
if not defined _LLAMACPP_RELAUNCH ( set "_LLAMACPP_RELAUNCH=1" & cmd /k call "%~f0" %* & exit /b )
REM ===========================================================================
REM  push.bat - deploy llama.cpp (CPU build) and GGUF models to the QCS9075.
REM  Thin wrapper around push.ps1; all the logic lives there.
REM
REM  Double-click it, or pass options through:
REM      .\push.bat                 llama.cpp + Qwen3.5-9B Q4_0 (5.7 GB)
REM      .\push.bat -Model 35b      llama.cpp + Qwen3.6-35B-A3B Q4_0 (20.8 GB)
REM      .\push.bat -Model both
REM      .\push.bat -Model none     only update llama.cpp and scripts
REM      .\push.bat -Bench          push, then run bench.sh on the board
REM  Flags combine:  .\push.bat -Model 35b -Bench
REM
REM  The board downloads the models itself (it needs Wi-Fi), so nothing
REM  large is stored on this PC.
REM
REM  This file is deliberately pure ASCII with CRLF line endings. On a
REM  Traditional Chinese Windows the console reads it as cp950, and UTF-8
REM  Chinese bytes - even inside a REM comment - crash the parser and the
REM  window vanishes. Chinese belongs in push.ps1, not here.
REM ===========================================================================
setlocal
cd /d "%~dp0"

echo push.bat for qcs9075-llamacpp (v2, 2026-09-11)
echo.

REM Board-side scripts print UTF-8; without this the console shows mojibake.
chcp 65001 >nul 2>&1

where powershell >nul 2>&1
if errorlevel 1 (
    echo [FAIL] powershell.exe not found in PATH.
    goto :hold
)

where adb >nul 2>&1
if errorlevel 1 (
    echo [WARN] adb not found in PATH - push.ps1 will stop with an error.
    echo        Install platform-tools or add adb to PATH first.
    echo.
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push.ps1" %*
set RC=%ERRORLEVEL%
echo.
echo [exit code %RC%]

:hold
echo.
echo This window stays open. Close it when you are done.
endlocal
