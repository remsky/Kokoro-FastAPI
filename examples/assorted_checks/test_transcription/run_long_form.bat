@echo off
REM Long-form runner. Usage: run_long_form.bat [full|synth|transcribe|short]
REM   full       synth + transcribe on the whole book (default)
REM   synth      synth only, no transcription (fast on GPU)
REM   short      synth + transcribe on a 65k-char slice (~chapters 1-7)
REM   transcribe transcribe only, reuses the wav from a previous synth run
REM Double-click runs 'full'. Output streams to the window and to a log file.

setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

set "EXTRA_ARGS="
if /I "%MODE%"=="full"       goto :full
if /I "%MODE%"=="synth"      goto :synth
if /I "%MODE%"=="transcribe" goto :transcribe
if /I "%MODE%"=="short"      goto :short
echo Unknown mode: %MODE%
echo Usage: %~nx0 [full^|synth^|transcribe^|short]
exit /b 2

:full
set "TAG=journey_all_full"
goto :run
:synth
set "TAG=journey_all_synth"
set "EXTRA_ARGS=--synth-only"
goto :run
:transcribe
set "TAG=journey_all_transcribe"
set "EXTRA_ARGS=--transcribe-only"
goto :run
:short
set "TAG=journey_short_full"
set "LONGFORM_CHARS=65000"
goto :run

:run
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\..\.."
set "LOG_DIR=%SCRIPT_DIR%output_long_form"
set "LOG_FILE=%LOG_DIR%\%TAG%.log"
set "LONGFORM_INPUT=examples/assorted_checks/test_transcription/input/journey_all.txt.gz"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

pushd "%REPO_ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Write-Output ('Run started ' + (Get-Date)) | Tee-Object -FilePath '%LOG_FILE%';" ^
    "uv run --project examples python examples/assorted_checks/test_transcription/test_long_form.py %EXTRA_ARGS% 2>&1 | Tee-Object -FilePath '%LOG_FILE%' -Append;" ^
    "Write-Output ('Run finished ' + (Get-Date)) | Tee-Object -FilePath '%LOG_FILE%' -Append;"
set RC=%ERRORLEVEL%
popd

echo.
echo === DONE (mode=%MODE%, exit %RC%) -- log: %LOG_FILE%
pause
