@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 scripts\use_auro.py %*
) else (
  python scripts\use_auro.py %*
)

if not %errorlevel%==0 (
  echo.
  echo Auro did not start. Review the error above.
  pause
)
endlocal
