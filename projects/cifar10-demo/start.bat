@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=py"
where py >nul 2>nul || set "PY=python"

where %PY% >nul 2>nul
if errorlevel 1 (
  echo.
  echo   没有找到 Python。请先安装 Python，或者手动运行：
  echo     py -m http.server 8000 --bind 0.0.0.0
  echo.
  pause
  exit /b 1
)

"%PY%" "tools\serve.py" %*

echo.
pause
