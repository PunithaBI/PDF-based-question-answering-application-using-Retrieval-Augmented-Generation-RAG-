
@echo off
setlocal

title PDF Evidence Explorer

cd /d "%~dp0"

echo.
echo ==========================================
echo       PDF EVIDENCE EXPLORER
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo.
    echo Create it first using:
    echo python -m venv .venv
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...

call ".venv\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking dependencies...

python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting FastAPI...
echo.
echo Dashboard: http://127.0.0.1:8000
echo API docs:  http://127.0.0.1:8000/docs
echo.
echo Press CTRL+C to stop the server.
echo.

start "" http://127.0.0.1:8000

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause

endlocal
