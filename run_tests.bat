
@echo off
setlocal

title PDF Evidence Explorer - Tests

cd /d "%~dp0"

echo.
echo ==========================================
echo       PDF EVIDENCE EXPLORER TESTS
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Create it using:
    echo python -m venv .venv
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

echo Running automated tests...
echo.

python -m pytest tests -v

if errorlevel 1 (
    echo.
    echo TESTS FAILED.
    pause
    exit /b 1
)

echo.
echo ALL TESTS PASSED.
pause

endlocal
