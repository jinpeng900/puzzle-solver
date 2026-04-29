@echo off
REM ============================================================
REM 编译脚本
REM ============================================================

echo === Compiling solver_test (local test version) ===
g++ -std=c++17 -O3 -o solver_test.exe solver_core.cpp solver_test.cpp
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: compile failed!
    pause
    exit /b 1
)
echo OK: solver_test.exe built

echo.
echo === Compiling solver (online submit version) ===
g++ -std=c++17 -O3 -o solver.exe solver_core.cpp solver.cpp
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: compile failed!
    pause
    exit /b 1
)
echo OK: solver.exe built

echo.
echo === Done ===
echo   solver.exe      - online submission binary
echo   solver_test.exe - local test binary
echo.
echo Run: solver_test.exe 42 114514 888888
pause
