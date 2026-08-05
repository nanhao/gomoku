@echo off
REM ============================================================================
REM  C++ MCTS one-click build script
REM  Usage:
REM    build_cpp.bat              — Default Release build
REM    build_cpp.bat --debug      — Debug build
REM    build_cpp.bat --test       — Release build + run C++ tests
REM    build_cpp.bat --clean      — Clean build dir and rebuild
REM ============================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM ---- Parse command line args ----
set BUILD_TYPE=Release
set RUN_TESTS=0
set CLEAN_BUILD=0

:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--debug"   (set BUILD_TYPE=Debug   & shift & goto :parse_args)
if /i "%~1"=="--release" (set BUILD_TYPE=Release & shift & goto :parse_args)
if /i "%~1"=="--test"    (set RUN_TESTS=1        & shift & goto :parse_args)
if /i "%~1"=="--clean"   (set CLEAN_BUILD=1      & shift & goto :parse_args)
echo [WARN] Unknown arg: %~1
shift
goto :parse_args
:end_parse

REM ---- Check prerequisites ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    exit /b 1
)

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch not installed. Run: pip install torch
    exit /b 1
)

python -c "import pybind11" >nul 2>&1
if errorlevel 1 (
    echo [WARN] pybind11 not found. Python extension will be skipped.
    echo        Install: pip install pybind11
)

REM ---- Detect cmake ----
set CMAKE_EXE=
for /f "delims=" %%i in ('python -c "import sys, os; d=os.path.join(sys.prefix,'Scripts','cmake.exe'); print(d if os.path.isfile(d) else '')" 2^>nul') do set CMAKE_EXE=%%i
if not defined CMAKE_EXE (
    for /f "delims=" %%i in ('python -c "import site, os; d=os.path.join(os.path.dirname(site.USER_SITE),'Scripts','cmake.exe') if site.USER_SITE else ''; print(d if d and os.path.isfile(d) else '')" 2^>nul') do set CMAKE_EXE=%%i
)
if not defined CMAKE_EXE (
    cmake --version >nul 2>&1
    if not errorlevel 1 (
        set CMAKE_EXE=cmake
    )
)
if not defined CMAKE_EXE (
    echo [ERROR] CMake not found. Install: pip install cmake
    exit /b 1
)
echo [INFO] cmake: %CMAKE_EXE%

REM ---- Detect pybind11 cmake dir ----
set PYBIND11_DIR=
for /f "delims=" %%i in ('python -c "import pybind11; print(pybind11.get_cmake_dir())" 2^>nul') do set PYBIND11_DIR=%%i
if defined PYBIND11_DIR (
    echo [INFO] pybind11: %PYBIND11_DIR%
)

REM ---- Clean (optional) ----
if %CLEAN_BUILD%==1 (
    echo [INFO] Cleaning build directory...
    if exist "cpp\build" rmdir /s /q "cpp\build"
)

REM ---- Create build directory ----
if not exist "cpp\build" mkdir "cpp\build"
cd cpp\build

REM ---- CMake Configure ----
echo.
echo ========================================
echo   CMake Configure (%BUILD_TYPE%)
echo ========================================

set CMAKE_ARGS=.. -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if defined PYBIND11_DIR (
    set CMAKE_ARGS=%CMAKE_ARGS% -Dpybind11_DIR="%PYBIND11_DIR%"
)

"%CMAKE_EXE%" %CMAKE_ARGS%
if errorlevel 1 (
    echo [ERROR] CMake configure failed
    cd ..\..
    exit /b 1
)

REM ---- Build ----
echo.
echo ========================================
echo   Building (%BUILD_TYPE%)
echo ========================================
"%CMAKE_EXE%" --build . --config %BUILD_TYPE%
if errorlevel 1 (
    echo [ERROR] Build failed
    cd ..\..
    exit /b 1
)

REM ---- Copy .pyd to Python module dir ----
set PYD_SRC=%BUILD_TYPE%\cpp_mcts.pyd
if not exist "%PYD_SRC%" set PYD_SRC=cpp_mcts.pyd
if exist "%PYD_SRC%" (
    echo [INFO] Copying cpp_mcts.pyd to cpp\python\
    copy /y "%PYD_SRC%" "..\python\cpp_mcts.pyd" >nul
    echo [ OK ] cpp_mcts.pyd ready

    REM ---- Copy required torch CPU DLLs (pyd 导入时即依赖这些) ----
    set TORCH_LIB_DIR=
    for /f "delims=" %%i in ('python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2^>nul') do set TORCH_LIB_DIR=%%i
    if defined TORCH_LIB_DIR (
        for %%d in (torch.dll torch_cpu.dll c10.dll) do (
            if exist "!TORCH_LIB_DIR!\%%d" (
                copy /y "!TORCH_LIB_DIR!\%%d" "..\python\%%d" >nul 2>&1
            )
        )
        echo [ OK ] Torch CPU DLLs copied
    )
) else (
    echo [WARN] cpp_mcts.pyd not found in build output
)

REM ---- Run tests (optional) ----
if %RUN_TESTS%==1 (
    echo.
    echo ========================================
    echo   Running C++ Tests
    echo ========================================
    ctest --output-on-failure -C %BUILD_TYPE%
)

echo.
echo ========================================
echo   Build Complete!
echo ========================================
cd ..\..
endlocal
