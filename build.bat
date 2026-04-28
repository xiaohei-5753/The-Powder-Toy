@echo off
cd /d %~dp0

call "E:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo == Configuring Meson ==
meson setup build --buildtype=debug --wipe

if %errorlevel% neq 0 (
    echo Configuring Meson failed
    pause
    exit /b %errorlevel%
)

echo == Building ==
meson compile -C build

if %errorlevel% neq 0 (
    echo Build failed
    pause
    exit /b %errorlevel%
)

echo == Copying output to build_dist ==
if not exist build_dist mkdir build_dist
copy /Y build\powder.exe build_dist\ > nul
copy /Y build\*.dll build_dist\ > nul

echo.
echo Build successful! Output in build_dist\powder.exe
echo.
pause
