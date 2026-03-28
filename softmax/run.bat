@echo off
cd /d "%~dp0build"
cmake --build . --config Release && .\Release\softmax.exe
