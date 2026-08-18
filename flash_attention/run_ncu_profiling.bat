@echo off
REM NCU Profiling Script for Flash Attention
REM 需要以管理员身份运行此批处理文件

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  NVIDIA Nsight Compute - Flash Attention Profiling        ║
echo ║  需要管理员权限来访问 GPU Performance Counters             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM 检查管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠ 未以管理员身份运行！
    echo.
    echo 请按以下步骤操作:
    echo 1. 右键单击此批处理文件 (.bat)
    echo 2. 选择 "以管理员身份运行"
    echo 或
    echo 1. 按 Win+X 打开系统菜单
    echo 2. 选择 'Windows PowerShell (管理员)' 
    echo 3. 进入此目录，运行: .\run_ncu_profiling.bat
    echo.
    pause
    exit /b 1
)

echo ✓ 以管理员身份运行
echo.

REM 设置变量
set NCU_PATH=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe
set FLASH_EXE=%~dp0flash_attn.exe
set REPORT_DIR=%~dp0ncu_report
set REPORT_PATH=%REPORT_DIR%\flash_attn_report

echo =══════════════════════════════════════════════════════════════
echo === 检查文件
echo =══════════════════════════════════════════════════════════════
echo.

if not exist "%NCU_PATH%" (
    echo ✗ 错误: Nsight Compute 未找到
    echo 预期路径: %NCU_PATH%
    echo.
    pause
    exit /b 1
)
echo ✓ NCU Tool: %NCU_PATH%

if not exist "%FLASH_EXE%" (
    echo ✗ 错误: flash_attn.exe 未找到
    echo 预期路径: %FLASH_EXE%
    echo.
    pause
    exit /b 1
)
echo ✓ Flash Exe: %FLASH_EXE%

if not exist "%REPORT_DIR%" (
    mkdir "%REPORT_DIR%"
)
echo ✓ Report Dir: %REPORT_DIR%
echo.

echo ╔════════════════════════════════════════════════════════════╗
echo ║  开始 NCU Profiling - Full Analysis (完整分析)              ║
echo ║  - 包含所有性能指标                                        ║
echo ║  - 需要较长时间 (~2-3 分钟)                               ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo 命令:
echo "%NCU_PATH%" --set full -o "%REPORT_PATH%" "%FLASH_EXE%"
echo.
echo 请等待...
echo.

REM 运行 NCU profiling
"%NCU_PATH%" --set full -o "%REPORT_PATH%" "%FLASH_EXE%"

echo.
echo =══════════════════════════════════════════════════════════════
echo === 完成
echo =══════════════════════════════════════════════════════════════
echo.

REM 检查结果
if exist "%REPORT_PATH%.ncu-rep" (
    echo ✓ 报告生成成功!
    echo   文件: %REPORT_PATH%.ncu-rep
    echo.
    echo 打开报告方式:
    echo   1. GUI 查看:
    echo      "%NCU_PATH%" --ui "%REPORT_PATH%.ncu-rep"
    echo.
    echo   2. CSV 导出:
    echo      "%NCU_PATH%" --csv -i "%REPORT_PATH%.ncu-rep"
    echo.
    echo   3. 文本格式:
    echo      "%NCU_PATH%" -i "%REPORT_PATH%.ncu-rep"
) else (
    echo ✗ 报告生成失败
    echo 请检查上方错误信息
)

echo.
echo 报告目录内容:
dir "%REPORT_DIR%" /B
echo.

pause
