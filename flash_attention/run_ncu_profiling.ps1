# NCU Profiling Script for Flash Attention
# 需要以管理员身份运行
# 右键单击此脚本选择 "使用 PowerShell 运行"，或在管理员 PowerShell 中执行

Write-Output "╔════════════════════════════════════════════════════════════╗"
Write-Output "║  NVIDIA Nsight Compute - Flash Attention Profiling        ║"
Write-Output "║  需要管理员权限来访问 GPU Performance Counters             ║"
Write-Output "╚════════════════════════════════════════════════════════════╝"
Write-Output ""

# 检查管理员权限
$adminCheck = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $adminCheck) {
    Write-Output "⚠ 警告: 未以管理员身份运行"
    Write-Output "请按以下步骤操作:"
    Write-Output "1. 按 Win+X 打开系统菜单"
    Write-Output "2. 选择 'Windows PowerShell (管理员)' 或 'Terminal (管理员)'"
    Write-Output "3. 运行: & '$(Get-Location)\run_ncu_profiling.ps1'"
    Write-Output ""
    Read-Host "按 Enter 退出"
    exit 1
}

Write-Output "✓ 以管理员身份运行"
Write-Output ""

# 定义路径
$ncuPath = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$flashExe = Join-Path $scriptDir "flash_attn.exe"
$reportDir = Join-Path $scriptDir "ncu_report"
$reportPath = Join-Path $reportDir "flash_attn_report"

# 验证文件
Write-Output "=== 检查文件 ==="
Write-Output "NCU Tool:        $(if (Test-Path $ncuPath) {'✓'} else {'✗'}) $ncuPath"
Write-Output "Flash Exe:       $(if (Test-Path $flashExe) {'✓'} else {'✗'}) $flashExe"
Write-Output "Report Dir:      $reportDir"
Write-Output ""

if (-not (Test-Path $ncuPath)) {
    Write-Output "✗ 错误: Nsight Compute 未找到"
    Read-Host "按 Enter 退出"
    exit 1
}

if (-not (Test-Path $flashExe)) {
    Write-Output "✗ 错误: flash_attn.exe 未找到"
    Read-Host "按 Enter 退出"
    exit 1
}

# 创建报告目录
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

Write-Output "╔════════════════════════════════════════════════════════════╗"
Write-Output "║  Profile 1: Full Analysis (完整分析)                       ║"
Write-Output "║  - 包含所有性能指标                                        ║"
Write-Output "║  - 需要较长时间 (~2-3 分钟)                               ║"
Write-Output "╚════════════════════════════════════════════════════════════╝"
Write-Output ""

$startTime = Get-Date
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] 开始 Full profiling..."
Write-Output ""

& $ncuPath --set full `
    -o $reportPath `
    $flashExe 2>&1 | Tee-Object -FilePath "$reportDir\ncu_full_output.log"

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Output ""
Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Profiling 完成 (耗时: ${duration}s)"
Write-Output ""

# 检查结果
if (Test-Path "$reportPath.ncu-rep") {
    $fileSize = (Get-Item "$reportPath.ncu-rep").Length / 1MB
    Write-Output "✓ 完整报告生成: $reportPath.ncu-rep ($([Math]::Round($fileSize, 2)) MB)"
    Write-Output ""
    Write-Output "打开报告:"
    Write-Output "  方式 1: GUI (推荐)"
    Write-Output "    & '$ncuPath' --ui '$reportPath.ncu-rep'"
    Write-Output ""
    Write-Output "  方式 2: 命令行查看"
    Write-Output "    & '$ncuPath' --csv -i '$reportPath.ncu-rep'"
} else {
    Write-Output "✗ 报告生成失败"
    Write-Output "检查日志: $reportDir\ncu_full_output.log"
}

Write-Output ""
Write-Output "生成的文件:"
Get-ChildItem $reportDir -File | Select-Object @{n='Name';e={$_.Name}}, @{n='Size(KB)';e={[Math]::Round($_.Length/1KB,2)}}

Write-Output ""
Write-Output "════════════════════════════════════════════════════════════"
Read-Host "按 Enter 关闭此窗口"
