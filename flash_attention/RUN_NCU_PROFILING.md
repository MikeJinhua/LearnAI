# 🚀 运行 NCU Profiling 的步骤

## V1 / V2 单独采集（推荐）

程序提供了只启动目标 kernel 的 profiling 模式。每种模式会启动两次同名 kernel：
第一次 warmup，第二次供 NCU 采集。这样报告里不会再混入 100 次 benchmark 和 naive kernels。

```powershell
cd F:\AI\flash_attention

# UTF-8 源码在中文 Windows 上需要显式传给 MSVC
nvcc -O2 -arch=sm_86 -lineinfo -Xcompiler /utf-8 flash_attn.cu -o flash_attn_v2.exe

$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"

# V1：跳过 warmup，只采第二次 launch
& $ncu --set full --kernel-name regex:flash_attn_kernel --launch-skip 1 --launch-count 1 `
    -o ncu_report\flash_v1 .\flash_attn_v2.exe --profile-v1

# V2
& $ncu --set full --kernel-name regex:flash_attn_v2_kernel --launch-skip 1 --launch-count 1 `
    -o ncu_report\flash_v2 .\flash_attn_v2.exe --profile-v2
```

打开并对比：

```powershell
ncu-ui ncu_report\flash_v1.ncu-rep
ncu-ui ncu_report\flash_v2.ncu-rep
```

重点比较 Occupancy、Scheduler Statistics、Warp State Statistics，以及 Source 页的
`Warp Stall Sampling (Not-Issued Samples)`。`Memory Throughput` 是综合指标，不等于 DRAM throughput。

已为你准备好可以直接运行的脚本。选择以下任意一种方法执行：

## 方法 1：批处理文件（最简单）✨

**步骤：**
1. 打开文件夹: `F:\AI.worktrees\todo-file-review\flash_attention`
2. **右键单击** `run_ncu_profiling.bat` 
3. 选择 **"以管理员身份运行"**
4. 等待 profiling 完成（约 2-3 分钟）

![image](https://imgur.com/4H8Kx.png)

---

## 方法 2：PowerShell 脚本

**步骤：**
1. 按 `Win+X`，选择 **"Windows PowerShell (管理员)"**
2. 进入目录：
   ```powershell
   cd F:\AI.worktrees\todo-file-review\flash_attention
   ```
3. 运行脚本：
   ```powershell
   .\run_ncu_profiling.ps1
   ```
4. 等待完成

---

## 方法 3：直接命令行

在**管理员 PowerShell** 中：

```powershell
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"
$exe = "F:\AI.worktrees\todo-file-review\flash_attention\flash_attn.exe"
$out = "F:\AI.worktrees\todo-file-review\flash_attention\ncu_report\flash_attn_report"

& $ncu --set full -o $out $exe
```

---

## ✅ 预期输出

运行后会看到：

```
==PROF== Connected to process XXXXX
==PROF== Collecting data...
==PROF== ... (数据收集中)
==PROF== Disconnected from process XXXXX
```

最后生成报告文件：
```
ncu_report/
  ├── flash_attn_report.ncu-rep      # 完整报告（GUI 查看）
  ├── ncu_full_output.log            # 日志
  └── ...
```

---

## 🔍 查看报告

生成后可以用以下方式查看：

### GUI 查看（推荐）
```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe" --ui "F:\AI.worktrees\todo-file-review\flash_attention\ncu_report\flash_attn_report.ncu-rep"
```

### CSV 导出
```powershell
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"
& $ncu --csv -i "ncu_report\flash_attn_report.ncu-rep" | Out-File "ncu_report\results.csv"
```

---

## 📊 关键指标说明

Profiling 完成后会收集以下数据：

| 指标 | 说明 |
|------|------|
| **Memory Throughput** | Flash Attn 应 < Naive GPU（内存节省证据） |
| **L2 Cache Hit Rate** | Flash Attn 应 > Naive GPU（局部性更好） |
| **Achieved Occupancy** | GPU 利用率（理想 > 80%） |
| **Global Memory Accesses** | 全局内存访问数（Flash Attn 应明显低于 Naive） |

---

## 💡 故障排查

### 如果报错 `ERR_NVGPUCTRPERM`
- ✓ 这是**权限问题**，需要管理员身份
- 确保右键点击脚本选择 "以管理员身份运行"

### 如果报错 `ncu.exe not found`
- 检查 Nsight Compute 安装路径
- 可能需要重新安装或更新路径

### 如果 profiling 很慢
- 这是**正常的**，Full set 需要收集很多数据
- 可以用 `--set basic` 快速运行（改脚本里的参数）

---

**准备好了吗？** 🚀 
- 方法 1: 双击运行 `.bat` 文件
- 方法 2: 管理员 PowerShell 运行 `.ps1` 脚本  
- 方法 3: 复制粘贴命令

选择最方便的方式执行吧！
