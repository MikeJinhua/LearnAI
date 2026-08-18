# 🔧 Nsight Compute 安装指南

## 快速安装步骤

### 方法 1：直接从 NVIDIA 官网下载（推荐）

1. **访问下载页面**
   - 浏览器打开: https://developer.nvidia.com/nsight-compute
   - 需要使用 NVIDIA Developer Account（免费注册）

2. **选择版本**
   - 选择 "Latest" 或 "Nsight Compute 2024.1" 版本
   - Windows 版本选择: `nsight-compute-windows-x.y.z.exe`

3. **下载并安装**
   ```
   双击下载的 exe 文件运行安装程序
   选择安装路径（默认：C:\Program Files\NVIDIA\Nsight Compute）
   完成安装
   ```

4. **验证安装**
   ```powershell
   # 在 PowerShell 中运行
   & "C:\Program Files\NVIDIA\Nsight Compute\ncu.exe" --version
   ```

### 方法 2：从 CUDA Toolkit 安装程序重新安装

如果已安装 CUDA Toolkit 13.2，可以重新运行安装程序选择 Nsight Compute：

```
1. 下载 CUDA Toolkit 13.2: https://developer.nvidia.com/cuda-downloads
2. 运行安装程序
3. 选择 "Custom Installation"
4. 勾选 "Nsight Compute"
5. 完成安装
```

### 方法 3：命令行安装（如果已有安装程序）

```powershell
cd "C:\path\to\installer"
.\nsight-compute-windows-x.y.z.exe --silent --accept-eula
```

---

## 使用 Nsight Compute 进行 Profiling

安装后，运行以下命令生成 Flash Attention 的性能报告：

```powershell
# 进入 flash_attention 目录
cd F:\AI.worktrees\todo-file-review\flash_attention

# 运行 NCU profiling
$ncu = "C:\Program Files\NVIDIA\Nsight Compute\ncu.exe"

# 完整 profiling（生成详细报告）
& $ncu --set full `
    --kernel-name flash_attn_kernel `
    --kernel-name qk_dot_kernel `
    --kernel-name softmax_kernel `
    --kernel-name sv_dot_kernel `
    -o ncu_report\flash_attn_report `
    flash_attn.exe

# 快速 profiling（快速收集基础指标）
& $ncu --set basic -o ncu_report\flash_attn_quick flash_attn.exe

# 生成 CSV 报告
& $ncu --set full --export csv -o ncu_report\flash_attn_report.csv flash_attn.exe

# 打开 GUI 查看报告
& $ncu --ui ncu_report\flash_attn_report.ncu-rep
```

---

## 关键指标说明

运行后 Nsight Compute 会收集以下重要指标：

### Memory Metrics
- **Global Load Efficiency** - 全局内存加载效率（理想值 100%）
- **Global Store Efficiency** - 全局内存存储效率
- **L1/L2/L3 Cache Hit Rate** - 各级缓存命中率
- **Memory Throughput (GB/s)** - 实际内存吞吐量

### Compute Metrics
- **SM (Streaming Multiprocessor) Utilization** - SM 利用率
- **Achieved Occupancy** - 实现占有率
- **Instruction Throughput** - 指令吞吐量

### Roofline Analysis
- **Memory Bound vs Compute Bound** - 瓶颈分析
- **Roof Line** - 性能天花板

### 性能对比建议
- 分别对 Flash Attention 和 Naive GPU 运行 NCU
- 对比 Memory Throughput 和 L2 Cache Hit Rate
- 验证 Flash Attention 的 "内存节省" 理论

---

## 如果无法安装

如果无法安装 Nsight Compute，可以使用以下替代方案：

### 1. 使用 NVIDIA Nsight Systems（命令行）
```powershell
# 如果已安装
nsys profile -o trace flash_attn.exe
nsys stats trace.nsys-rep
```

### 2. 使用 CUPTI 基础性能收集
```powershell
# CUPTI 包含在 CUDA Toolkit 中
# 位置: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\extras\CUPTI
```

### 3. 基于实测数据的理论分析（已完成）
- 参考 `PROFILING_REPORT.md`
- 使用 benchmark 时间推算带宽利用率
- 基于硬件规格进行 roofline 分析

---

## 状态

- ✓ Benchmark 运行完毕（所有模块已完成）
- ⏳ NCU 待安装（需要手动从 NVIDIA 官网下载）
- ✓ 理论性能分析已完成

**下一步**: 
1. 按上述步骤安装 Nsight Compute
2. 运行 profiling 命令
3. 将报告截图保存到 `ncu_report/` 目录
4. 更新 flash_attention/README.md 中的 NCU 部分
