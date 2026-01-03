# Batch Size Debugging Guide

This guide helps you find the optimal batch size for your GPU using the included PowerShell script.

## Quick Start

```powershell
# Basic usage - tests default batch sizes (64, 96, 128, 160, 192, 224, 256, 320, 384)
.\debug_batching.ps1 -AudioFile path\to\recording.wav

# Test specific batch sizes
.\debug_batching.ps1 -AudioFile recording.wav -BatchSizes @(128, 192, 256)

# With ONNX Runtime profiling enabled
.\debug_batching.ps1 -AudioFile recording.wav -EnableProfiling
```

## What It Does

The script will:

1. **Test each batch size** and measure:
   - Total processing time
   - Throughput (segments/second)
   - VRAM usage and allocation patterns
   - GPU utilization (compute and memory)
   - Power consumption and temperature

2. **Monitor GPU in real-time** using `nvidia-smi`:
   - Samples GPU metrics every 200ms during inference
   - Tracks utilization, memory, temperature, power

3. **Parse ONNX Runtime logs** to find:
   - Total VRAM allocated
   - Large memory allocations (>= 1GB)
   - BFCArena extension patterns

4. **Generate comprehensive report**:
   - Performance comparison table
   - Optimal batch size recommendation
   - Degradation point detection
   - Memory-bound vs compute-bound analysis

## Example Output

```
================================================================================
Recommendations
================================================================================
Optimal Batch Size: 192
  Throughput: 50.72 segments/s
  VRAM Usage: 6.84 GB
  GPU Utilization: avg 94.3% / max 98.0%
  Memory Utilization: avg 76.2% / max 82.1%

⚠ Performance degradation detected at batch size 256
  Slowdown: 18.5% compared to batch 192
  VRAM increase: 6.84 GB → 8.33 GB (+1.49 GB)
```

## Understanding the Results

### Optimal Batch Size
The script recommends the batch size with **highest throughput** (segments/second).

### Performance Degradation
If throughput **drops by 5% or more** when increasing batch size, it indicates:
- **Memory bandwidth saturation** - VRAM access is bottleneck
- **L2 cache exhaustion** - working set too large
- **Excessive allocations** - memory management overhead

### Memory Analysis

**Large Allocations (>= 1GB)**:
```
Batch 192: (none)
Batch 256: 2.27GB, 1.02GB  ← Red flag!
```
Multiple 1GB+ allocations indicate memory pressure.

**Total VRAM Usage**:
- < 8 GB: Usually optimal
- 8-12 GB: May hit bandwidth limits
- > 12 GB: Likely degraded performance

### GPU Utilization

**Memory-Bound** (bad):
```
GPU Util: 70%
Mem Util: 95%  ← Memory bottleneck!
```
GPU cores are waiting for memory.

**Compute-Bound** (good):
```
GPU Util: 95%
Mem Util: 75%
```
GPU cores are doing work.

## Output Files

All debug data is saved to `batch_debug_YYYYMMDD_HHMMSS/`:

- `batch_results.csv` - Summary table (import into Excel)
- `batch_128_log.txt` - Full ORT logs for batch 128
- `batch_128_gpu.csv` - GPU metrics for batch 128
- (one log + csv per batch size tested)

## Parameters

```powershell
-BirdaPath <path>         # Path to birda.exe (default: .\birda.exe)
-AudioFile <path>         # Audio file to test (required)
-BatchSizes <array>       # Batch sizes to test (default: 64-384)
-Confidence <float>       # Confidence threshold (default: 0.8)
-MonitorInterval <ms>     # GPU sample interval (default: 200)
-EnableProfiling          # Enable ORT profiling (creates .json files)
```

## Tips

### For Quick Testing
Test fewer batch sizes around your expected optimal:
```powershell
.\debug_batching.ps1 -AudioFile test.wav -BatchSizes @(128, 160, 192, 224)
```

### For Deep Analysis
Enable profiling and test wide range:
```powershell
.\debug_batching.ps1 -AudioFile test.wav -EnableProfiling
```

### For Production Validation
Use a representative audio file (similar length to typical use case).

## Interpreting Results for Different GPUs

### 4GB VRAM (e.g., GTX 1650)
- Expect optimal: 32-64
- Degradation at: 96-128
- Max VRAM: 3-4 GB

### 8GB VRAM (e.g., RTX 3060 Ti)
- Expect optimal: 96-128
- Degradation at: 192-256
- Max VRAM: 6-7 GB

### 16GB VRAM (e.g., RTX 5080)
- Expect optimal: 160-192
- Degradation at: 256-320
- Max VRAM: 6-8 GB

### 24GB VRAM (e.g., RTX 4090)
- Expect optimal: 256-384
- Degradation at: 512+
- Max VRAM: 10-14 GB

**Note**: Optimal batch often uses **only 50-60% of total VRAM** due to memory bandwidth limits!

## Troubleshooting

### Script fails immediately
- Check `birda.exe` path with `-BirdaPath`
- Verify audio file exists
- Ensure NVIDIA drivers installed (`nvidia-smi --version`)

### GPU monitoring shows all zeros
- Run PowerShell as Administrator
- Check nvidia-smi works: `nvidia-smi`

### ORT profiling not generating files
- Add `-EnableProfiling` flag
- Check current directory for `onnxruntime_profile_*.json`

## Clean Up

The script creates a timestamped directory with all debug data. Safe to delete when done:

```powershell
Remove-Item -Recurse batch_debug_*
```

Files are already in `.gitignore` so won't be committed.
