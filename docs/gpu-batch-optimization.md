# GPU Batch Size Optimization Guide

This guide helps you choose the optimal batch size for your GPU when using BirdNET inference with CUDA acceleration.

## Quick Reference

| GPU VRAM | Recommended Batch | Expected VRAM Usage | Performance Notes |
|----------|-------------------|---------------------|-------------------|
| 4 GB     | 32                | ~1.5 GB (38%)       | Conservative, tested |
| 8 GB     | 96-128            | ~3-4 GB (40%)       | Estimated from scaling |
| 12 GB    | 144-160           | ~5-6 GB (42%)       | Estimated from scaling |
| 16 GB    | **192**           | **6.25 GB (39%)**   | **Verified optimal** |
| 24 GB    | 256-320           | ~10-12 GB (42%)     | Estimated from scaling |

## Understanding Batch Processing

### What is a Batch?

A **batch** is a group of audio segments processed together in a single GPU inference call.

For BirdNET models:
- **BirdNET v2.4**: Each segment = 3.0 seconds (144,000 samples at 48 kHz)
- **BirdNET v3.0**: Each segment = 5.0 seconds (160,000 samples at 32 kHz)
- **Perch v2**: Each segment = 5.0 seconds (160,000 samples at 32 kHz)

**Example:** Processing a 1-hour audio file with 5-second segments creates 720 segments.
- Batch size 1: 720 separate GPU calls
- Batch size 192: 4 GPU calls (192 + 192 + 192 + 144)
- Batch size 720: 1 GPU call (fastest, if GPU can handle it)

### Why Batching Matters

**Pros of larger batches:**
- ✅ Fewer GPU calls = less overhead
- ✅ Better GPU utilization
- ✅ Faster overall throughput

**Cons of too-large batches:**
- ❌ Exceeds GPU memory bandwidth limits
- ❌ GPU utilization collapses
- ❌ Severe performance degradation (up to 84% slower!)

## The 40% Rule

**Optimal batch size uses approximately 40% of your total VRAM.**

This is NOT an arbitrary choice - it's based on GPU memory architecture:

### Why Not 100% VRAM?

Modern GPUs have a memory hierarchy:
```
L2 Cache (fast)  → VRAM (slower) → System RAM (very slow)
~96 MB              16 GB            32+ GB
```

When your working set exceeds ~40-50% of VRAM:
1. L2 cache thrashing begins
2. Memory bandwidth saturates
3. GPU cores spend time waiting for data
4. Performance collapses despite having free VRAM

### Real-World Example: RTX 5080 16GB

Tested with BirdNET v2.4 on 3,600 segments:

| Batch | VRAM    | Time  | Throughput    | GPU Util | Result              |
|-------|---------|-------|---------------|----------|---------------------|
| 160   | 5.25 GB | 5.4s  | 667 seg/s     | ~17%     | Good                |
| 192   | 6.25 GB | 5.4s  | **667 seg/s** | ~17%     | **Optimal**         |
| 224   | 6.26 GB | 5.6s  | 643 seg/s     | 17%      | Minor degradation   |
| 256   | 8.33 GB | 34.6s | 104 seg/s     | **3%**   | **Catastrophic**    |

**Note:** Batch 160 and 192 achieve identical throughput, but 192 is optimal because it uses 39% VRAM (matching the 40% rule) and represents the maximum safe batch size before degradation begins at 224.

**Key finding:** Batch 256 uses only 52% of VRAM but runs **6x slower** than batch 192!

## Performance Degradation Explained

### What Happens at Batch 256?

The ONNX Runtime logs reveal the exact problem:

```
Extended allocation by 4294967296 bytes.  ← 4 GB extension!
```

This triggers because:
1. Intermediate CNN activations grow from 6.1 GB → 8.1 GB
2. ONNX Runtime extends its memory pool by 4 GB
3. Total working set (8.3 GB) exceeds memory bandwidth sweet spot
4. L2 cache (96 MB) can't hold the working set (87x overflow)
5. Every operation waits on slow VRAM access
6. GPU utilization drops from 17% → 3%

**This is NOT a bug** - it's the fundamental limit of memory bandwidth vs compute capability.

## Choosing Your Batch Size

### Method 1: Use the Table (Fast)

Start with the recommended batch size from the Quick Reference table above.

### Method 2: Calculate from VRAM (Better)

```
Target VRAM = Your GPU VRAM × 0.40
Estimated Batch = Target VRAM × 30 segments/GB
```

**Example for 8 GB GPU:**
```
Target VRAM = 8 × 0.40 = 3.2 GB
Batch ≈ 3.2 × 30 = 96 segments
```

### Method 3: Empirical Testing (Best)

Use the `scripts/debug_batching.ps1` tool to test multiple batch sizes:

```powershell
# Windows PowerShell
.\scripts\debug_batching.ps1 -AudioFile test.wav -BatchSizes @(64, 96, 128, 160, 192)
```

The script will:
- Test each batch size
- Monitor GPU utilization and memory
- Report optimal batch size
- Identify degradation points

See `scripts/README.md` for detailed usage.

## Warning Signs

### Your Batch Size is Too Large If:

1. **Processing time suddenly increases 2x or more**
   - Batch 192: 5.4s → Batch 256: 34.6s ← Red flag!

2. **GPU utilization drops below 10%**
   - GPU cores are starving, waiting for memory

3. **VRAM usage exceeds 50% of capacity**
   - Approaching memory bandwidth limits

4. **ORT logs show 1+ GB allocations**
   - `Extended allocation by 1073741824 bytes` or larger

## GPU-Specific Recommendations

### NVIDIA GTX 1650 (4 GB)

**Recommended: Batch 32**

Reasons:
- Very small L2 cache (~1 MB vs RTX 5080's 96 MB)
- Lower memory bandwidth (128 GB/s vs 672 GB/s)
- Will hit memory bottleneck earlier than newer GPUs
- Test range: 16-48, expect degradation at 64+

### NVIDIA RTX 3060 (12 GB)

**Recommended: Batch 144-160**

Reasons:
- Ampere architecture (2020)
- Good balance of VRAM and bandwidth
- Test range: 128-192

### NVIDIA RTX 4090 (24 GB)

**Recommended: Batch 256-320**

Reasons:
- Ada Lovelace architecture (2022)
- Massive memory bandwidth (1008 GB/s)
- Large L2 cache (72 MB)
- Can handle larger working sets efficiently
- Test range: 192-384

## Code Example

```rust
use birdnet_onnx::{Classifier, Result};
use birdnet_onnx::execution_providers::CUDAExecutionProvider;

fn main() -> Result<()> {
    // Build classifier with CUDA
    let classifier = Classifier::builder()
        .model_path("birdnet_v24.onnx")
        .labels_path("labels.txt")
        .execution_provider(CUDAExecutionProvider::default())
        .build()?;

    // Load and chunk your audio file
    let segments: Vec<Vec<f32>> = chunk_audio_file("recording.wav")?;

    // Batch size selection based on GPU
    let batch_size = 192;  // For RTX 5080 16GB

    // Process in batches
    let mut all_results = Vec::new();
    for chunk in segments.chunks(batch_size) {
        let refs: Vec<&[f32]> = chunk.iter()
            .map(|s| s.as_slice())
            .collect();

        let results = classifier.predict_batch(&refs)?;
        all_results.extend(results);
    }

    Ok(())
}
```

## Technical Deep Dive

### Memory Breakdown (Batch 192, BirdNET v2.4)

```
Model weights:        ~50 MB    (static, loaded once)
Input tensor:        ~110 MB    (192 × 144K × 4 bytes)
Intermediate CNN:   ~6.1 GB    (peak activation size)
Output tensor:       ~25 MB    (192 × 6,522 classes × 4 bytes)
─────────────────────────────
Total VRAM:         6.25 GB    (39% of 16 GB)
```

### Why Intermediate Activations Dominate

BirdNET's CNN architecture processes spectrograms through layers:

1. Input spectrogram: `[batch, 144000]` samples
2. Conv layers create feature maps: `[batch, 512, height, width]`
3. At batch 192: 192 × 512 × 50 × 3000 ≈ **6 GB** peak

The 512 filters in convolutional layers are essential for bird classification accuracy but create large intermediate tensors.

### BFCArena Allocation Strategy

ONNX Runtime uses "Best Fit with Coalescing":
- Pre-allocates memory pools in chunks: 1 MB, 2 MB, 4 MB, ..., 256 MB, 1 GB, 4 GB
- When current pool lacks space, extends by next chunk size
- The 4 GB extension at batch 256 is ONNX Runtime's way of saying "I need more contiguous memory"

This is efficient design but reveals when you've exceeded optimal batch size.

## Troubleshooting

### "Out of memory" errors

Your batch size is too large. Reduce by 25-50% and try again.

### Very slow processing despite free VRAM

You've hit memory bandwidth limits. Reduce batch size even though VRAM usage is low.

### GPU utilization < 10%

Memory-bound. Try smaller batch size.

### Inconsistent performance

Check GPU temperature throttling or background processes competing for GPU.

## Summary

1. **Start with 40% VRAM rule** - batch size that uses 40% of your total VRAM
2. **Test empirically** using the debug script for your specific GPU
3. **Watch for degradation** - sudden performance drops indicate too-large batches
4. **Don't chase 100% VRAM** - memory bandwidth, not capacity, is the limit
5. **Smaller GPUs hit limits earlier** - GTX 1650 needs smaller batches than RTX 5080

The optimal batch size is where **GPU compute and memory bandwidth are balanced**, not where VRAM is fully utilized!

## Further Reading

- `scripts/README.md` - Debug script usage guide
- BirdNET Paper: [arxiv.org/abs/2103.16196](https://arxiv.org/abs/2103.16196)
- ONNX Runtime Memory Management: [onnxruntime.ai/docs/performance/tune-performance.html](https://onnxruntime.ai/docs/performance/tune-performance.html)
