#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Debug batching performance for BirdNET inference

.DESCRIPTION
    Tests multiple batch sizes, monitors GPU usage, and analyzes ONNX Runtime memory allocations
    to find optimal batch size for your GPU.

.PARAMETER BirdaPath
    Path to birda.exe (default: current directory)

.PARAMETER AudioFile
    Path to audio file to test with

.PARAMETER BatchSizes
    Array of batch sizes to test (default: 64, 96, 128, 160, 192, 224, 256, 320, 384)

.PARAMETER Confidence
    Confidence threshold (default: 0.8)

.PARAMETER MonitorInterval
    GPU monitoring interval in milliseconds (default: 200)

.PARAMETER EnableProfiling
    Enable ONNX Runtime profiling (generates .json files)

.EXAMPLE
    .\debug_batching.ps1 -AudioFile recording.wav

.EXAMPLE
    .\debug_batching.ps1 -AudioFile recording.wav -BatchSizes @(128, 192, 256) -EnableProfiling
#>

param(
    [string]$BirdaPath = ".\birda.exe",
    [Parameter(Mandatory=$true)]
    [string]$AudioFile,
    [int[]]$BatchSizes = @(64, 96, 128, 160, 192, 224, 256, 320, 384),
    [float]$Confidence = 0.8,
    [int]$MonitorInterval = 200,
    [switch]$EnableProfiling
)

# Color output helpers
function Write-Header($text) {
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

function Write-Success($text) {
    Write-Host $text -ForegroundColor Green
}

function Write-Warning2($text) {
    Write-Host $text -ForegroundColor Yellow
}

function Write-Info($text) {
    Write-Host $text -ForegroundColor White
}

# Check prerequisites
Write-Header "Batch Performance Debugger"

if (-not (Test-Path $BirdaPath)) {
    Write-Error "birda.exe not found at: $BirdaPath"
    exit 1
}

if (-not (Test-Path $AudioFile)) {
    Write-Error "Audio file not found: $AudioFile"
    exit 1
}

# Check for nvidia-smi
try {
    $null = nvidia-smi --version
} catch {
    Write-Error "nvidia-smi not found. Is NVIDIA GPU driver installed?"
    exit 1
}

# Get GPU info
Write-Info "Detecting GPU..."
$gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
Write-Success "GPU: $gpuInfo"

# Create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "batch_debug_$timestamp"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Write-Info "Output directory: $outputDir"

# Results array
$results = @()

# Test each batch size
foreach ($batch in $BatchSizes) {
    Write-Header "Testing Batch Size: $batch"

    $logFile = Join-Path $outputDir "batch_${batch}_log.txt"
    $gpuFile = Join-Path $outputDir "batch_${batch}_gpu.csv"

    # Start GPU monitoring in background job
    Write-Info "Starting GPU monitor..."
    $gpuMonitorJob = Start-Job -ScriptBlock {
        param($interval, $outputFile)

        # Header (no special chars in column names)
        "Timestamp,GPU_Util,Memory_Util,Memory_Used_MB,Memory_Free_MB,Temp_C,Power_W" | Out-File -FilePath $outputFile

        while ($true) {
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
            $metrics = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits

            if ($metrics) {
                "$timestamp,$metrics" | Out-File -FilePath $outputFile -Append
            }

            Start-Sleep -Milliseconds $interval
        }
    } -ArgumentList $MonitorInterval, $gpuFile

    Start-Sleep -Milliseconds 500  # Let monitor start

    # Set environment variables for verbose logging
    $env:ORT_LOG_SEVERITY_LEVEL = "1"  # Info level
    if ($EnableProfiling) {
        $env:ORT_ENABLE_PROFILING = "1"
    }

    Write-Info "Running inference (batch=$batch, confidence=$Confidence)..."

    # Run birda and capture output
    $startTime = Get-Date

    try {
        $output = & $BirdaPath -vvv -b $batch -c $Confidence --force $AudioFile 2>&1
        $exitCode = $LASTEXITCODE

        $endTime = Get-Date
        $elapsed = ($endTime - $startTime).TotalSeconds

        # Save log
        $output | Out-File -FilePath $logFile

        # Stop GPU monitoring
        Stop-Job $gpuMonitorJob
        Remove-Job $gpuMonitorJob

        # Parse results
        $detections = 0
        $segments = 0
        [long]$totalVramAllocated = 0
        $largeAllocations = @()

        # Parse log for statistics
        foreach ($line in $output) {
            if ($line -match "Found (\d+) detections") {
                $detections = [int]$matches[1]
            }
            if ($line -match "Running inference on (\d+) segments") {
                $segments = [int]$matches[1]
            }
            if ($line -match "Total allocated bytes: (\d+)") {
                [long]$bytes = $matches[1]
                if ($bytes -gt $totalVramAllocated) {
                    $totalVramAllocated = $bytes
                }
            }
            if ($line -match "Extended allocation by (\d+) bytes") {
                [long]$bytes = $matches[1]
                if ($bytes -ge 1073741824) {  # >= 1GB
                    $largeAllocations += [Math]::Round($bytes / 1GB, 2)
                }
            }
        }

        # Parse GPU metrics
        if (Test-Path $gpuFile) {
            $gpuMetrics = Import-Csv $gpuFile
            $avgGpuUtil = ($gpuMetrics.GPU_Util | Measure-Object -Average).Average
            $maxGpuUtil = ($gpuMetrics.GPU_Util | Measure-Object -Maximum).Maximum
            $avgMemUtil = ($gpuMetrics.Memory_Util | Measure-Object -Average).Average
            $maxMemUtil = ($gpuMetrics.Memory_Util | Measure-Object -Maximum).Maximum
            $maxMemUsed = ($gpuMetrics.Memory_Used_MB | Measure-Object -Maximum).Maximum
            $avgPower = ($gpuMetrics.Power_W | Measure-Object -Average).Average
            $maxTemp = ($gpuMetrics.Temp_C | Measure-Object -Maximum).Maximum
        } else {
            $avgGpuUtil = 0
            $maxGpuUtil = 0
            $avgMemUtil = 0
            $maxMemUtil = 0
            $maxMemUsed = 0
            $avgPower = 0
            $maxTemp = 0
        }

        # Calculate throughput
        $throughput = if ($elapsed -gt 0) { [Math]::Round($segments / $elapsed, 2) } else { 0 }

        # Store results
        $result = [PSCustomObject]@{
            BatchSize = $batch
            TotalTime_s = [Math]::Round($elapsed, 2)
            Segments = $segments
            Throughput_seg_s = $throughput
            Detections = $detections
            TotalVRAM_GB = [Math]::Round($totalVramAllocated / 1GB, 2)
            LargeAlloc_GB = ($largeAllocations -join ", ")
            AvgGPU_Pct = [Math]::Round($avgGpuUtil, 1)
            MaxGPU_Pct = [Math]::Round($maxGpuUtil, 1)
            AvgMemUtil_Pct = [Math]::Round($avgMemUtil, 1)
            MaxMemUtil_Pct = [Math]::Round($maxMemUtil, 1)
            MaxMemUsed_MB = [Math]::Round($maxMemUsed, 0)
            AvgPower_W = [Math]::Round($avgPower, 1)
            MaxTemp_C = [Math]::Round($maxTemp, 0)
            ExitCode = $exitCode
        }

        $results += $result

        # Print summary
        Write-Success "✓ Completed in $($elapsed.ToString('F2'))s"
        Write-Info "  Throughput: $throughput segments/s"
        Write-Info "  VRAM Peak: $($result.TotalVRAM_GB) GB"
        Write-Info "  GPU Util: avg $($result.AvgGPU_Pct)% / max $($result.MaxGPU_Pct)%"
        Write-Info "  Mem Util: avg $($result.AvgMemUtil_Pct)% / max $($result.MaxMemUtil_Pct)%"
        if ($largeAllocations.Count -gt 0) {
            Write-Warning2 "  ⚠ Large allocations (>=1GB): $($largeAllocations -join 'GB, ')GB"
        }

    } catch {
        Write-Error "Failed to run batch size $batch`: $_"
        Stop-Job $gpuMonitorJob -ErrorAction SilentlyContinue
        Remove-Job $gpuMonitorJob -ErrorAction SilentlyContinue
    }

    # Cleanup environment
    Remove-Item Env:ORT_LOG_SEVERITY_LEVEL -ErrorAction SilentlyContinue
    Remove-Item Env:ORT_ENABLE_PROFILING -ErrorAction SilentlyContinue

    # Small delay between tests
    Start-Sleep -Seconds 2
}

# Generate report
Write-Header "Performance Analysis Report"

# Display results table
Write-Info "`nResults Summary:"
$results | Format-Table -AutoSize

# Export detailed results
$csvPath = Join-Path $outputDir "batch_results.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Success "Detailed results saved to: $csvPath"

# Find optimal batch size (highest throughput)
$optimal = $results | Sort-Object -Property Throughput_seg_s -Descending | Select-Object -First 1
Write-Header "Recommendations"
Write-Success "Optimal Batch Size: $($optimal.BatchSize)"
Write-Info "  Throughput: $($optimal.Throughput_seg_s) segments/s"
Write-Info "  VRAM Usage: $($optimal.TotalVRAM_GB) GB"
Write-Info "  GPU Utilization: avg $($optimal.AvgGPU_Pct)% / max $($optimal.MaxGPU_Pct)%"
Write-Info "  Memory Utilization: avg $($optimal.AvgMemUtil_Pct)% / max $($optimal.MaxMemUtil_Pct)%"

# Identify degradation point
$sortedByBatch = $results | Sort-Object -Property BatchSize
$degradationPoint = $null
for ($i = 0; $i -lt ($sortedByBatch.Count - 1); $i++) {
    $current = $sortedByBatch[$i]
    $next = $sortedByBatch[$i + 1]

    if ($next.Throughput_seg_s -lt $current.Throughput_seg_s * 0.95) {  # 5% slowdown
        $degradationPoint = $next.BatchSize
        $slowdown = [Math]::Round((($current.Throughput_seg_s - $next.Throughput_seg_s) / $current.Throughput_seg_s) * 100, 1)
        Write-Warning2 "`n⚠ Performance degradation detected at batch size $degradationPoint"
        Write-Info "  Slowdown: $slowdown% compared to batch $($current.BatchSize)"
        Write-Info "  VRAM increase: $($current.TotalVRAM_GB) GB → $($next.TotalVRAM_GB) GB (+$([Math]::Round($next.TotalVRAM_GB - $current.TotalVRAM_GB, 2)) GB)"
        break
    }
}

# Memory analysis
Write-Info "`nMemory Allocation Analysis:"
$highVram = $results | Where-Object { $_.TotalVRAM_GB -ge 8 } | Sort-Object -Property TotalVRAM_GB
if ($highVram) {
    Write-Warning2 "Batch sizes with high VRAM usage (>= 8 GB):"
    foreach ($r in $highVram) {
        Write-Info "  Batch $($r.BatchSize): $($r.TotalVRAM_GB) GB VRAM"
        if ($r.LargeAlloc_GB) {
            Write-Info "    Large allocations: $($r.LargeAlloc_GB) GB"
        }
    }
} else {
    Write-Success "No batch sizes exceeded 8 GB VRAM"
}

# GPU utilization analysis
$lowGpuUtil = $results | Where-Object { $_.AvgGPU_Pct -lt 70 }
if ($lowGpuUtil) {
    Write-Warning2 "`nBatch sizes with low GPU utilization (<70%):"
    foreach ($r in $lowGpuUtil) {
        Write-Info "  Batch $($r.BatchSize): avg $($r.AvgGPU_Pct)% GPU, $($r.AvgMemUtil_Pct)% Mem"
        if ($r.AvgMemUtil_Pct -gt 80) {
            Write-Warning2 "    → Memory-bound (high mem util, low GPU util)"
        }
    }
}

# Create performance chart (simple text-based)
Write-Info "`nThroughput Chart:"
$maxThroughput = ($results.Throughput_seg_s | Measure-Object -Maximum).Maximum
foreach ($r in ($results | Sort-Object -Property BatchSize)) {
    $barLength = [Math]::Round(($r.Throughput_seg_s / $maxThroughput) * 50)
    $bar = "█" * $barLength
    $padding = " " * (6 - $r.BatchSize.ToString().Length)
    Write-Host "  $($r.BatchSize):$padding $bar $($r.Throughput_seg_s) seg/s" -ForegroundColor $(if ($r.BatchSize -eq $optimal.BatchSize) { "Green" } else { "White" })
}

Write-Header "Debug Data Location"
Write-Info "All logs and metrics saved to: $outputDir"
Write-Info "  - batch_results.csv - Summary results"
Write-Info "  - batch_*_log.txt - Full ORT logs per batch size"
Write-Info "  - batch_*_gpu.csv - GPU metrics per batch size"
if ($EnableProfiling) {
    Write-Info "  - onnxruntime_profile_*.json - ORT profiling data (if generated)"
}

Write-Success "`nAnalysis complete!"
