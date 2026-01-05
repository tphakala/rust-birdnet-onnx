# birdnet-onnx

[![CI](https://github.com/tphakala/rust-birdnet-onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/tphakala/rust-birdnet-onnx/actions/workflows/ci.yml)
[![Security](https://github.com/tphakala/rust-birdnet-onnx/actions/workflows/security.yml/badge.svg)](https://github.com/tphakala/rust-birdnet-onnx/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/tphakala/rust-birdnet-onnx/graph/badge.svg)](https://codecov.io/gh/tphakala/rust-birdnet-onnx)
[![Crates.io](https://img.shields.io/crates/v/birdnet-onnx.svg)](https://crates.io/crates/birdnet-onnx)
[![docs.rs](https://docs.rs/birdnet-onnx/badge.svg)](https://docs.rs/birdnet-onnx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.92%2B-blue.svg)](https://www.rust-lang.org/)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub-pink.svg)](https://github.com/sponsors/tphakala)

A Rust library for running inference on BirdNET and Perch ONNX models with CUDA GPU support.

## Features

- Support for BirdNET v2.4, v3.0, and Perch v2 models
- Automatic model type detection from ONNX tensor shapes
- Thread-safe classifier with builder pattern
- Top-K predictions with configurable confidence threshold
- Batch inference for GPU efficiency
- CLI tool for WAV file analysis

## Supported Models

| Model | Sample Rate | Segment | Embeddings |
|-------|-------------|---------|------------|
| BirdNET v2.4 | 48 kHz | 3.0s | No |
| BirdNET v3.0 | 32 kHz | 5.0s | 1024-dim |
| Perch v2 | 32 kHz | 5.0s | Variable |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
birdnet-onnx = "2.0"
```

## Library Usage

```rust
use birdnet_onnx::{Classifier, InferenceOptions, Result};

fn main() -> Result<()> {
    // Build classifier
    let classifier = Classifier::builder()
        .model_path("birdnet_v24.onnx")
        .labels_path("labels.txt")
        .top_k(5)
        .min_confidence(0.1)
        .build()?;

    // Prepare audio segment (48kHz, 3.0s = 144,000 samples for v2.4)
    let audio: Vec<f32> = load_audio_segment();

    // Run inference
    let result = classifier.predict(&audio, &InferenceOptions::default())?;

    for pred in &result.predictions {
        println!("{}: {:.1}%", pred.species, pred.confidence * 100.0);
    }

    Ok(())
}
```

### With CUDA GPU

```rust
use birdnet_onnx::Classifier;

let classifier = Classifier::builder()
    .model_path("model.onnx")
    .labels_path("labels.txt")
    .with_cuda()  // Uses safe defaults for memory allocation
    .build()?;
```

For fine-grained control over CUDA memory allocation:

```rust
use birdnet_onnx::{Classifier, CUDAConfig, ArenaExtendStrategy};

let classifier = Classifier::builder()
    .model_path("model.onnx")
    .labels_path("labels.txt")
    .with_cuda_config(
        CUDAConfig::new()
            .with_memory_limit(4 * 1024 * 1024 * 1024)  // 4GB limit
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
    )
    .build()?;
```

### Batch Inference

```rust
use birdnet_onnx::InferenceOptions;

let segments: Vec<Vec<f32>> = chunk_audio_file();
let refs: Vec<&[f32]> = segments.iter().map(|s| s.as_slice()).collect();

let results = classifier.predict_batch(&refs, &InferenceOptions::default())?;
```

### GPU Memory-Efficient Batch Processing

For processing many batches on GPU, use `BatchInferenceContext` to prevent memory growth:

```rust
use birdnet_onnx::{Classifier, InferenceOptions};

let classifier = Classifier::builder()
    .model_path("model.onnx")
    .labels_path("labels.txt")
    .with_cuda()
    .build()?;

// Create context with pre-allocated buffers (max 32 segments per batch)
let mut ctx = classifier.create_batch_context(32)?;

// Process multiple batches - memory is reused across calls
for chunk in audio_segments.chunks(32) {
    let refs: Vec<&[f32]> = chunk.iter().map(|s| s.as_slice()).collect();
    let results = classifier.predict_batch_with_context(
        &mut ctx,
        &refs,
        &InferenceOptions::default(),
    )?;
}
```

### Timeout and Cancellation

```rust
use birdnet_onnx::{InferenceOptions, CancellationToken};
use std::time::Duration;

// With timeout
let options = InferenceOptions::timeout(Duration::from_secs(30));
let result = classifier.predict(&audio, &options)?;

// With cancellation token (for graceful shutdown)
let token = CancellationToken::new();
let options = InferenceOptions::new().with_cancellation_token(token.clone());

// Cancel from another thread
token.cancel();
```

## Execution Provider Query

Query which execution provider was requested:

```rust
let classifier = Classifier::builder()
    .model_path("model.onnx")
    .labels_path("labels.txt")
    .with_cuda()
    .build()?;

// Returns the requested provider (not necessarily the active one)
println!("Requested: {}", classifier.requested_provider().as_str());
```

**Note:** This returns the *requested* execution provider. If the requested
provider is unavailable, ONNX Runtime silently falls back to CPU. To verify
which provider is actually running:

1. Enable verbose logging: `export ORT_LOG_LEVEL=Verbose`
2. Check log output for "Using [provider]" messages

## CLI Usage

A basic CLI tool is included for quick testing of the library. It is not intended for production analysis tasks.

Build:

```bash
cargo build --release --bin birdnet-analyze
```

Analyze a WAV file:

```bash
birdnet-analyze recording.wav -m birdnet_v24.onnx -l labels.txt
```

With options:

```bash
birdnet-analyze recording.wav \
    -m birdnet_v24.onnx \
    -l labels.txt \
    -o 1.5 \              # 1.5s overlap between segments
    -k 5 \                # Top 5 predictions
    --min-confidence 0.2 \
    --batch-size 32 \     # Segments per batch
    --timeout 30 \        # Per-batch timeout in seconds
    -v                    # Verbose output (shows timing, memory usage)
```

With GPU acceleration:

```bash
birdnet-analyze recording.wav -m model.onnx -l labels.txt --cuda
birdnet-analyze recording.wav -m model.onnx -l labels.txt --tensorrt
```

Example output:

```
Analyzing: recording.wav (3m 21s, 48000 Hz)
Model: BirdNET v2.4 (3.0s segments, 1.5s overlap)
Provider: CUDA

00:00.0  Eurasian Pygmy-Owl (92.4%)
00:01.5  Eurasian Pygmy-Owl (97.8%)
00:03.0  Eurasian Pygmy-Owl (98.5%)
...

134 segments analyzed in 1.2s
```

## Range Filter (Meta Model)

Filter species predictions by location and date using BirdNET's meta model:

```rust
use birdnet_onnx::RangeFilter;

// Load the meta model
let range_filter = RangeFilter::builder()
    .model_path("birdnet_data_model.onnx")
    .labels(labels)
    .threshold(0.01)
    .build()?;

// Get species likely at location/date
// Helsinki, Finland on June 15th
let scores = range_filter.predict(60.1695, 24.9354, 6, 15)?;

println!("Expected {} species", scores.len());
for score in scores.iter().take(10) {
    println!("{}: {:.1}%", score.species, score.score * 100.0);
}
```

The meta model uses BirdNET's 48-week calendar (4 weeks per month).

### Using RangeFilter for Location-Based Filtering

```rust
use birdnet_onnx::{Classifier, RangeFilter};

// Build classifier
let classifier = Classifier::builder()
    .model_path("birdnet.onnx")
    .labels_path("labels.txt")
    .build()?;

// Build range filter using classifier labels
let range_filter = RangeFilter::builder()
    .model_path("birdnet_data_model.onnx")
    .from_classifier_labels(classifier.labels())
    .threshold(0.01)
    .build()?;

// Get predictions
let result = classifier.predict(&audio_segment)?;

// Filter by location (Helsinki, June 15th)
let location_scores = range_filter.predict(60.1695, 24.9354, 6, 15)?;
let filtered = range_filter.filter_predictions(
    &result.predictions,
    &location_scores,
    false,
);

for pred in filtered {
    println!("{}: {:.1}%", pred.species, pred.confidence * 100.0);
}
```

**Batch processing multiple files:**

```rust
// Calculate location scores once
let location_scores = range_filter.predict(lat, lon, month, day)?;

// Process multiple audio segments from same location
let mut predictions_batch = Vec::new();
for segment in audio_segments {
    let result = classifier.predict(&segment)?;
    predictions_batch.push(result.predictions);
}

// Filter all predictions at once
let filtered_batch = range_filter.filter_batch_predictions(
    predictions_batch,
    &location_scores,
    true, // rerank: multiply confidence by location score
);
```

## Development

Requires [Task](https://taskfile.dev/) runner:

```bash
task --list        # Show available tasks
task build         # Build in debug mode
task build:release # Build in release mode
task test          # Run unit tests
task lint          # Run clippy
task ci            # Run all CI checks
```

## Acknowledgments

This library provides Rust bindings for running inference on models from these projects:

- [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) - Bird sound identification by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology and Chemnitz University of Technology
- [Perch](https://github.com/google-research/perch) - Bioacoustics research by Google Research

Built with:

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Cross-platform inference engine by Microsoft
- [ort](https://github.com/pykeio/ort) - Rust bindings for ONNX Runtime

## License

MIT
