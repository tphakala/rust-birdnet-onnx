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
birdnet-onnx = "1.0"
```

## Library Usage

```rust
use birdnet_onnx::{Classifier, Result};

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
    let result = classifier.predict(&audio)?;

    for pred in &result.predictions {
        println!("{}: {:.1}%", pred.species, pred.confidence * 100.0);
    }

    Ok(())
}
```

### With CUDA GPU

```rust
use birdnet_onnx::{Classifier, execution_providers::CUDAExecutionProvider};

let classifier = Classifier::builder()
    .model_path("model.onnx")
    .labels_path("labels.txt")
    .execution_provider(CUDAExecutionProvider::default())
    .build()?;
```

### Batch Inference

```rust
let segments: Vec<Vec<f32>> = chunk_audio_file();
let refs: Vec<&[f32]> = segments.iter().map(|s| s.as_slice()).collect();

let results = classifier.predict_batch(&refs)?;
```

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
    -o 1.5 \          # 1.5s overlap between segments
    -k 5 \            # Top 5 predictions
    --min-confidence 0.2
```

Example output:

```
Analyzing: recording.wav (3m 21s, 48000 Hz)
Model: BirdNET v2.4 (3.0s segments, 1.5s overlap)

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
