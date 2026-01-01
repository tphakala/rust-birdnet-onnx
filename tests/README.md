# Integration Tests

These tests require actual BirdNET/Perch ONNX model files.

## Setup

1. Download the models and labels to `tests/fixtures/`:

```
tests/fixtures/
├── birdnet_v24.onnx
├── birdnet_v24_labels.txt
├── birdnet_v30.onnx
├── birdnet_v30_labels.csv
├── perch_v2.onnx
└── perch_v2_labels.json
```

2. Run ignored tests:

```bash
cargo test -- --ignored
```

## Running Specific Tests

```bash
# Run only BirdNET v2.4 tests
cargo test birdnet_v24 -- --ignored

# Run only batch tests
cargo test batch -- --ignored
```

## Test Audio

Tests use synthetic audio (silence or sine waves). For real-world testing,
place WAV files in `tests/fixtures/audio/` and modify tests accordingly.
