# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ExecutionProviderInfo` enum to represent different hardware backends (CPU, GPU, NPU, etc.)
- `available_execution_providers()` function to query which providers are compiled in
- `Classifier::execution_provider()` method to query which provider is actually being used
- Typed builder methods: `with_cuda()`, `with_tensorrt()`, `with_directml()`, `with_coreml()`
- Runtime detection of execution provider fallback using tracing events
- Accurate reporting of active backend even when requested provider is unavailable

### Changed
- `Classifier::builder()` now tracks and detects which execution provider is active
- Debug output for `Classifier` now includes execution provider information

### Fixed
- Applications can now accurately report which device (CPU/GPU/etc.) is running inference
- Fixes misleading "GPU" messages when CUDA is unavailable and falls back to CPU
