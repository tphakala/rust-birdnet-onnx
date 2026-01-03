# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2026-01-04

### Changed

- **BREAKING (behavioral)**: `with_tensorrt()` now enables performance optimizations by default
  - Enables FP16 precision, CUDA graphs, engine caching, and timing cache
  - Expected 4x performance improvement vs previous zero-config behavior
  - Fixes [#18](https://github.com/tphakala/rust-birdnet-onnx/issues/18)
  - To disable optimizations, use `TensorRTConfig` to customize settings

### Added

- `TensorRTConfig` for fine-grained TensorRT configuration
- `ClassifierBuilder::with_tensorrt_config()` for custom TensorRT settings

## [1.4.0] - 2026-01-03

### Added

- `ExecutionProviderInfo` enum for execution provider types
- `available_execution_providers()` to query compile-time availability
- `Classifier::requested_provider()` to query requested execution provider
- Typed builder methods: `with_cuda()`, `with_tensorrt()`, etc.

### Changed

- `Classifier::builder()` now tracks requested execution provider

### Breaking Changes

- The `ort::execution_providers` re-export has been renamed to `ort_execution_providers`
  - **Migration**: Change `use birdnet_onnx::execution_providers` to `use birdnet_onnx::ort_execution_providers`
  - **Reason**: Prevents naming conflict with the new typed builder methods and clarifies it's the raw ort module

### Notes

- `requested_provider()` returns the *requested* provider, not the active one
- Use `ORT_LOG_LEVEL=Verbose` to verify actual runtime provider usage

## [1.3.0] - 2025-01-03

### Added
- Auto-detection and support for Perch v2 models
- GPU batch size optimization guide and debugging tools

### Fixed
- Use random delimiter for GitHub Actions multiline output

## [1.2.0] - Previous releases

### Added
- Initial release with BirdNET ONNX model support
- Basic classification functionality
- Audio processing capabilities
