# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **BSG Finland model support** ([BSG](https://github.com/luomus/BSG))
  - New `ModelType::BsgFinland` for the fused BirdNET v2.4 + Finnish classification head (265 species)
  - Pre-sigmoided output handling (no double-sigmoid)
  - `BsgPostProcessor` with per-species logistic calibration (Platt scaling)
  - Species Distribution Model (SDM) adjustment using migration curves and geographic distribution maps
  - CLI flags: `--calibration`, `--migration`, `--distribution-maps`, `--lat`, `--lon`, `--day-of-year`, `--csv`

### Fixed

- **`Perch` v2 output roles are resolved by name rather than by position.** The
  class scores were read from output index 3 and the embeddings from index 0,
  hardcoded at all three detection sites. That is correct for every published
  `Perch` export, but nothing verified it, so a re-export that reordered its
  outputs was read against the wrong tensors. Moving the class scores off index
  3 is caught by the existing label-count check, but swapping `embedding` with
  `spatial_embedding` is not, because both are 1536 wide: the 4-D spatial
  tensor is then returned as the pooled embedding with no error at all. The
  class-score output is now matched on `label` and the embeddings on
  `embedding`/`embeddings`, matched exactly so that `spatial_embedding` cannot
  stand in for `embedding`. A name that identifies a role wins; the published
  3/0 layout fills only the role the names leave open. An export whose named
  role lands on the index the layout reserves for the other one is reported as
  a detection error rather than resolved to a guess. That error is not bypassed
  by `--model perch`, which selects the model family and not the output layout.
  No published model is affected: every published `Perch` export names both
  roles, and an export with no usable names still gets the 3/0 layout.
- **A fresh install of the crate did not compile.** The `ort` requirement was
  written as `"2.0.0-rc.11"`, which cargo reads as a caret range, so any new
  dependant resolved `ort` to the newest release candidate. rc.13 relocated the
  `ort::ep::*` execution provider types and the build failed on unresolved
  imports. `ort` is now pinned to `=2.0.0-rc.12`, the only release candidate
  this crate compiles against (rc.11 does not export `IoBinding`).
- The README installation snippets asked for `birdnet-onnx = "2.0"`, which
  cargo never matches because every 2.0.0 release so far is a pre-release. Both
  snippets now name the version in full.

## [2.0.0-rc.1] - 2026-01-04

### Changed

- **BREAKING**: `TensorRTConfig` now disables CUDA graphs by default
  - CUDA graphs are disabled to avoid ONNX Runtime 1.22.0 bug #20050
  - Bug causes panic on batch 2+ with "expected 'typeinfo_ptr' to not be null"
  - Fixes [#26](https://github.com/tphakala/rust-birdnet-onnx/issues/26)
  - **Migration**: To enable CUDA graphs (if your workload doesn't trigger the bug):

    ```rust
    let config = TensorRTConfig::new().with_cuda_graph(true);
    let classifier = Classifier::builder()
        .with_tensorrt_config(config)
        .build()?;
    ```

  - **Impact**: Minimal performance difference in practice (batches 2+ still fast)
  - See upstream: [ONNX Runtime Issue #20050](https://github.com/microsoft/onnxruntime/issues/20050)

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
