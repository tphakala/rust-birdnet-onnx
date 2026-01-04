//! `TensorRT` execution provider configuration
//!
//! This module provides fine-grained control over `TensorRT` optimization settings.
//! For most users, the default [`crate::ClassifierBuilder::with_tensorrt()`] method provides
//! optimal performance. Use [`TensorRTConfig`] when you need custom settings.
//!
//! # Performance Notes
//!
//! The default configuration enables:
//! - **FP16 precision**: 2x faster inference on GPUs with tensor cores
//! - **Engine caching**: Reduces session creation from minutes to seconds
//! - **Timing cache**: Accelerates future builds with similar layer configurations
//!
//! **CUDA graphs are disabled by default** to avoid a known bug in ONNX Runtime 1.22.0.
//! You can enable them explicitly with `.with_cuda_graph(true)` if your workload doesn't
//! trigger the bug. See [`TensorRTConfig::new()`] for details
//!
//! # Example
//!
//! ```no_run
//! use birdnet_onnx::{Classifier, TensorRTConfig};
//!
//! let config = TensorRTConfig::new()
//!     .with_fp16(false)
//!     .with_builder_optimization_level(5)
//!     .with_engine_cache_path("/tmp/trt_cache");
//!
//! let classifier = Classifier::builder()
//!     .model_path("model.onnx")
//!     .labels_path("labels.txt")
//!     .with_tensorrt_config(config)
//!     .build()?;
//! # Ok::<(), birdnet_onnx::Error>(())
//! ```

/// Configuration for `TensorRT` execution provider
///
/// This struct provides fine-grained control over `TensorRT` optimization settings.
/// For most users, the default [`crate::ClassifierBuilder::with_tensorrt()`] method provides
/// optimal performance.
///
/// # Performance Notes
///
/// The default configuration enables:
/// - **FP16 precision**: 2x faster inference on GPUs with tensor cores
/// - **Engine caching**: Reduces session creation from minutes to seconds
/// - **Timing cache**: Accelerates future builds with similar layer configurations
///
/// **CUDA graphs are disabled by default** to avoid a known bug in ONNX Runtime 1.22.0.
/// See [`TensorRTConfig::new()`] for details and how to enable them if needed
///
/// # Example: Custom Configuration
///
/// ```no_run
/// use birdnet_onnx::TensorRTConfig;
///
/// let config = TensorRTConfig::new()
///     .with_fp16(false)
///     .with_builder_optimization_level(5)
///     .with_engine_cache_path("/tmp/trt_cache")
///     .with_device_id(1);  // Use second GPU
/// # let _ = config;
/// ```
///
/// # Example: Disable Optimizations
///
/// ```no_run
/// use birdnet_onnx::TensorRTConfig;
///
/// let config = TensorRTConfig::new()
///     .with_fp16(false)
///     .with_cuda_graph(false)
///     .with_engine_cache(false)
///     .with_timing_cache(false);
/// # let _ = config;
/// ```
#[derive(Debug, Clone)]
pub struct TensorRTConfig {
    // Performance options
    fp16: Option<bool>,
    int8: Option<bool>,
    cuda_graph: Option<bool>,
    builder_optimization_level: Option<u8>,

    // Caching options
    engine_cache: Option<bool>,
    engine_cache_path: Option<String>,
    timing_cache: Option<bool>,
    timing_cache_path: Option<String>,

    // Hardware options
    device_id: Option<i32>,
    max_workspace_size: Option<usize>,

    // Advanced options
    min_subgraph_size: Option<usize>,
    layer_norm_fp32_fallback: Option<bool>,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            // Performance defaults (same as with_tensorrt())
            fp16: Some(true),
            cuda_graph: Some(false), // Disabled by default due to ONNX Runtime bug #20050
            engine_cache: Some(true),
            timing_cache: Some(true),
            builder_optimization_level: Some(3),

            // Everything else None (uses TensorRT defaults)
            int8: None,
            engine_cache_path: None,
            timing_cache_path: None,
            device_id: None,
            max_workspace_size: None,
            min_subgraph_size: None,
            layer_norm_fp32_fallback: None,
        }
    }
}

impl TensorRTConfig {
    /// Create a new `TensorRT` configuration with optimized defaults
    ///
    /// # CUDA Graphs
    ///
    /// CUDA graphs are **disabled by default** to avoid a known bug in ONNX Runtime 1.22.0
    /// where graph replay fails on batch 2+ with "expected 'typeinfo_ptr' to not be null".
    /// See [ONNX Runtime Issue #20050](https://github.com/microsoft/onnxruntime/issues/20050).
    ///
    /// If you need CUDA graphs for performance and your workload doesn't trigger the bug,
    /// you can enable them explicitly:
    ///
    /// ```no_run
    /// use birdnet_onnx::TensorRTConfig;
    ///
    /// let config = TensorRTConfig::new()
    ///     .with_cuda_graph(true);  // Opt-in to CUDA graphs
    /// # let _ = config;
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            // Performance defaults
            fp16: Some(true),
            cuda_graph: Some(false), // Disabled by default due to ONNX Runtime bug #20050
            engine_cache: Some(true),
            timing_cache: Some(true),
            builder_optimization_level: Some(3),

            // Everything else None
            int8: None,
            engine_cache_path: None,
            timing_cache_path: None,
            device_id: None,
            max_workspace_size: None,
            min_subgraph_size: None,
            layer_norm_fp32_fallback: None,
        }
    }

    /// Enable or disable FP16 precision mode
    ///
    /// FP16 provides ~2x speedup on GPUs with tensor cores (Volta and newer).
    /// Disable if you need full FP32 precision for accuracy-critical applications.
    ///
    /// Default: `true`
    #[must_use]
    pub const fn with_fp16(mut self, enable: bool) -> Self {
        self.fp16 = Some(enable);
        self
    }

    /// Enable or disable INT8 precision mode
    ///
    /// Requires calibration data. Provides additional speedup over FP16.
    /// See `TensorRT` documentation for calibration requirements.
    ///
    /// Default: `false` (not enabled by default)
    #[must_use]
    pub const fn with_int8(mut self, enable: bool) -> Self {
        self.int8 = Some(enable);
        self
    }

    /// Enable or disable CUDA graph capture
    ///
    /// Reduces CPU launch overhead for models with many small layers.
    /// Provides significant speedup by batching GPU operations.
    ///
    /// **Default: `false`** (disabled due to ONNX Runtime bug #20050)
    ///
    /// # Warning
    ///
    /// ONNX Runtime 1.22.0 has a bug where CUDA graph replay fails on batch 2+
    /// with "expected 'typeinfo_ptr' to not be null". Only enable if you've
    /// verified your workload doesn't trigger this bug.
    ///
    /// See: <https://github.com/microsoft/onnxruntime/issues/20050>
    #[must_use]
    pub const fn with_cuda_graph(mut self, enable: bool) -> Self {
        self.cuda_graph = Some(enable);
        self
    }

    /// Set builder optimization level (0-5)
    ///
    /// Higher values take longer to build but may produce faster engines.
    /// - Level 3 (default): Balanced optimization
    /// - Level 5: Maximum optimization (longer build time)
    /// - Level 0-2: Faster builds, may sacrifice performance
    ///
    /// Default: `3`
    #[must_use]
    pub const fn with_builder_optimization_level(mut self, level: u8) -> Self {
        self.builder_optimization_level = Some(level);
        self
    }

    /// Enable or disable engine caching
    ///
    /// Caches compiled `TensorRT` engines to disk, dramatically reducing
    /// session creation time on subsequent runs (384s → 9s in benchmarks).
    ///
    /// **Important**: Clear cache when model, ONNX Runtime, or `TensorRT` version changes.
    ///
    /// Default: `true`
    #[must_use]
    pub const fn with_engine_cache(mut self, enable: bool) -> Self {
        self.engine_cache = Some(enable);
        self
    }

    /// Set custom path for engine cache
    ///
    /// By default, `TensorRT` uses system temp directory.
    /// Set a custom path for persistent caching across system restarts.
    ///
    /// Default: None (uses `TensorRT` default)
    #[must_use]
    pub fn with_engine_cache_path(mut self, path: impl Into<String>) -> Self {
        self.engine_cache_path = Some(path.into());
        self
    }

    /// Enable or disable timing cache
    ///
    /// Stores kernel timing data to accelerate future builds with similar
    /// layer configurations (34.6s → 7.7s in benchmarks).
    ///
    /// Default: `true`
    #[must_use]
    pub const fn with_timing_cache(mut self, enable: bool) -> Self {
        self.timing_cache = Some(enable);
        self
    }

    /// Set custom path for timing cache
    ///
    /// By default, `TensorRT` uses system temp directory.
    ///
    /// Default: None (uses `TensorRT` default)
    #[must_use]
    pub fn with_timing_cache_path(mut self, path: impl Into<String>) -> Self {
        self.timing_cache_path = Some(path.into());
        self
    }

    /// Set GPU device ID for multi-GPU systems
    ///
    /// Default: None (uses default GPU)
    #[must_use]
    pub const fn with_device_id(mut self, device_id: i32) -> Self {
        self.device_id = Some(device_id);
        self
    }

    /// Set maximum workspace size in bytes
    ///
    /// `TensorRT` may allocate up to this much GPU memory for optimization.
    /// Larger values may enable more optimizations but use more memory.
    ///
    /// Default: None (uses `TensorRT` default)
    #[must_use]
    pub const fn with_max_workspace_size(mut self, max_size: usize) -> Self {
        self.max_workspace_size = Some(max_size);
        self
    }

    /// Set minimum subgraph size for `TensorRT` acceleration
    ///
    /// Subgraphs smaller than this will not be accelerated by `TensorRT`.
    ///
    /// Default: None (uses `TensorRT` default)
    #[must_use]
    pub const fn with_min_subgraph_size(mut self, min_size: usize) -> Self {
        self.min_subgraph_size = Some(min_size);
        self
    }

    /// Enable or disable FP32 fallback for layer normalization
    ///
    /// When enabled, layer norm operations use FP32 even in FP16 mode,
    /// improving accuracy at slight performance cost.
    ///
    /// Default: None (uses `TensorRT` default)
    #[must_use]
    pub const fn with_layer_norm_fp32_fallback(mut self, enable: bool) -> Self {
        self.layer_norm_fp32_fallback = Some(enable);
        self
    }

    /// Apply configuration to a `TensorRT` execution provider
    ///
    /// This is an internal method used by `ClassifierBuilder::with_tensorrt_config()`.
    pub(crate) fn apply_to(
        self,
        provider: ort::execution_providers::TensorRTExecutionProvider,
    ) -> ort::execution_providers::TensorRTExecutionProvider {
        let mut p = provider;

        if let Some(v) = self.fp16 {
            p = p.with_fp16(v);
        }
        if let Some(v) = self.int8 {
            p = p.with_int8(v);
        }
        if let Some(v) = self.cuda_graph {
            p = p.with_cuda_graph(v);
        }
        if let Some(v) = self.builder_optimization_level {
            p = p.with_builder_optimization_level(v);
        }
        if let Some(v) = self.engine_cache {
            p = p.with_engine_cache(v);
        }
        if let Some(path) = self.engine_cache_path {
            p = p.with_engine_cache_path(path);
        }
        if let Some(v) = self.timing_cache {
            p = p.with_timing_cache(v);
        }
        if let Some(path) = self.timing_cache_path {
            p = p.with_timing_cache_path(path);
        }
        if let Some(id) = self.device_id {
            p = p.with_device_id(id);
        }
        if let Some(size) = self.max_workspace_size {
            p = p.with_max_workspace_size(size);
        }
        if let Some(size) = self.min_subgraph_size {
            p = p.with_min_subgraph_size(size);
        }
        if let Some(v) = self.layer_norm_fp32_fallback {
            p = p.with_layer_norm_fp32_fallback(v);
        }

        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn test_tensorrt_config_default() {
        let config = TensorRTConfig::default();
        assert_eq!(config.fp16, Some(true));
        assert_eq!(config.cuda_graph, Some(false)); // Disabled by default due to ONNX Runtime bug
        assert_eq!(config.engine_cache, Some(true));
        assert_eq!(config.timing_cache, Some(true));
        assert_eq!(config.builder_optimization_level, Some(3));
        assert_eq!(config.int8, None);
        assert_eq!(config.engine_cache_path, None);
    }

    #[test]
    fn test_tensorrt_config_new() {
        let config = TensorRTConfig::new();
        assert_eq!(config.fp16, Some(true));
        assert_eq!(config.cuda_graph, Some(false)); // Disabled by default due to ONNX Runtime bug
        assert_eq!(config.engine_cache, Some(true));
        assert_eq!(config.timing_cache, Some(true));
        assert_eq!(config.builder_optimization_level, Some(3));
    }

    #[test]
    fn test_tensorrt_config_builder_pattern() {
        let config = TensorRTConfig::new()
            .with_fp16(false)
            .with_device_id(1)
            .with_max_workspace_size(1_000_000_000);

        assert_eq!(config.fp16, Some(false));
        assert_eq!(config.device_id, Some(1));
        assert_eq!(config.max_workspace_size, Some(1_000_000_000));
    }

    #[test]
    fn test_tensorrt_config_disable_all_optimizations() {
        let config = TensorRTConfig::new()
            .with_fp16(false)
            .with_cuda_graph(false)
            .with_engine_cache(false)
            .with_timing_cache(false);

        assert_eq!(config.fp16, Some(false));
        assert_eq!(config.cuda_graph, Some(false));
        assert_eq!(config.engine_cache, Some(false));
        assert_eq!(config.timing_cache, Some(false));
    }

    #[test]
    fn test_tensorrt_config_cache_paths() {
        let config = TensorRTConfig::new()
            .with_engine_cache_path("/tmp/engines")
            .with_timing_cache_path("/tmp/timing");

        assert_eq!(config.engine_cache_path, Some("/tmp/engines".to_string()));
        assert_eq!(config.timing_cache_path, Some("/tmp/timing".to_string()));
    }

    #[test]
    fn test_tensorrt_config_optimization_levels() {
        let config0 = TensorRTConfig::new().with_builder_optimization_level(0);
        let config5 = TensorRTConfig::new().with_builder_optimization_level(5);

        assert_eq!(config0.builder_optimization_level, Some(0));
        assert_eq!(config5.builder_optimization_level, Some(5));
    }

    #[test]
    fn test_tensorrt_config_int8() {
        let config = TensorRTConfig::new().with_int8(true);
        assert_eq!(config.int8, Some(true));
    }

    #[test]
    fn test_tensorrt_config_layer_norm_fallback() {
        let config = TensorRTConfig::new().with_layer_norm_fp32_fallback(true);
        assert_eq!(config.layer_norm_fp32_fallback, Some(true));
    }

    #[test]
    fn test_tensorrt_config_min_subgraph_size() {
        let config = TensorRTConfig::new().with_min_subgraph_size(5);
        assert_eq!(config.min_subgraph_size, Some(5));
    }

    #[test]
    fn test_tensorrt_config_cuda_graph_opt_in() {
        // Verify users can explicitly enable CUDA graphs if needed
        let config = TensorRTConfig::new().with_cuda_graph(true);
        assert_eq!(config.cuda_graph, Some(true));
    }
}
