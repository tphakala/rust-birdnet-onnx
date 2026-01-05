//! CUDA execution provider configuration
//!
//! This module provides fine-grained control over CUDA memory allocation settings.
//! For most users, the default [`crate::ClassifierBuilder::with_cuda()`] method provides
//! safe defaults. Use [`CUDAConfig`] when you need custom memory settings.
//!
//! # Memory Management
//!
//! ONNX Runtime's CUDA provider uses a `BFCArena` allocator that can cause sudden
//! large memory allocations with its default `NextPowerOfTwo` strategy. This
//! configuration allows you to:
//! - Control how memory allocation grows over time
//! - Limit total GPU memory usage
//!
//! # Example
//!
//! ```no_run
//! use birdnet_onnx::{Classifier, CUDAConfig};
//!
//! let config = CUDAConfig::new()
//!     .with_memory_limit(8 * 1024 * 1024 * 1024)  // 8GB limit
//!     .with_device_id(1);  // Use second GPU
//!
//! let classifier = Classifier::builder()
//!     .model_path("model.onnx")
//!     .labels_path("labels.txt")
//!     .with_cuda_config(config)
//!     .build()?;
//! # Ok::<(), birdnet_onnx::Error>(())
//! ```

/// Re-export arena strategy from ort for user convenience
pub use ort::execution_providers::ArenaExtendStrategy;

/// Configuration for CUDA execution provider
///
/// This struct provides control over CUDA memory allocation and device settings.
/// For most users, the default [`crate::ClassifierBuilder::with_cuda()`] method
/// with its safe defaults is sufficient.
///
/// # Safe Defaults
///
/// The default configuration uses:
/// - **`SameAsRequested` arena strategy**: Prevents exponential memory growth
/// - **No memory limit**: Uses all available GPU memory (set a limit for constrained systems)
///
/// # Why `SameAsRequested`?
///
/// ONNX Runtime's default `NextPowerOfTwo` strategy doubles allocation sizes each time
/// more memory is needed. This can cause sudden jumps from 2GB to 4GB that freeze
/// Windows systems when total VRAM is exhausted. `SameAsRequested` allocates exactly
/// what's needed, providing more gradual and predictable memory growth.
///
/// # Example: Memory-Constrained System
///
/// ```no_run
/// use birdnet_onnx::CUDAConfig;
///
/// // Limit to 4GB for systems with 8GB VRAM
/// let config = CUDAConfig::new()
///     .with_memory_limit(4 * 1024 * 1024 * 1024);
/// # let _ = config;
/// ```
///
/// # Example: Restore Default ONNX Runtime Behavior
///
/// ```no_run
/// use birdnet_onnx::{CUDAConfig, ArenaExtendStrategy};
///
/// // Use ONNX Runtime's default exponential growth (not recommended)
/// let config = CUDAConfig::new()
///     .with_arena_extend_strategy(ArenaExtendStrategy::NextPowerOfTwo);
/// # let _ = config;
/// ```
#[derive(Debug, Clone)]
pub struct CUDAConfig {
    // Memory options
    memory_limit: Option<usize>,
    arena_extend_strategy: Option<ArenaExtendStrategy>,

    // Hardware options
    device_id: Option<i32>,

    // Performance options
    cuda_graph: Option<bool>,
    tf32: Option<bool>,
    conv_max_workspace: Option<bool>,
}

impl Default for CUDAConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl CUDAConfig {
    /// Create a new CUDA configuration with safe defaults
    ///
    /// # Safe Defaults
    ///
    /// - **Arena extend strategy**: `SameAsRequested` to prevent exponential memory growth
    /// - **Memory limit**: None (uses all available GPU memory)
    ///
    /// The `SameAsRequested` strategy is used by default because ONNX Runtime's
    /// default `NextPowerOfTwo` strategy can cause sudden 4GB+ allocations that
    /// freeze Windows systems with limited GPU memory.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            // Safe default: prevent exponential memory growth
            arena_extend_strategy: Some(ArenaExtendStrategy::SameAsRequested),

            // No limit by default - users can set based on their system
            memory_limit: None,

            // Hardware options - use system defaults
            device_id: None,

            // Performance options - use ort defaults
            cuda_graph: None,
            tf32: None,
            conv_max_workspace: None,
        }
    }

    /// Set the device memory arena size limit in bytes
    ///
    /// This limits how much GPU memory the CUDA execution provider can allocate.
    /// Note that actual memory usage may be slightly higher due to non-arena allocations.
    ///
    /// # Common Values
    ///
    /// - `4 * 1024 * 1024 * 1024` (4GB) - Conservative for 8GB cards
    /// - `8 * 1024 * 1024 * 1024` (8GB) - Safe for 16GB cards
    /// - `12 * 1024 * 1024 * 1024` (12GB) - For 24GB+ cards
    ///
    /// # Example
    ///
    /// ```no_run
    /// use birdnet_onnx::CUDAConfig;
    ///
    /// let config = CUDAConfig::new()
    ///     .with_memory_limit(8 * 1024 * 1024 * 1024);  // 8GB
    /// # let _ = config;
    /// ```
    ///
    /// Default: None (uses all available GPU memory)
    #[must_use]
    pub const fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Set the strategy for extending the device memory arena
    ///
    /// # Strategies
    ///
    /// - `SameAsRequested` (default): Allocates exactly what's requested. More predictable
    ///   memory usage but may have slightly higher allocation overhead.
    /// - `NextPowerOfTwo`: Doubles allocation sizes. Can lead to sudden large allocations
    ///   (e.g., jumping from 2GB to 4GB).
    ///
    /// # Warning
    ///
    /// `NextPowerOfTwo` is ONNX Runtime's default and can cause sudden 4GB+ allocations
    /// that may freeze systems with limited GPU memory.
    ///
    /// Default: `SameAsRequested`
    #[must_use]
    pub const fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
        self.arena_extend_strategy = Some(strategy);
        self
    }

    /// Set GPU device ID for multi-GPU systems
    ///
    /// Default: None (uses default GPU, usually device 0)
    #[must_use]
    pub const fn with_device_id(mut self, device_id: i32) -> Self {
        self.device_id = Some(device_id);
        self
    }

    /// Enable or disable CUDA graph capture
    ///
    /// CUDA graphs reduce CPU launch overhead but may not be compatible with all models.
    ///
    /// Default: None (uses ort default, typically disabled)
    #[must_use]
    pub const fn with_cuda_graph(mut self, enable: bool) -> Self {
        self.cuda_graph = Some(enable);
        self
    }

    /// Enable or disable TensorFloat-32 precision
    ///
    /// TF32 provides ~3x faster matrix operations on Ampere+ GPUs with
    /// slightly reduced precision (acceptable for most ML workloads).
    ///
    /// Default: None (uses ort default)
    #[must_use]
    pub const fn with_tf32(mut self, enable: bool) -> Self {
        self.tf32 = Some(enable);
        self
    }

    /// Enable or disable unlimited convolution workspace memory
    ///
    /// When disabled, limits convolution workspace to 32MB.
    ///
    /// Default: None (uses ort default)
    #[must_use]
    pub const fn with_conv_max_workspace(mut self, enable: bool) -> Self {
        self.conv_max_workspace = Some(enable);
        self
    }

    /// Apply configuration to a CUDA execution provider
    ///
    /// This is an internal method used by `ClassifierBuilder::with_cuda_config()`.
    pub(crate) fn apply_to(
        self,
        provider: ort::execution_providers::CUDAExecutionProvider,
    ) -> ort::execution_providers::CUDAExecutionProvider {
        let mut p = provider;

        if let Some(limit) = self.memory_limit {
            p = p.with_memory_limit(limit);
        }
        if let Some(strategy) = self.arena_extend_strategy {
            p = p.with_arena_extend_strategy(strategy);
        }
        if let Some(id) = self.device_id {
            p = p.with_device_id(id);
        }
        if let Some(v) = self.cuda_graph {
            p = p.with_cuda_graph(v);
        }
        if let Some(v) = self.tf32 {
            p = p.with_tf32(v);
        }
        if let Some(v) = self.conv_max_workspace {
            p = p.with_conv_max_workspace(v);
        }

        p
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn test_cuda_config_default() {
        let config = CUDAConfig::default();
        // Safe default: SameAsRequested to prevent exponential growth
        assert!(matches!(
            config.arena_extend_strategy,
            Some(ArenaExtendStrategy::SameAsRequested)
        ));
        assert_eq!(config.memory_limit, None);
        assert_eq!(config.device_id, None);
        assert_eq!(config.cuda_graph, None);
        assert_eq!(config.tf32, None);
        assert_eq!(config.conv_max_workspace, None);
    }

    #[test]
    fn test_cuda_config_new() {
        let config = CUDAConfig::new();
        assert!(matches!(
            config.arena_extend_strategy,
            Some(ArenaExtendStrategy::SameAsRequested)
        ));
    }

    #[test]
    fn test_cuda_config_memory_limit() {
        let config = CUDAConfig::new().with_memory_limit(8 * 1024 * 1024 * 1024);
        assert_eq!(config.memory_limit, Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_cuda_config_arena_strategy_next_power_of_two() {
        let config =
            CUDAConfig::new().with_arena_extend_strategy(ArenaExtendStrategy::NextPowerOfTwo);
        assert!(matches!(
            config.arena_extend_strategy,
            Some(ArenaExtendStrategy::NextPowerOfTwo)
        ));
    }

    #[test]
    fn test_cuda_config_device_id() {
        let config = CUDAConfig::new().with_device_id(1);
        assert_eq!(config.device_id, Some(1));
    }

    #[test]
    fn test_cuda_config_cuda_graph() {
        let config = CUDAConfig::new().with_cuda_graph(true);
        assert_eq!(config.cuda_graph, Some(true));
    }

    #[test]
    fn test_cuda_config_tf32() {
        let config = CUDAConfig::new().with_tf32(true);
        assert_eq!(config.tf32, Some(true));
    }

    #[test]
    fn test_cuda_config_conv_max_workspace() {
        let config = CUDAConfig::new().with_conv_max_workspace(false);
        assert_eq!(config.conv_max_workspace, Some(false));
    }

    #[test]
    fn test_cuda_config_builder_chain() {
        let config = CUDAConfig::new()
            .with_memory_limit(4_000_000_000)
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .with_device_id(0)
            .with_cuda_graph(true)
            .with_tf32(true)
            .with_conv_max_workspace(true);

        assert_eq!(config.memory_limit, Some(4_000_000_000));
        assert!(matches!(
            config.arena_extend_strategy,
            Some(ArenaExtendStrategy::SameAsRequested)
        ));
        assert_eq!(config.device_id, Some(0));
        assert_eq!(config.cuda_graph, Some(true));
        assert_eq!(config.tf32, Some(true));
        assert_eq!(config.conv_max_workspace, Some(true));
    }
}
