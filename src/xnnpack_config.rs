//! XNNPACK execution provider configuration.
//!
//! XNNPACK provides optimized CPU inference for neural networks on ARM and x86
//! platforms, particularly effective for models with `Conv`, `Gemm`, and `MatMul`
//! operations (which `BirdNET` models use extensively).
//!
//! # Threading
//!
//! XNNPACK manages its own internal threadpool, separate from ONNX Runtime's
//! intra-op threadpool. The [`with_intra_op_num_threads()`](XNNPACKConfig::with_intra_op_num_threads)
//! method configures XNNPACK's threadpool size.
//!
//! For advanced session-level thread tuning (e.g., reducing ONNX Runtime's intra-op
//! threads to minimize contention), use [`ClassifierBuilder::execution_provider()`]
//! with a custom `ort::ep::XNNPACK` configuration.
//!
//! # Example
//!
//! ```no_run
//! use birdnet_onnx::{Classifier, XNNPACKConfig};
//! use core::num::NonZeroUsize;
//!
//! let config = XNNPACKConfig::new()
//!     .with_intra_op_num_threads(NonZeroUsize::new(4).unwrap());
//!
//! let classifier = Classifier::builder()
//!     .model_path("model.onnx")
//!     .labels_path("labels.txt")
//!     .with_xnnpack_config(config)
//!     .build()?;
//! # Ok::<(), birdnet_onnx::Error>(())
//! ```

use core::num::NonZeroUsize;

/// Configuration for the XNNPACK execution provider.
///
/// Use [`XNNPACKConfig::new()`] to create a default configuration, then customize
/// with builder methods like [`with_intra_op_num_threads()`](Self::with_intra_op_num_threads).
///
/// See the [module-level documentation](crate::xnnpack_config) for threading
/// recommendations and usage examples.
#[derive(Debug, Clone, Default)]
pub struct XNNPACKConfig {
    intra_op_num_threads: Option<NonZeroUsize>,
}

impl XNNPACKConfig {
    /// Create a new XNNPACK configuration with default settings.
    ///
    /// Default: Uses XNNPACK's default thread count.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            intra_op_num_threads: None,
        }
    }

    /// Set the number of threads for XNNPACK's internal threadpool.
    ///
    /// If not set, XNNPACK uses its default thread count (typically the number
    /// of CPU cores).
    ///
    /// # Example
    ///
    /// ```
    /// use birdnet_onnx::XNNPACKConfig;
    /// use core::num::NonZeroUsize;
    ///
    /// let config = XNNPACKConfig::new()
    ///     .with_intra_op_num_threads(NonZeroUsize::new(4).unwrap());
    /// # let _ = config;
    /// ```
    #[must_use]
    pub const fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self {
        self.intra_op_num_threads = Some(num_threads);
        self
    }

    /// Apply this configuration to an XNNPACK execution provider.
    ///
    /// This is an internal method used by `ClassifierBuilder::with_xnnpack_config()`.
    pub(crate) fn apply_to(self, mut provider: ort::ep::XNNPACK) -> ort::ep::XNNPACK {
        if let Some(threads) = self.intra_op_num_threads {
            provider = provider.with_intra_op_num_threads(threads);
        }
        provider
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn test_xnnpack_config_default() {
        let config = XNNPACKConfig::default();
        assert_eq!(config.intra_op_num_threads, None);
    }

    #[test]
    fn test_xnnpack_config_new() {
        let config = XNNPACKConfig::new();
        assert_eq!(config.intra_op_num_threads, None);
    }

    #[test]
    fn test_xnnpack_config_with_threads() {
        let config = XNNPACKConfig::new().with_intra_op_num_threads(NonZeroUsize::new(4).unwrap());
        assert_eq!(
            config.intra_op_num_threads,
            Some(NonZeroUsize::new(4).unwrap())
        );
    }
}
