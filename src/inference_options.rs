//! Inference execution options for timeout and cancellation control.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Token for cancelling inference from another thread.
///
/// Clone this token and pass it to the inference call. The original can be used
/// to cancel the inference from any thread.
///
/// # Example
///
/// ```
/// use birdnet_onnx::CancellationToken;
///
/// let token = CancellationToken::new();
/// let token_for_handler = token.clone();
///
/// // In a shutdown handler or another thread:
/// // token_for_handler.cancel();
/// ```
#[derive(Clone, Default, Debug)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation.
    ///
    /// Any inference using this token will be terminated as soon as possible.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation has been signalled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Options for controlling inference execution.
///
/// Use this to set timeouts and cancellation tokens for inference calls.
///
/// # Example
///
/// ```
/// use birdnet_onnx::InferenceOptions;
/// use std::time::Duration;
///
/// // No timeout (default behavior)
/// let opts = InferenceOptions::default();
///
/// // With 30 second timeout
/// let opts = InferenceOptions::timeout(Duration::from_secs(30));
///
/// // With timeout and cancellation token
/// use birdnet_onnx::CancellationToken;
/// let token = CancellationToken::new();
/// let opts = InferenceOptions::new()
///     .with_timeout(Duration::from_secs(30))
///     .with_cancellation_token(token);
/// ```
#[derive(Clone, Default, Debug)]
pub struct InferenceOptions {
    /// Maximum duration for inference before timing out.
    pub timeout: Option<Duration>,
    /// Token for external cancellation.
    pub cancellation_token: Option<CancellationToken>,
}

impl InferenceOptions {
    /// Create new inference options with defaults (no timeout, no cancellation).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create inference options with only a timeout.
    #[must_use]
    pub const fn timeout(duration: Duration) -> Self {
        Self {
            timeout: Some(duration),
            cancellation_token: None,
        }
    }

    /// Set the timeout duration.
    #[must_use]
    pub const fn with_timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Set the cancellation token.
    #[must_use]
    pub fn with_cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Check if any options are configured that require monitoring.
    #[allow(dead_code)] // Used in classifier.rs, will be used in next task
    pub(crate) const fn needs_monitor(&self) -> bool {
        self.timeout.is_some() || self.cancellation_token.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token_default() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_cancel() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone_shares_state() {
        let token1 = CancellationToken::new();
        let token2 = token1.clone();

        assert!(!token1.is_cancelled());
        assert!(!token2.is_cancelled());

        token1.cancel();

        assert!(token1.is_cancelled());
        assert!(token2.is_cancelled());
    }

    #[test]
    fn test_inference_options_default() {
        let opts = InferenceOptions::default();
        assert!(opts.timeout.is_none());
        assert!(opts.cancellation_token.is_none());
        assert!(!opts.needs_monitor());
    }

    #[test]
    fn test_inference_options_timeout() {
        let opts = InferenceOptions::timeout(Duration::from_secs(30));
        assert_eq!(opts.timeout, Some(Duration::from_secs(30)));
        assert!(opts.cancellation_token.is_none());
        assert!(opts.needs_monitor());
    }

    #[test]
    fn test_inference_options_builder() {
        let token = CancellationToken::new();
        let opts = InferenceOptions::new()
            .with_timeout(Duration::from_secs(60))
            .with_cancellation_token(token.clone());

        assert_eq!(opts.timeout, Some(Duration::from_secs(60)));
        assert!(opts.cancellation_token.is_some());
        assert!(opts.needs_monitor());
    }

    #[test]
    fn test_inference_options_only_cancellation() {
        let token = CancellationToken::new();
        let opts = InferenceOptions::new().with_cancellation_token(token);

        assert!(opts.timeout.is_none());
        assert!(opts.cancellation_token.is_some());
        assert!(opts.needs_monitor());
    }

    #[test]
    fn test_inference_options_debug() {
        let opts = InferenceOptions::timeout(Duration::from_secs(10));
        let debug = format!("{opts:?}");
        assert!(debug.contains("InferenceOptions"));
        assert!(debug.contains("timeout"));
    }

    #[test]
    fn test_cancellation_token_debug() {
        let token = CancellationToken::new();
        let debug = format!("{token:?}");
        assert!(debug.contains("CancellationToken"));
    }
}
