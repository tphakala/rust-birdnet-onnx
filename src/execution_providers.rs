//! Execution provider detection and query utilities

use crate::types::ExecutionProviderInfo;
use ort::execution_providers::{
    ACLExecutionProvider, ArmNNExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProvider, OneDNNExecutionProvider,
    OpenVINOExecutionProvider, QNNExecutionProvider, ROCmExecutionProvider,
    TensorRTExecutionProvider,
};
use std::sync::{Arc, Mutex};
use tracing::Subscriber;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};

/// Helper macro to check provider availability and add to list
macro_rules! check_provider {
    ($providers:ident, $provider:ty, $variant:expr) => {
        if <$provider>::default().is_available().unwrap_or(false) {
            $providers.push($variant);
        }
    };
}

/// Query all execution providers compiled into ONNX Runtime.
///
/// Returns a vector of execution providers that ONNX Runtime was built with support for.
/// Note: This checks compile-time availability, not runtime usability. A provider may be
/// compiled in but fail to initialize due to missing drivers or hardware.
///
/// # Example
///
/// ```
/// use birdnet_onnx::available_execution_providers;
///
/// let providers = available_execution_providers();
/// for provider in providers {
///     println!("Available: {} ({})", provider.as_str(), provider.category());
/// }
/// ```
#[must_use]
pub fn available_execution_providers() -> Vec<ExecutionProviderInfo> {
    let mut providers = Vec::new();

    // CPU is always available
    providers.push(ExecutionProviderInfo::Cpu);

    // Check each provider using is_available()
    // These checks only verify compile-time support, not runtime initialization
    check_provider!(
        providers,
        CUDAExecutionProvider,
        ExecutionProviderInfo::Cuda
    );
    check_provider!(
        providers,
        TensorRTExecutionProvider,
        ExecutionProviderInfo::TensorRt
    );
    check_provider!(
        providers,
        DirectMLExecutionProvider,
        ExecutionProviderInfo::DirectMl
    );
    check_provider!(
        providers,
        CoreMLExecutionProvider,
        ExecutionProviderInfo::CoreMl
    );
    check_provider!(
        providers,
        ROCmExecutionProvider,
        ExecutionProviderInfo::Rocm
    );
    check_provider!(
        providers,
        OpenVINOExecutionProvider,
        ExecutionProviderInfo::OpenVino
    );
    check_provider!(
        providers,
        OneDNNExecutionProvider,
        ExecutionProviderInfo::OneDnn
    );
    check_provider!(providers, QNNExecutionProvider, ExecutionProviderInfo::Qnn);
    check_provider!(providers, ACLExecutionProvider, ExecutionProviderInfo::Acl);
    check_provider!(
        providers,
        ArmNNExecutionProvider,
        ExecutionProviderInfo::ArmNn
    );

    providers
}

/// Detects execution provider fallback by capturing tracing events.
#[derive(Clone)]
pub struct ExecutionProviderDetector {
    state: Arc<Mutex<DetectorState>>,
}

#[derive(Default)]
struct DetectorState {
    cpu_fallback: bool,
    registration_errors: Vec<String>,
}

impl ExecutionProviderDetector {
    /// Create a new detector and install it as a tracing subscriber.
    ///
    /// Returns the detector and a guard that will uninstall the subscriber when dropped.
    pub fn new() -> (Self, tracing::subscriber::DefaultGuard) {
        let detector = Self {
            state: Arc::new(Mutex::new(DetectorState::default())),
        };

        let layer = DetectorLayer {
            state: Arc::clone(&detector.state),
        };

        let subscriber = tracing_subscriber::registry().with(layer);
        let guard = tracing::subscriber::set_default(subscriber);

        (detector, guard)
    }

    /// Check if CPU fallback was detected.
    pub fn cpu_fallback_detected(&self) -> bool {
        self.state.lock().map(|s| s.cpu_fallback).unwrap_or(false)
    }

    /// Get registration error messages.
    #[allow(dead_code)]
    pub fn registration_errors(&self) -> Vec<String> {
        self.state
            .lock()
            .map(|s| s.registration_errors.clone())
            .unwrap_or_default()
    }
}

/// Tracing layer that captures execution provider events.
struct DetectorLayer {
    state: Arc<Mutex<DetectorState>>,
}

impl<S: Subscriber> Layer<S> for DetectorLayer {
    #[allow(clippy::items_after_statements)]
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        use tracing::field::Visit;

        // Check if this is an ort execution provider event
        let metadata = event.metadata();
        if metadata.target() != "ort::execution_providers" {
            return;
        }

        // Extract the message field
        struct MessageVisitor(String);
        impl Visit for MessageVisitor {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "message" {
                    self.0 = value.to_string();
                }
            }

            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{value:?}");
                }
            }
        }

        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        let message = visitor.0;

        // Detect CPU fallback warning
        if metadata.level() == &tracing::Level::WARN
            && message
                .contains("No execution providers from session options registered successfully")
            && let Ok(mut state) = self.state.lock()
        {
            state.cpu_fallback = true;
        }

        // Capture registration errors
        if metadata.level() == &tracing::Level::ERROR
            && message.contains("An error occurred when attempting to register")
            && let Ok(mut state) = self.state.lock()
        {
            state.registration_errors.push(message);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_cpu_always_available() {
        let providers = available_execution_providers();
        assert!(
            providers.contains(&ExecutionProviderInfo::Cpu),
            "CPU should always be available"
        );
    }

    #[test]
    fn test_returns_non_empty() {
        let providers = available_execution_providers();
        assert!(!providers.is_empty(), "Should have at least CPU");
    }

    #[test]
    fn test_no_duplicates() {
        let providers = available_execution_providers();
        let unique_count = providers
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(
            providers.len(),
            unique_count,
            "Should not have duplicate providers"
        );
    }

    #[test]
    fn test_detect_cpu_fallback_from_tracing() {
        let (detector, _guard) = ExecutionProviderDetector::new();

        // Simulate the warning log that ort emits
        tracing::warn!(
            target: "ort::execution_providers",
            "No execution providers from session options registered successfully; may fall back to CPU."
        );

        assert!(detector.cpu_fallback_detected());
    }
}
