//! Execution provider detection and query utilities

use crate::types::ExecutionProviderInfo;
use ort::execution_providers::{
    ACLExecutionProvider, ArmNNExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProvider, OneDNNExecutionProvider,
    OpenVINOExecutionProvider, QNNExecutionProvider, ROCmExecutionProvider,
    TensorRTExecutionProvider,
};

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
}
