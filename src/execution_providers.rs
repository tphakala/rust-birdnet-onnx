//! Execution provider detection and query utilities

use crate::types::ExecutionProviderInfo;
use ort::execution_providers::{
    ACLExecutionProvider, ArmNNExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProvider, OneDNNExecutionProvider,
    OpenVINOExecutionProvider, QNNExecutionProvider, ROCmExecutionProvider,
    TensorRTExecutionProvider,
};

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

    if CUDAExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::Cuda);
    }

    if TensorRTExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::TensorRt);
    }

    if DirectMLExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::DirectMl);
    }

    if CoreMLExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::CoreMl);
    }

    if ROCmExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::Rocm);
    }

    if OpenVINOExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::OpenVino);
    }

    if OneDNNExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::OneDnn);
    }

    if QNNExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::Qnn);
    }

    if ACLExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::Acl);
    }

    if ArmNNExecutionProvider::default()
        .is_available()
        .unwrap_or(false)
    {
        providers.push(ExecutionProviderInfo::ArmNn);
    }

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
