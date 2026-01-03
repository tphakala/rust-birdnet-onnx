//! Execution provider availability detection.
//!
//! This module provides functionality to query which ONNX Runtime execution
//! providers are available at compile-time.

use crate::types::ExecutionProviderInfo;
use ort::execution_providers::{
    ACLExecutionProvider, ArmNNExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
    DirectMLExecutionProvider, ExecutionProvider, OneDNNExecutionProvider,
    OpenVINOExecutionProvider, QNNExecutionProvider, ROCmExecutionProvider,
    TensorRTExecutionProvider,
};

/// Returns a list of execution providers available at compile-time.
///
/// This function checks which execution providers the `ort` crate was built with
/// support for. CPU is always available. Other providers (CUDA, `TensorRT`, etc.)
/// are only included if they are available.
///
/// # Returns
///
/// A vector of [`ExecutionProviderInfo`] representing all available providers.
/// CPU is always first in the list.
///
/// # Example
///
/// ```
/// use birdnet_onnx::execution_providers::available_execution_providers;
///
/// let providers = available_execution_providers();
/// assert!(!providers.is_empty());
/// println!("Available providers: {:?}", providers);
/// ```
#[must_use]
pub fn available_execution_providers() -> Vec<ExecutionProviderInfo> {
    let mut providers = vec![ExecutionProviderInfo::Cpu];

    macro_rules! check_provider {
        ($provider:ty, $variant:expr) => {
            if <$provider>::default().is_available().unwrap_or(false) {
                providers.push($variant);
            }
        };
    }

    check_provider!(CUDAExecutionProvider, ExecutionProviderInfo::Cuda);
    check_provider!(TensorRTExecutionProvider, ExecutionProviderInfo::TensorRt);
    check_provider!(DirectMLExecutionProvider, ExecutionProviderInfo::DirectMl);
    check_provider!(CoreMLExecutionProvider, ExecutionProviderInfo::CoreMl);
    check_provider!(ROCmExecutionProvider, ExecutionProviderInfo::Rocm);
    check_provider!(OpenVINOExecutionProvider, ExecutionProviderInfo::OpenVino);
    check_provider!(OneDNNExecutionProvider, ExecutionProviderInfo::OneDnn);
    check_provider!(QNNExecutionProvider, ExecutionProviderInfo::Qnn);
    check_provider!(ACLExecutionProvider, ExecutionProviderInfo::Acl);
    check_provider!(ArmNNExecutionProvider, ExecutionProviderInfo::ArmNn);

    providers
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_cpu_always_available() {
        let providers = available_execution_providers();
        assert!(!providers.is_empty(), "Providers list should not be empty");
        assert_eq!(
            providers[0],
            ExecutionProviderInfo::Cpu,
            "CPU should always be the first provider"
        );
    }

    #[test]
    fn test_no_duplicates() {
        let providers = available_execution_providers();
        let mut seen = std::collections::HashSet::new();
        for provider in &providers {
            assert!(
                seen.insert(provider),
                "Duplicate provider found: {:?}",
                provider
            );
        }
    }

    #[test]
    fn test_function_doesnt_panic() {
        // This test verifies the function completes without panicking
        let providers = available_execution_providers();
        // The actual list depends on compile-time configuration
        assert!(!providers.is_empty());
    }

    #[test]
    fn test_consistent_results() {
        // Calling multiple times should give consistent results
        let result1 = available_execution_providers();
        let result2 = available_execution_providers();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_cpu_only_in_result_once() {
        let providers = available_execution_providers();
        let cpu_count = providers
            .iter()
            .filter(|p| **p == ExecutionProviderInfo::Cpu)
            .count();
        assert_eq!(cpu_count, 1, "CPU should appear exactly once");
    }
}
