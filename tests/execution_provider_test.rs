//! Tests for execution provider detection

#![allow(clippy::unwrap_used)]

use birdnet_onnx::{Classifier, ExecutionProviderInfo, available_execution_providers};
use std::path::Path;

const FIXTURES_DIR: &str = "tests/fixtures";

fn fixtures_available() -> bool {
    Path::new(FIXTURES_DIR).join("birdnet_v24.onnx").exists()
}

#[test]
fn test_available_providers_includes_cpu() {
    let providers = available_execution_providers();
    assert!(
        providers.contains(&ExecutionProviderInfo::Cpu),
        "CPU should always be available"
    );
}

#[test]
fn test_classifier_default_uses_cpu() {
    if !fixtures_available() {
        eprintln!("Skipping: fixtures not available");
        return;
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()
        .unwrap();

    assert_eq!(
        classifier.execution_provider(),
        ExecutionProviderInfo::Cpu,
        "Default should use CPU"
    );
}

#[test]
fn test_classifier_cuda_fallback_to_cpu() {
    if !fixtures_available() {
        eprintln!("Skipping: fixtures not available");
        return;
    }

    // Request CUDA (will likely fail on most CI systems)
    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .with_cuda()
        .build()
        .unwrap();

    // Should either be CUDA (if available) or CPU (if not)
    let provider = classifier.execution_provider();
    assert!(
        provider == ExecutionProviderInfo::Cuda || provider == ExecutionProviderInfo::Cpu,
        "Should be CUDA or CPU, got: {:?}",
        provider
    );
}

#[test]
fn test_execution_provider_info_display() {
    assert_eq!(ExecutionProviderInfo::Cpu.to_string(), "CPU");
    assert_eq!(ExecutionProviderInfo::Cuda.to_string(), "CUDA");
    assert_eq!(ExecutionProviderInfo::TensorRt.to_string(), "TensorRT");
}

#[test]
fn test_execution_provider_info_category() {
    assert_eq!(ExecutionProviderInfo::Cpu.category(), "CPU");
    assert_eq!(ExecutionProviderInfo::Cuda.category(), "GPU");
    assert_eq!(ExecutionProviderInfo::CoreMl.category(), "Neural Engine");
    assert_eq!(ExecutionProviderInfo::Qnn.category(), "NPU");
}
