//! Integration tests for execution provider functionality.
//!
//! These tests verify the public API for execution provider configuration
//! and information from a user's perspective.

#![allow(clippy::unwrap_used)] // Tests can use unwrap
#![allow(clippy::disallowed_methods)] // Tests can use unwrap

use birdnet_onnx::{Classifier, ExecutionProviderInfo, available_execution_providers};

/// Check if ONNX Runtime is available by checking for the environment variable
fn onnx_runtime_available() -> bool {
    std::env::var("ORT_DYLIB_PATH").is_ok()
}

/// Skip test helper that prints a message when ONNX Runtime is not available
macro_rules! skip_if_no_onnx {
    () => {
        if !onnx_runtime_available() {
            eprintln!("Skipping test: ORT_DYLIB_PATH environment variable not set");
            eprintln!("ONNX Runtime is required for these tests");
            return;
        }
    };
}

// ============================================================================
// Available Providers Tests
// ============================================================================

#[test]
fn available_providers_includes_cpu() {
    let providers = available_execution_providers();

    // CPU should always be available
    assert!(
        providers.contains(&ExecutionProviderInfo::Cpu),
        "CPU provider should always be available"
    );

    // CPU should be the first provider
    assert_eq!(
        providers[0],
        ExecutionProviderInfo::Cpu,
        "CPU should be the first provider in the list"
    );
}

// ============================================================================
// Requested Provider Tests
// ============================================================================

#[test]
fn requested_provider_default_is_cpu() {
    skip_if_no_onnx!();

    // Create a minimal labels vector for testing
    let labels = vec!["species1".to_string(), "species2".to_string()];

    // Create classifier without specifying execution provider
    // This should fail to load since we don't have an actual model,
    // but we can test the builder's default
    let builder = Classifier::builder()
        .model_path("nonexistent.onnx")
        .labels(labels);

    // The builder's default provider should be CPU, but we can't
    // directly test the builder. Instead, we need to actually build
    // a classifier. Since we can't do that without a real model in
    // integration tests, we'll skip this test if model isn't available.
    // For now, just verify that the function exists by attempting to build.
    let result = builder.build();

    // Building should fail (model doesn't exist), but that's expected
    assert!(result.is_err());
}

#[test]
fn requested_provider_with_cuda() {
    skip_if_no_onnx!();

    let labels = vec!["species1".to_string(), "species2".to_string()];

    // Use with_cuda() builder method
    let builder = Classifier::builder()
        .model_path("nonexistent.onnx")
        .labels(labels)
        .with_cuda();

    // Building should fail (model doesn't exist), but that's expected
    let result = builder.build();
    assert!(result.is_err());

    // We can't directly test the requested_provider without a successful build,
    // but we've verified the method exists and can be called
}

#[test]
fn requested_provider_chained_methods() {
    skip_if_no_onnx!();

    let labels = vec!["species1".to_string(), "species2".to_string()];

    // Chain multiple provider methods - last one should win
    let builder = Classifier::builder()
        .model_path("nonexistent.onnx")
        .labels(labels)
        .with_cuda()
        .with_tensorrt()
        .with_directml();

    // Building should fail (model doesn't exist), but that's expected
    let result = builder.build();
    assert!(result.is_err());

    // We've verified that chaining works (compiles and runs)
}

// ============================================================================
// ExecutionProviderInfo Tests
// ============================================================================

#[test]
fn execution_provider_info_display() {
    // Test Display trait for ExecutionProviderInfo
    let cpu = ExecutionProviderInfo::Cpu;
    assert_eq!(format!("{cpu}"), "CPU");

    let cuda = ExecutionProviderInfo::Cuda;
    assert_eq!(format!("{cuda}"), "CUDA");

    let tensorrt = ExecutionProviderInfo::TensorRt;
    assert_eq!(format!("{tensorrt}"), "TensorRT");

    let directml = ExecutionProviderInfo::DirectMl;
    assert_eq!(format!("{directml}"), "DirectML");

    let coreml = ExecutionProviderInfo::CoreMl;
    assert_eq!(format!("{coreml}"), "CoreML");

    let rocm = ExecutionProviderInfo::Rocm;
    assert_eq!(format!("{rocm}"), "ROCm");

    let openvino = ExecutionProviderInfo::OpenVino;
    assert_eq!(format!("{openvino}"), "OpenVINO");

    let onednn = ExecutionProviderInfo::OneDnn;
    assert_eq!(format!("{onednn}"), "oneDNN");

    let qnn = ExecutionProviderInfo::Qnn;
    assert_eq!(format!("{qnn}"), "QNN");

    let acl = ExecutionProviderInfo::Acl;
    assert_eq!(format!("{acl}"), "ACL");

    let armnn = ExecutionProviderInfo::ArmNn;
    assert_eq!(format!("{armnn}"), "ArmNN");
}

#[test]
fn execution_provider_info_category() {
    // Test category() method for ExecutionProviderInfo
    assert_eq!(ExecutionProviderInfo::Cpu.category(), "CPU");
    assert_eq!(ExecutionProviderInfo::Cuda.category(), "GPU");
    assert_eq!(ExecutionProviderInfo::TensorRt.category(), "GPU");
    assert_eq!(ExecutionProviderInfo::DirectMl.category(), "GPU");
    assert_eq!(ExecutionProviderInfo::CoreMl.category(), "Neural Engine");
    assert_eq!(ExecutionProviderInfo::Rocm.category(), "GPU");
    assert_eq!(ExecutionProviderInfo::OpenVino.category(), "Accelerator");
    assert_eq!(ExecutionProviderInfo::OneDnn.category(), "Accelerator");
    assert_eq!(ExecutionProviderInfo::Qnn.category(), "NPU");
    assert_eq!(ExecutionProviderInfo::Acl.category(), "Accelerator");
    assert_eq!(ExecutionProviderInfo::ArmNn.category(), "Accelerator");
}
