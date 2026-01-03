//! Integration tests for execution provider functionality.
//!
//! These tests verify the public API for execution provider configuration
//! and information from a user's perspective.

#![allow(clippy::unwrap_used)] // Tests can use unwrap
#![allow(clippy::disallowed_methods)] // Tests can use unwrap

use birdnet_onnx::{ExecutionProviderInfo, available_execution_providers};

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
    skip_if_no_onnx!();
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
