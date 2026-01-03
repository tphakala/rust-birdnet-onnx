//! Example: Query available execution providers
//!
//! Run with: cargo run --example execution_provider_query

use birdnet_onnx::{ExecutionProviderInfo, available_execution_providers};

fn main() {
    println!("Execution Provider Detection Example\n");

    // Query all available execution providers
    let providers = available_execution_providers();

    println!("Available execution providers ({}):", providers.len());
    for provider in &providers {
        println!("  - {} ({})", provider.as_str(), provider.category());
    }

    println!("\nProvider Categories:");
    let mut by_category: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();

    for provider in &providers {
        by_category
            .entry(provider.category())
            .or_default()
            .push(provider.as_str());
    }

    for (category, providers) in by_category {
        println!("  {}: {}", category, providers.join(", "));
    }

    // Check for specific providers
    println!("\nGPU Support:");
    if providers.contains(&ExecutionProviderInfo::Cuda) {
        println!("  ✓ CUDA (NVIDIA)");
    }
    if providers.contains(&ExecutionProviderInfo::Rocm) {
        println!("  ✓ ROCm (AMD)");
    }
    if providers.contains(&ExecutionProviderInfo::DirectMl) {
        println!("  ✓ DirectML (Windows)");
    }

    println!("\nNPU Support:");
    if providers.contains(&ExecutionProviderInfo::Qnn) {
        println!("  ✓ QNN (Qualcomm)");
    }

    println!("\nNeural Engine Support:");
    if providers.contains(&ExecutionProviderInfo::CoreMl) {
        println!("  ✓ CoreML (Apple)");
    }
}
