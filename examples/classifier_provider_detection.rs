//! Example: Detect active execution provider in classifier
//!
//! Run with: cargo run --example classifier_provider_detection

use birdnet_onnx::Classifier;

fn main() -> birdnet_onnx::Result<()> {
    println!("Classifier Execution Provider Detection Example\n");

    // You would need actual model files to run this
    // This is a template showing the API usage

    let model_path =
        std::env::var("BIRDNET_MODEL").unwrap_or_else(|_| "path/to/model.onnx".to_string());
    let labels_path =
        std::env::var("BIRDNET_LABELS").unwrap_or_else(|_| "path/to/labels.txt".to_string());

    if !std::path::Path::new(&model_path).exists() {
        println!("Model not found. Set BIRDNET_MODEL and BIRDNET_LABELS env vars.");
        println!("\nExample usage:");
        println!("  export BIRDNET_MODEL=/path/to/model.onnx");
        println!("  export BIRDNET_LABELS=/path/to/labels.txt");
        println!("  cargo run --example classifier_provider_detection");
        return Ok(());
    }

    // Build with CUDA request
    println!("Building classifier with CUDA request...");
    let classifier = Classifier::builder()
        .model_path(model_path)
        .labels_path(labels_path)
        .with_cuda()
        .build()?;

    // Check actual provider
    let provider = classifier.execution_provider();
    println!("\nActive execution provider:");
    println!("  Name: {}", provider.as_str());
    println!("  Category: {}", provider.category());

    match provider {
        birdnet_onnx::ExecutionProviderInfo::Cuda => {
            println!("\n✓ Successfully running on CUDA GPU");
        }
        birdnet_onnx::ExecutionProviderInfo::Cpu => {
            println!("\n⚠ Fell back to CPU (CUDA not available)");
        }
        _ => {
            println!("\n✓ Running on {} accelerator", provider.as_str());
        }
    }

    Ok(())
}
