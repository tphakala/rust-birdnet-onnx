//! # birdnet-onnx
//!
//! A Rust library for running inference on `BirdNET` and Perch ONNX models.
//!
//! ## Supported Models
//!
//! - **`BirdNET` v2.4**: 48kHz, 3s segments (144,000 samples)
//! - **`BirdNET` v3.0**: 32kHz, 5s segments (160,000 samples)
//! - **Perch v2**: 32kHz, 5s segments (160,000 samples)
//!
//! ## Example
//!
//! ```ignore
//! use birdnet_onnx::{Classifier, available_execution_providers};
//!
//! // Query available execution providers
//! let providers = available_execution_providers();
//! println!("Available providers: {:?}", providers);
//!
//! let classifier = Classifier::builder()
//!     .model_path("model.onnx")
//!     .labels_path("labels.txt")
//!     .with_cuda()  // Request CUDA acceleration
//!     .build()?;
//!
//! // Check which provider is actually being used
//! println!("Using: {} ({})",
//!     classifier.execution_provider().as_str(),
//!     classifier.execution_provider().category()
//! );
//!
//! let result = classifier.predict(&audio_segment)?;
//! for pred in &result.predictions {
//!     println!("{}: {:.1}%", pred.species, pred.confidence * 100.0);
//! }
//! ```

// Crate-level lint configuration
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_crate_versions)] // ort and tracing-subscriber use different smallvec versions

mod classifier;
mod detection;
mod error;
mod execution_providers;
mod labels;
mod postprocess;
mod rangefilter;
mod runtime;
#[cfg(test)]
mod testutil;
mod types;

pub use classifier::{Classifier, ClassifierBuilder};
pub use error::{Error, Result};
pub use execution_providers::available_execution_providers;
pub use rangefilter::{
    RangeFilter, RangeFilterBuilder, calculate_week, validate_coordinates, validate_date,
};
pub use runtime::init_runtime;
pub use types::{
    ExecutionProviderInfo, LabelFormat, LocationScore, ModelConfig, ModelType, Prediction,
    PredictionResult,
};
