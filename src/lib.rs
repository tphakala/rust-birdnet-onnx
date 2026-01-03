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
//! use birdnet_onnx::Classifier;
//! use ort::execution_providers::CUDAExecutionProvider;
//!
//! let classifier = Classifier::builder()
//!     .model_path("model.onnx")
//!     .labels_path("labels.txt")
//!     .execution_provider(CUDAExecutionProvider::default())
//!     .build()?;
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
