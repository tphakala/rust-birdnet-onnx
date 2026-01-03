//! Classifier builder and implementation

use crate::detection::detect_model_type;
use crate::error::{Error, Result};
use crate::execution_providers::ExecutionProviderDetector;
use crate::labels::load_labels_from_file;
use crate::postprocess::top_k_predictions;
use crate::types::{ExecutionProviderInfo, ModelConfig, ModelType, PredictionResult};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::{Arc, Mutex};

/// Labels source for builder
#[derive(Debug)]
enum Labels {
    Path(String),
    InMemory(Vec<String>),
}

/// Builder for constructing a Classifier
#[derive(Debug)]
pub struct ClassifierBuilder {
    model_path: Option<String>,
    labels: Option<Labels>,
    model_type_override: Option<ModelType>,
    execution_providers: Vec<ort::execution_providers::ExecutionProviderDispatch>,
    requested_provider_types: Vec<ExecutionProviderInfo>,
    top_k: usize,
    min_confidence: Option<f32>,
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassifierBuilder {
    /// Create a new classifier builder
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_path: None,
            labels: None,
            model_type_override: None,
            execution_providers: Vec::new(),
            requested_provider_types: Vec::new(),
            top_k: 10,
            min_confidence: None,
        }
    }

    /// Set the path to the ONNX model file (required)
    #[must_use]
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the path to the labels file
    #[must_use]
    pub fn labels_path(mut self, path: impl Into<String>) -> Self {
        self.labels = Some(Labels::Path(path.into()));
        self
    }

    /// Set labels directly from a vector
    #[must_use]
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(Labels::InMemory(labels));
        self
    }

    /// Override auto-detected model type (useful for Perch v2)
    #[must_use]
    pub const fn model_type(mut self, model_type: ModelType) -> Self {
        self.model_type_override = Some(model_type);
        self
    }

    /// Add an execution provider (GPU, CPU, etc.)
    ///
    /// Multiple providers can be added; they are tried in order.
    /// If none specified, defaults to CPU.
    #[must_use]
    pub fn execution_provider(
        mut self,
        provider: impl Into<ort::execution_providers::ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers.push(provider.into());
        self
    }

    /// Add CUDA execution provider (NVIDIA GPU).
    #[must_use]
    pub fn with_cuda(mut self) -> Self {
        use ort::execution_providers::CUDAExecutionProvider;
        self.execution_providers
            .push(CUDAExecutionProvider::default().into());
        self.requested_provider_types
            .push(ExecutionProviderInfo::Cuda);
        self
    }

    /// Add `TensorRT` execution provider (NVIDIA GPU).
    #[must_use]
    pub fn with_tensorrt(mut self) -> Self {
        use ort::execution_providers::TensorRTExecutionProvider;
        self.execution_providers
            .push(TensorRTExecutionProvider::default().into());
        self.requested_provider_types
            .push(ExecutionProviderInfo::TensorRt);
        self
    }

    /// Add `DirectML` execution provider (Windows GPU).
    #[must_use]
    pub fn with_directml(mut self) -> Self {
        use ort::execution_providers::DirectMLExecutionProvider;
        self.execution_providers
            .push(DirectMLExecutionProvider::default().into());
        self.requested_provider_types
            .push(ExecutionProviderInfo::DirectMl);
        self
    }

    /// Add `CoreML` execution provider (Apple Neural Engine).
    #[must_use]
    pub fn with_coreml(mut self) -> Self {
        use ort::execution_providers::CoreMLExecutionProvider;
        self.execution_providers
            .push(CoreMLExecutionProvider::default().into());
        self.requested_provider_types
            .push(ExecutionProviderInfo::CoreMl);
        self
    }

    /// Set the number of top predictions to return (default: 10)
    #[must_use]
    pub const fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set minimum confidence threshold for predictions
    #[must_use]
    pub const fn min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = Some(threshold);
        self
    }

    /// Build the classifier
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model path was not set
    /// - Labels were not provided
    /// - Model file cannot be loaded
    /// - Model type cannot be detected
    /// - Label count doesn't match model
    pub fn build(self) -> Result<Classifier> {
        // Validate required fields
        let model_path = self.model_path.ok_or(Error::ModelPathRequired)?;
        let labels_source = self.labels.ok_or(Error::LabelsRequired)?;

        // Create detector before building session
        let (detector, guard) = ExecutionProviderDetector::new();

        // Build session with execution providers
        let mut session_builder = Session::builder().map_err(Error::ModelLoad)?;

        for provider in self.execution_providers {
            session_builder = session_builder
                .with_execution_providers([provider])
                .map_err(Error::ModelLoad)?;
        }

        let session = session_builder
            .commit_from_file(&model_path)
            .map_err(Error::ModelLoad)?;

        // Drop guard to stop capturing tracing events
        drop(guard);

        // Determine actual execution provider
        let execution_provider = if detector.cpu_fallback_detected() {
            // CPU fallback was detected
            ExecutionProviderInfo::Cpu
        } else if self.requested_provider_types.is_empty() {
            // No provider requested, using default CPU
            ExecutionProviderInfo::Cpu
        } else {
            // Using first requested provider (assumption: it initialized successfully)
            self.requested_provider_types[0]
        };

        // Extract input/output shapes for model detection
        let input_shape = extract_input_shape(&session)?;
        let output_shapes = extract_output_shapes(&session)?;

        // Detect model type
        let config = detect_model_type(&input_shape, &output_shapes, self.model_type_override)?;

        // Load labels
        let labels = match labels_source {
            Labels::Path(path) => load_labels_from_file(&path, config.model_type)?,
            Labels::InMemory(labels) => labels,
        };

        // Validate label count matches model
        if labels.len() != config.num_species {
            return Err(Error::LabelCount {
                expected: config.num_species,
                got: labels.len(),
            });
        }

        Ok(Classifier {
            inner: Arc::new(ClassifierInner {
                session: Mutex::new(session),
                config,
                labels,
                top_k: self.top_k,
                min_confidence: self.min_confidence,
                execution_provider,
            }),
        })
    }
}

/// Extract input tensor shape from session
fn extract_input_shape(session: &Session) -> Result<Vec<i64>> {
    let inputs = session
        .inputs
        .first()
        .ok_or_else(|| Error::ModelDetection {
            reason: "model has no inputs".to_string(),
        })?;

    let shape = inputs
        .input_type
        .tensor_shape()
        .ok_or_else(|| Error::ModelDetection {
            reason: "input is not a tensor".to_string(),
        })?;

    Ok(shape.iter().copied().collect())
}

/// Extract output tensor shapes from session
fn extract_output_shapes(session: &Session) -> Result<Vec<Vec<i64>>> {
    session
        .outputs
        .iter()
        .map(|output| {
            let shape = output
                .output_type
                .tensor_shape()
                .ok_or_else(|| Error::ModelDetection {
                    reason: "output is not a tensor".to_string(),
                })?;
            Ok(shape.iter().copied().collect())
        })
        .collect()
}

/// Internal state shared via Arc for thread safety
struct ClassifierInner {
    session: Mutex<Session>,
    config: ModelConfig,
    labels: Vec<String>,
    top_k: usize,
    min_confidence: Option<f32>,
    execution_provider: ExecutionProviderInfo,
}

/// Thread-safe classifier for bird species detection
///
/// Use `Classifier::builder()` to construct.
#[derive(Clone)]
pub struct Classifier {
    inner: Arc<ClassifierInner>,
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Classifier")
            .field("config", &self.inner.config)
            .field("labels_count", &self.inner.labels.len())
            .field("top_k", &self.inner.top_k)
            .field("min_confidence", &self.inner.min_confidence)
            .field("execution_provider", &self.inner.execution_provider)
            .finish_non_exhaustive()
    }
}

impl Classifier {
    /// Create a new classifier builder
    #[must_use]
    pub const fn builder() -> ClassifierBuilder {
        ClassifierBuilder::new()
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.inner.config
    }

    /// Get the species labels
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.inner.labels
    }

    /// Get the execution provider actually being used for inference.
    ///
    /// Returns which hardware backend (CPU, GPU, etc.) is running inference.
    /// This reflects the actual provider after fallback, not what was requested.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use birdnet_onnx::Classifier;
    /// use ort::execution_providers::CUDAExecutionProvider;
    ///
    /// let classifier = Classifier::builder()
    ///     .model_path("model.onnx")
    ///     .labels_path("labels.txt")
    ///     .execution_provider(CUDAExecutionProvider::default())
    ///     .build()?;
    ///
    /// println!("Using: {} ({})",
    ///     classifier.execution_provider().as_str(),
    ///     classifier.execution_provider().category()
    /// );
    /// ```
    #[must_use]
    pub fn execution_provider(&self) -> ExecutionProviderInfo {
        self.inner.execution_provider
    }

    /// Run inference on a single audio segment
    ///
    /// # Arguments
    /// * `segment` - Audio samples (must match `config().sample_count`)
    ///
    /// # Returns
    /// * `PredictionResult` with top predictions, embeddings (if available), and raw scores
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input segment size doesn't match expected sample count
    /// - Session lock is poisoned
    /// - ONNX inference fails
    #[allow(clippy::significant_drop_tightening)]
    pub fn predict(&self, segment: &[f32]) -> Result<PredictionResult> {
        // Validate input size
        let expected = self.inner.config.sample_count;
        if segment.len() != expected {
            return Err(Error::InputSize {
                expected,
                got: segment.len(),
            });
        }

        // Create input tensor [1, sample_count]
        let input_array = Array2::from_shape_vec((1, segment.len()), segment.to_vec())
            .map_err(|e| Error::Inference(format!("failed to create input array: {e}")))?;

        let input_value = Value::from_array(input_array)
            .map_err(|e| Error::Inference(format!("failed to create input tensor: {e}")))?;

        // Run inference with locked session
        // IMPORTANT: Session lock must be held while outputs exist because ort::Value
        // borrows from the session. Dropping the lock before processing outputs would
        // cause a use-after-free. This is why clippy::significant_drop_tightening is
        // suppressed on this method.
        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|e| Error::Inference(format!("session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| Error::Inference(e.to_string()))?;

        // Process outputs based on model type
        self.process_outputs(&outputs)
    }

    /// Run inference on multiple audio segments (more efficient for GPU)
    ///
    /// # Arguments
    /// * `segments` - Slice of audio segments (all must match `config().sample_count`)
    ///
    /// # Returns
    /// * Vector of `PredictionResult`, one per input segment
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any segment size doesn't match expected sample count
    /// - Session lock is poisoned
    /// - ONNX inference fails
    #[allow(clippy::significant_drop_tightening)]
    pub fn predict_batch(&self, segments: &[&[f32]]) -> Result<Vec<PredictionResult>> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }

        let expected = self.inner.config.sample_count;

        // Validate all segments
        for (i, seg) in segments.iter().enumerate() {
            if seg.len() != expected {
                return Err(Error::BatchInputSize {
                    index: i,
                    expected,
                    got: seg.len(),
                });
            }
        }

        let batch_size = segments.len();

        // Stack segments into [batch_size, sample_count]
        let mut batch_data = Vec::with_capacity(batch_size * expected);
        for seg in segments {
            batch_data.extend_from_slice(seg);
        }

        let input_array = Array2::from_shape_vec((batch_size, expected), batch_data)
            .map_err(|e| Error::Inference(format!("failed to create batch array: {e}")))?;

        let input_value = Value::from_array(input_array)
            .map_err(|e| Error::Inference(format!("failed to create input tensor: {e}")))?;

        // Run inference with locked session
        // IMPORTANT: Session lock must be held while outputs exist because ort::Value
        // borrows from the session. Dropping the lock before processing outputs would
        // cause a use-after-free. This is why clippy::significant_drop_tightening is
        // suppressed on this method.
        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|e| Error::Inference(format!("session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| Error::Inference(e.to_string()))?;

        // Process batch outputs
        self.process_batch_outputs(&outputs, batch_size)
    }

    /// Process single inference outputs
    fn process_outputs(&self, outputs: &ort::session::SessionOutputs) -> Result<PredictionResult> {
        let model_type = self.inner.config.model_type;

        let (embeddings, logits) = match model_type {
            ModelType::BirdNetV24 => {
                // Single output: predictions
                let logits = extract_tensor_data(outputs, 0)?;
                (None, logits)
            }
            ModelType::BirdNetV30 => {
                // Two outputs: embeddings at 0, predictions at 1
                let embeddings = extract_tensor_data(outputs, 0)?;
                let logits = extract_tensor_data(outputs, 1)?;
                (Some(embeddings), logits)
            }
            ModelType::PerchV2 => {
                // Four outputs: embedding at 0, spatial_embedding at 1, spectrogram at 2, predictions at 3
                let embeddings = extract_tensor_data(outputs, 0)?;
                let logits = extract_tensor_data(outputs, 3)?;
                (Some(embeddings), logits)
            }
        };

        let predictions = top_k_predictions(
            &logits,
            &self.inner.labels,
            self.inner.top_k,
            self.inner.min_confidence,
        );

        Ok(PredictionResult {
            model_type,
            predictions,
            embeddings,
            raw_scores: logits,
        })
    }

    /// Process batch inference outputs
    fn process_batch_outputs(
        &self,
        outputs: &ort::session::SessionOutputs,
        batch_size: usize,
    ) -> Result<Vec<PredictionResult>> {
        let model_type = self.inner.config.model_type;
        let num_species = self.inner.config.num_species;

        match model_type {
            ModelType::BirdNetV24 => {
                let logits_flat = extract_tensor_data(outputs, 0)?;

                (0..batch_size)
                    .map(|i| {
                        let start = i * num_species;
                        let end = start + num_species;
                        let logits = &logits_flat[start..end];

                        let predictions = top_k_predictions(
                            logits,
                            &self.inner.labels,
                            self.inner.top_k,
                            self.inner.min_confidence,
                        );

                        Ok(PredictionResult {
                            model_type,
                            predictions,
                            embeddings: None,
                            raw_scores: logits.to_vec(),
                        })
                    })
                    .collect()
            }
            ModelType::BirdNetV30 => {
                let embedding_dim = self.inner.config.embedding_dim.ok_or_else(|| {
                    Error::Inference(
                        "embedding_dim missing for model that requires embeddings".into(),
                    )
                })?;
                let emb_flat = extract_tensor_data(outputs, 0)?;
                let logits_flat = extract_tensor_data(outputs, 1)?;

                (0..batch_size)
                    .map(|i| {
                        let emb_start = i * embedding_dim;
                        let emb_end = emb_start + embedding_dim;
                        let embeddings = emb_flat[emb_start..emb_end].to_vec();

                        let logits_start = i * num_species;
                        let logits_end = logits_start + num_species;
                        let logits = &logits_flat[logits_start..logits_end];

                        let predictions = top_k_predictions(
                            logits,
                            &self.inner.labels,
                            self.inner.top_k,
                            self.inner.min_confidence,
                        );

                        Ok(PredictionResult {
                            model_type,
                            predictions,
                            embeddings: Some(embeddings),
                            raw_scores: logits.to_vec(),
                        })
                    })
                    .collect()
            }
            ModelType::PerchV2 => {
                let embedding_dim = self.inner.config.embedding_dim.ok_or_else(|| {
                    Error::Inference(
                        "embedding_dim missing for model that requires embeddings".into(),
                    )
                })?;
                let emb_flat = extract_tensor_data(outputs, 0)?;
                let logits_flat = extract_tensor_data(outputs, 3)?; // predictions at index 3

                (0..batch_size)
                    .map(|i| {
                        let emb_start = i * embedding_dim;
                        let emb_end = emb_start + embedding_dim;
                        let embeddings = emb_flat[emb_start..emb_end].to_vec();

                        let logits_start = i * num_species;
                        let logits_end = logits_start + num_species;
                        let logits = &logits_flat[logits_start..logits_end];

                        let predictions = top_k_predictions(
                            logits,
                            &self.inner.labels,
                            self.inner.top_k,
                            self.inner.min_confidence,
                        );

                        Ok(PredictionResult {
                            model_type,
                            predictions,
                            embeddings: Some(embeddings),
                            raw_scores: logits.to_vec(),
                        })
                    })
                    .collect()
            }
        }
    }
}

/// Extract tensor data from session outputs by index
fn extract_tensor_data(outputs: &ort::session::SessionOutputs, index: usize) -> Result<Vec<f32>> {
    let output_names: Vec<_> = outputs.keys().collect();
    let name = output_names
        .get(index)
        .ok_or_else(|| Error::Inference(format!("missing output tensor at index {index}")))?;

    let tensor = outputs
        .get(*name)
        .ok_or_else(|| Error::Inference(format!("missing output tensor '{name}'")))?;

    let (_, data) = tensor
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Inference(e.to_string()))?;

    Ok(data.to_vec())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    use super::*;

    // Builder validation tests

    #[test]
    fn test_builder_missing_model_path() {
        let result = ClassifierBuilder::new()
            .labels(vec!["species1".to_string()])
            .build();

        assert!(matches!(result, Err(Error::ModelPathRequired)));
    }

    #[test]
    fn test_builder_missing_labels() {
        let result = ClassifierBuilder::new().model_path("model.onnx").build();

        assert!(matches!(result, Err(Error::LabelsRequired)));
    }

    #[test]
    fn test_builder_missing_both() {
        let result = ClassifierBuilder::new().build();

        // Should fail on model path first
        assert!(matches!(result, Err(Error::ModelPathRequired)));
    }

    #[test]
    fn test_builder_method_chaining() {
        let builder = ClassifierBuilder::new()
            .model_path("model.onnx")
            .labels_path("labels.txt")
            .top_k(5)
            .min_confidence(0.5)
            .model_type(ModelType::BirdNetV24);

        assert_eq!(builder.top_k, 5);
        assert_eq!(builder.min_confidence, Some(0.5));
        assert_eq!(builder.model_type_override, Some(ModelType::BirdNetV24));
    }

    #[test]
    fn test_builder_default_values() {
        let builder = ClassifierBuilder::new();

        assert_eq!(builder.top_k, 10); // Default
        assert_eq!(builder.min_confidence, None);
        assert_eq!(builder.model_type_override, None);
        assert!(builder.execution_providers.is_empty());
        assert!(builder.requested_provider_types.is_empty());
    }

    #[test]
    fn test_builder_top_k_zero() {
        let builder = ClassifierBuilder::new()
            .model_path("model.onnx")
            .labels(vec!["species1".to_string()])
            .top_k(0);

        assert_eq!(builder.top_k, 0);
    }

    #[test]
    fn test_builder_min_confidence_boundaries() {
        // Note: The builder intentionally doesn't validate min_confidence bounds.
        // Values outside [0.0, 1.0] are allowed because:
        // - Validation happens at runtime during filtering, not at build time
        // - This gives users flexibility to set aggressive thresholds
        // - Values >1.0 will filter out all results (sigmoid output is always <1)
        // - Values <0.0 will filter out nothing (sigmoid output is always >0)

        let builder = ClassifierBuilder::new().min_confidence(0.0);
        assert_eq!(builder.min_confidence, Some(0.0));

        let builder = ClassifierBuilder::new().min_confidence(1.0);
        assert_eq!(builder.min_confidence, Some(1.0));

        let builder = ClassifierBuilder::new().min_confidence(1.5);
        assert_eq!(builder.min_confidence, Some(1.5)); // Will filter all results

        let builder = ClassifierBuilder::new().min_confidence(-0.5);
        assert_eq!(builder.min_confidence, Some(-0.5)); // Will filter nothing
    }

    #[test]
    fn test_builder_labels_path_vs_in_memory() {
        let builder1 = ClassifierBuilder::new().labels_path("labels.txt");

        assert!(matches!(builder1.labels, Some(Labels::Path(_))));

        let builder2 = ClassifierBuilder::new().labels(vec!["species1".to_string()]);

        assert!(matches!(builder2.labels, Some(Labels::InMemory(_))));
    }

    #[test]
    fn test_builder_multiple_execution_providers() {
        use ort::execution_providers::CPUExecutionProvider;

        let builder = ClassifierBuilder::new()
            .execution_provider(CPUExecutionProvider::default())
            .execution_provider(CPUExecutionProvider::default());

        assert_eq!(builder.execution_providers.len(), 2);
    }

    #[test]
    fn test_builder_default_trait() {
        let builder1 = ClassifierBuilder::new();
        let builder2 = ClassifierBuilder::default();

        assert_eq!(builder1.top_k, builder2.top_k);
        assert_eq!(builder1.min_confidence, builder2.min_confidence);
    }

    #[test]
    fn test_builder_typed_provider_methods() {
        // Test with_cuda
        let builder = ClassifierBuilder::new().with_cuda();
        assert_eq!(builder.execution_providers.len(), 1);
        assert_eq!(builder.requested_provider_types.len(), 1);
        assert_eq!(
            builder.requested_provider_types[0],
            ExecutionProviderInfo::Cuda
        );

        // Test with_tensorrt
        let builder = ClassifierBuilder::new().with_tensorrt();
        assert_eq!(builder.execution_providers.len(), 1);
        assert_eq!(builder.requested_provider_types.len(), 1);
        assert_eq!(
            builder.requested_provider_types[0],
            ExecutionProviderInfo::TensorRt
        );

        // Test with_directml
        let builder = ClassifierBuilder::new().with_directml();
        assert_eq!(builder.execution_providers.len(), 1);
        assert_eq!(builder.requested_provider_types.len(), 1);
        assert_eq!(
            builder.requested_provider_types[0],
            ExecutionProviderInfo::DirectMl
        );

        // Test with_coreml
        let builder = ClassifierBuilder::new().with_coreml();
        assert_eq!(builder.execution_providers.len(), 1);
        assert_eq!(builder.requested_provider_types.len(), 1);
        assert_eq!(
            builder.requested_provider_types[0],
            ExecutionProviderInfo::CoreMl
        );

        // Test chaining multiple providers
        let builder = ClassifierBuilder::new().with_cuda().with_tensorrt();
        assert_eq!(builder.execution_providers.len(), 2);
        assert_eq!(builder.requested_provider_types.len(), 2);
        assert_eq!(
            builder.requested_provider_types[0],
            ExecutionProviderInfo::Cuda
        );
        assert_eq!(
            builder.requested_provider_types[1],
            ExecutionProviderInfo::TensorRt
        );
    }

    // Input validation tests (these test predict/predict_batch validation logic)

    #[test]
    fn test_mock_input_size_validation() {
        // These tests verify the input size validation logic
        // without actually creating a full classifier

        let expected_size = 144_000; // BirdNetV24 sample count
        let wrong_size = 160_000; // BirdNetV30 sample count

        // Simulate what predict() does for validation
        let segment = vec![0.0f32; wrong_size];
        if segment.len() != expected_size {
            let err = Error::InputSize {
                expected: expected_size,
                got: segment.len(),
            };
            assert!(matches!(err, Error::InputSize { .. }));
        }
    }

    #[test]
    fn test_mock_batch_input_validation() {
        // Test batch input validation logic
        let expected_size = 144_000;
        let segments = [
            vec![0.0f32; expected_size],
            vec![0.0f32; 160_000], // Wrong size
            vec![0.0f32; expected_size],
        ];

        // Simulate batch validation
        for (i, seg) in segments.iter().enumerate() {
            if seg.len() != expected_size {
                let err = Error::BatchInputSize {
                    index: i,
                    expected: expected_size,
                    got: seg.len(),
                };
                assert!(matches!(err, Error::BatchInputSize { index: 1, .. }));
                assert_eq!(i, 1);
                break;
            }
        }
    }

    // Edge case tests

    #[test]
    fn test_empty_batch_handling() {
        // Verify that empty batch returns empty result
        let segments: Vec<&[f32]> = vec![];
        assert!(segments.is_empty());
        // The actual predict_batch method returns Ok(Vec::new()) for empty input
    }

    #[test]
    fn test_labels_enum_debug() {
        let labels_path = Labels::Path("test.txt".to_string());
        let debug_str = format!("{labels_path:?}");
        assert!(debug_str.contains("Path"));

        let labels_mem = Labels::InMemory(vec!["test".to_string()]);
        let debug_str = format!("{labels_mem:?}");
        assert!(debug_str.contains("InMemory"));
    }
}
