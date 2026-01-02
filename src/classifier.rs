//! Classifier builder and implementation

use crate::detection::detect_model_type;
use crate::error::{Error, Result};
use crate::labels::load_labels_from_file;
use crate::postprocess::top_k_predictions;
use crate::types::{ModelConfig, ModelType, PredictionResult};
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
            ModelType::BirdNetV30 | ModelType::PerchV2 => {
                // Two outputs: embeddings, predictions
                let embeddings = extract_tensor_data(outputs, 0)?;
                let logits = extract_tensor_data(outputs, 1)?;
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
            ModelType::BirdNetV30 | ModelType::PerchV2 => {
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

// Unit tests for classifier are in task-08 integration tests
// since they require actual ONNX models
