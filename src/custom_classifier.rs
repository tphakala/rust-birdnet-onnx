//! Custom classifier for running secondary models on embedding vectors.

use crate::error::{Error, Result};
use crate::labels;
use crate::postprocess::top_k_predictions;
use crate::types::Prediction;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A lightweight classifier that runs on embedding vectors from a primary model.
///
/// Used for custom classification heads (e.g., bat species detection from
/// `BirdNET` embeddings).
#[derive(Debug)]
pub struct CustomClassifier {
    session: Mutex<Session>,
    labels: Vec<String>,
    input_dim: usize,
    num_classes: usize,
    top_k: usize,
    min_confidence: Option<f32>,
}

/// Builder for [`CustomClassifier`].
#[derive(Debug, Default)]
pub struct CustomClassifierBuilder {
    model_path: Option<PathBuf>,
    labels_path: Option<PathBuf>,
    top_k: Option<usize>,
    min_confidence: Option<f32>,
}

impl CustomClassifierBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ONNX model path.
    #[must_use]
    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the labels file path (one label per line).
    #[must_use]
    pub fn labels_path(mut self, path: impl AsRef<Path>) -> Self {
        self.labels_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the number of top predictions to return (default: all classes).
    #[must_use]
    pub const fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub const fn min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = Some(threshold);
        self
    }

    /// Build the custom classifier.
    ///
    /// # Errors
    ///
    /// Returns an error if model path or labels path are not set, the model
    /// cannot be loaded, or the label count does not match the model output.
    pub fn build(self) -> Result<CustomClassifier> {
        let model_path = self.model_path.ok_or(Error::ModelPathRequired)?;
        let labels_path = self.labels_path.ok_or(Error::LabelsRequired)?;

        let session = Session::builder()
            .map_err(Error::ModelLoad)?
            .commit_from_file(&model_path)
            .map_err(Error::ModelLoad)?;

        let input_dim = extract_last_dim(session.inputs(), "input")?;
        let num_classes = extract_last_dim(session.outputs(), "output")?;

        let content = std::fs::read_to_string(&labels_path).map_err(|e| Error::LabelLoad {
            path: labels_path.display().to_string(),
            reason: e.to_string(),
        })?;
        let labels = labels::parse_labels(&content, crate::types::LabelFormat::Text)?;

        if labels.len() != num_classes {
            return Err(Error::LabelCount {
                expected: num_classes,
                got: labels.len(),
            });
        }

        let top_k = self.top_k.unwrap_or(num_classes);

        Ok(CustomClassifier {
            session: Mutex::new(session),
            labels,
            input_dim,
            num_classes,
            top_k,
            min_confidence: self.min_confidence,
        })
    }
}

impl CustomClassifier {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> CustomClassifierBuilder {
        CustomClassifierBuilder::new()
    }

    /// Classify a single embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding length does not match the expected
    /// input dimension, the input tensor cannot be created, or inference fails.
    #[allow(clippy::significant_drop_tightening)] // outputs borrows from session; early drop is not possible
    pub fn predict(&self, embeddings: &[f32]) -> Result<Vec<Prediction>> {
        if embeddings.len() != self.input_dim {
            return Err(Error::EmbeddingDimMismatch {
                expected: self.input_dim,
                got: embeddings.len(),
            });
        }

        let input = Array2::from_shape_vec((1, self.input_dim), embeddings.to_vec())
            .map_err(|e| Error::Inference(format!("failed to create input array: {e}")))?;

        let input_value = Value::from_array(input)
            .map_err(|e| Error::Inference(format!("failed to create input tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Inference(format!("session lock poisoned: {e}")))?;

        // outputs borrows from session, so session must stay alive until extraction is done
        let outputs = session
            .run(ort::inputs![input_value.view()])
            .map_err(|e| Error::Inference(e.to_string()))?;

        let logits = extract_tensor_data(&outputs, 0, self.num_classes)?;

        Ok(top_k_predictions(
            &logits,
            &self.labels,
            self.top_k,
            self.min_confidence,
        ))
    }

    /// Classify a batch of embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if any embedding length does not match the expected
    /// input dimension, the input tensor cannot be created, or inference fails.
    #[allow(clippy::significant_drop_tightening)] // outputs borrows from session; early drop is not possible
    pub fn predict_batch(
        &self,
        embeddings_batch: &[Vec<f32>],
    ) -> Result<Vec<Vec<Prediction>>> {
        if embeddings_batch.is_empty() {
            return Ok(Vec::new());
        }

        for emb in embeddings_batch {
            if emb.len() != self.input_dim {
                return Err(Error::EmbeddingDimMismatch {
                    expected: self.input_dim,
                    got: emb.len(),
                });
            }
        }

        let batch_size = embeddings_batch.len();
        let flat: Vec<f32> = embeddings_batch
            .iter()
            .flat_map(|e| e.iter().copied())
            .collect();

        let input =
            Array2::from_shape_vec((batch_size, self.input_dim), flat)
                .map_err(|e| Error::Inference(format!("failed to create batch input: {e}")))?;

        let input_value = Value::from_array(input)
            .map_err(|e| Error::Inference(format!("failed to create input tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| Error::Inference(format!("session lock poisoned: {e}")))?;

        // outputs borrows from session, so session must stay alive until extraction is done
        let outputs = session
            .run(ort::inputs![input_value.view()])
            .map_err(|e| Error::Inference(e.to_string()))?;

        let all_logits =
            extract_tensor_data(&outputs, 0, batch_size * self.num_classes)?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * self.num_classes;
            let end = start + self.num_classes;
            let logits = &all_logits[start..end];

            results.push(top_k_predictions(
                logits,
                &self.labels,
                self.top_k,
                self.min_confidence,
            ));
        }

        Ok(results)
    }

    /// Return the labels loaded from the labels file.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Return the embedding dimension this classifier expects.
    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Return the number of output classes.
    #[must_use]
    pub const fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Extract the last dimension of the first tensor in a slice of outlets.
///
/// `role` is used only in error messages ("input" or "output").
fn extract_last_dim(outlets: &[ort::value::Outlet], role: &str) -> Result<usize> {
    let info = outlets
        .first()
        .ok_or_else(|| Error::Inference(format!("custom classifier has no {role}s")))?;

    let shape = info.dtype().tensor_shape().ok_or_else(|| {
        Error::Inference(format!("custom classifier {role} is not a tensor"))
    })?;

    let last = shape.last().copied().ok_or_else(|| {
        Error::Inference(format!("custom classifier {role} has empty shape"))
    })?;

    if last < 0 {
        return Err(Error::Inference(format!(
            "custom classifier {role} has dynamic last dimension"
        )));
    }

    usize::try_from(last).map_err(|_| {
        Error::Inference(format!("custom classifier {role} dimension overflows usize"))
    })
}

/// Extract flat f32 data from session outputs at the given index.
fn extract_tensor_data(
    outputs: &ort::session::SessionOutputs,
    index: usize,
    expected_len: usize,
) -> Result<Vec<f32>> {
    let name = outputs
        .keys()
        .nth(index)
        .ok_or_else(|| Error::Inference(format!("missing output tensor at index {index}")))?;

    let tensor = outputs
        .get(name)
        .ok_or_else(|| Error::Inference(format!("missing output tensor '{name}'")))?;

    let (_, data) = tensor
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Inference(format!("failed to extract output tensor: {e}")))?;

    let flat = data.to_vec();

    if flat.len() < expected_len {
        return Err(Error::Inference(format!(
            "output tensor too small: expected {expected_len}, got {}",
            flat.len()
        )));
    }

    Ok(flat)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_builder_requires_model_path() {
        let result = CustomClassifierBuilder::new()
            .labels_path("/some/labels.txt")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("model path"));
    }

    #[test]
    fn test_builder_requires_labels() {
        let result = CustomClassifierBuilder::new()
            .model_path("/some/model.onnx")
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("labels"));
    }

    #[test]
    fn test_embedding_dim_mismatch_error() {
        let err = Error::EmbeddingDimMismatch {
            expected: 1024,
            got: 512,
        };
        assert_eq!(
            err.to_string(),
            "embedding dimension mismatch: classifier expects 1024, got 512"
        );
    }
}
