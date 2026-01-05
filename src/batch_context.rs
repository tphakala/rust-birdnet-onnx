//! Batch inference context for GPU memory reuse.
//!
//! This module provides [`BatchInferenceContext`] for efficient repeated batch inference
//! on GPU. By pre-allocating input and output buffers, memory is reused across inference
//! calls, preventing the memory growth issues that can occur with standard batch inference.
//!
//! # When to Use
//!
//! Use `BatchInferenceContext` when:
//! - Processing many batches of audio segments on GPU
//! - Experiencing memory growth with repeated `predict_batch()` calls
//! - Running on systems with limited VRAM
//!
//! # Example
//!
//! ```ignore
//! use birdnet_onnx::{Classifier, InferenceOptions};
//!
//! let classifier = Classifier::builder()
//!     .model_path("model.onnx")
//!     .labels_path("labels.txt")
//!     .with_cuda()
//!     .build()?;
//!
//! // Create context with max batch size of 32
//! let mut ctx = classifier.create_batch_context(32)?;
//!
//! // Process multiple batches - memory is reused
//! for chunk in audio_segments.chunks(32) {
//!     let results = classifier.predict_batch_with_context(
//!         &mut ctx,
//!         chunk,
//!         &InferenceOptions::default(),
//!     )?;
//! }
//! ```

use crate::error::{Error, Result};
use crate::types::{ModelConfig, ModelType};
use ort::io_binding::IoBinding;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::MutexGuard;

/// Pre-allocated buffers for efficient repeated batch inference.
///
/// This struct holds GPU memory that is reused across multiple inference calls,
/// preventing memory growth issues with repeated batch processing.
///
/// # Memory Management
///
/// The context pre-allocates buffers for the maximum batch size specified at creation.
/// Smaller batches can be processed without reallocation - only the relevant portion
/// of the output is used.
///
/// # Thread Safety
///
/// `BatchInferenceContext` is **not** thread-safe. Each thread should create its own
/// context if parallel processing is needed. The CUDA allocator used internally may
/// not be safe to share across threads.
///
/// # Supported Models
///
/// Currently supports:
/// - `BirdNET` v2.4 (1 output)
/// - `BirdNET` v3.0 (2 outputs)
///
/// `Perch` v2 is not yet supported due to its complex 4-output structure.
#[derive(Debug)]
pub struct BatchInferenceContext {
    /// `IoBinding` for the session
    pub(crate) io_binding: IoBinding,
    /// Pre-allocated input buffer to avoid per-batch allocations
    input_buffer: Vec<f32>,
    /// Maximum batch size this context supports
    max_batch_size: usize,
    /// Sample count per segment
    sample_count: usize,
    /// Number of species in model output
    num_species: usize,
    /// Embedding dimension (if applicable)
    embedding_dim: Option<usize>,
    /// Model type for output processing
    model_type: ModelType,
}

impl BatchInferenceContext {
    /// Create a new batch inference context with pre-allocated buffers.
    ///
    /// # Arguments
    ///
    /// * `session` - Locked session reference
    /// * `config` - Model configuration
    /// * `max_batch_size` - Maximum segments per batch
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model type is `PerchV2` (not yet supported)
    /// - Buffer allocation fails
    /// - `IoBinding` creation fails
    pub(crate) fn new(
        session: &MutexGuard<'_, Session>,
        config: &ModelConfig,
        max_batch_size: usize,
    ) -> Result<Self> {
        // PerchV2 has complex 4-output structure - not yet supported
        if config.model_type == ModelType::PerchV2 {
            return Err(Error::Inference(
                "BatchInferenceContext does not yet support PerchV2 models. \
                 Use predict_batch() instead."
                    .into(),
            ));
        }

        // Create IoBinding
        let io_binding = session
            .create_binding()
            .map_err(|e| Error::Inference(format!("failed to create IoBinding: {e}")))?;

        // Pre-allocate input buffer for max batch size
        let input_buffer = vec![0.0f32; max_batch_size * config.sample_count];

        Ok(Self {
            io_binding,
            input_buffer,
            max_batch_size,
            sample_count: config.sample_count,
            num_species: config.num_species,
            embedding_dim: config.embedding_dim,
            model_type: config.model_type,
        })
    }

    /// Returns the maximum batch size this context supports.
    #[must_use]
    pub const fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Returns the sample count per segment.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the pre-allocated input buffer capacity in samples.
    ///
    /// This buffer is reused across batch inference calls to avoid heap allocations.
    #[must_use]
    pub const fn input_buffer_capacity(&self) -> usize {
        self.input_buffer.len()
    }

    /// Returns the pre-allocated input buffer capacity in bytes.
    #[must_use]
    pub const fn input_buffer_bytes(&self) -> usize {
        self.input_buffer.len() * std::mem::size_of::<f32>()
    }

    /// Returns the model type this context was created for.
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Prepare input by filling the pre-allocated buffer and binding to `IoBinding`.
    ///
    /// This method:
    /// 1. Validates segment sizes
    /// 2. Copies audio data into the pre-allocated buffer
    /// 3. Creates a tensor reference from the buffer (zero-copy)
    /// 4. Binds the tensor to the `IoBinding`
    ///
    /// The input buffer is reused across calls, avoiding heap allocations.
    ///
    /// # Arguments
    ///
    /// * `segments` - Audio segments to process
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Batch size exceeds `max_batch_size`
    /// - Any segment has incorrect sample count
    /// - Tensor creation fails
    /// - Binding fails
    pub(crate) fn prepare_input(&mut self, segments: &[&[f32]]) -> Result<()> {
        let batch_size = segments.len();

        if batch_size > self.max_batch_size {
            return Err(Error::Inference(format!(
                "batch size {} exceeds context max {}",
                batch_size, self.max_batch_size
            )));
        }

        // Copy segment data into pre-allocated buffer
        for (i, segment) in segments.iter().enumerate() {
            if segment.len() != self.sample_count {
                return Err(Error::BatchInputSize {
                    index: i,
                    expected: self.sample_count,
                    got: segment.len(),
                });
            }

            let start = i * self.sample_count;
            let end = start + self.sample_count;
            self.input_buffer[start..end].copy_from_slice(segment);
        }

        // Create tensor reference from the filled portion of buffer (zero-copy)
        let input_ref = TensorRef::from_array_view((
            [batch_size, self.sample_count],
            &self.input_buffer[..batch_size * self.sample_count],
        ))
        .map_err(|e| Error::Inference(format!("failed to create input tensor: {e}")))?;

        // Bind to IoBinding
        self.io_binding
            .bind_input("input", &input_ref)
            .map_err(|e| Error::Inference(format!("failed to bind input: {e}")))?;

        Ok(())
    }

    /// Bind outputs to device memory.
    ///
    /// This tells ONNX Runtime to allocate outputs on the GPU and reuse them.
    pub(crate) fn bind_outputs_to_device(&mut self) -> Result<()> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

        // Use CPU memory for outputs so they can be read with try_extract_tensor.
        // IoBinding still provides memory reuse benefits - the allocator reuses
        // the same CPU buffers across inference calls.
        let mem_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| Error::Inference(format!("failed to create memory info: {e}")))?;

        match self.model_type {
            ModelType::BirdNetV24 => {
                // Single output: logits
                self.io_binding
                    .bind_output_to_device("output", &mem_info)
                    .map_err(|e| Error::Inference(format!("failed to bind output: {e}")))?;
            }
            ModelType::BirdNetV30 => {
                // Two outputs: embeddings and logits
                self.io_binding
                    .bind_output_to_device("output_0", &mem_info)
                    .map_err(|e| {
                        Error::Inference(format!("failed to bind embeddings output: {e}"))
                    })?;
                self.io_binding
                    .bind_output_to_device("output_1", &mem_info)
                    .map_err(|e| Error::Inference(format!("failed to bind logits output: {e}")))?;
            }
            ModelType::PerchV2 => {
                // Should not reach here - checked in new()
                return Err(Error::Inference("PerchV2 not supported".into()));
            }
        }

        Ok(())
    }

    /// Synchronize outputs after inference.
    ///
    /// This ensures any pending asynchronous GPU operations are complete
    /// and helps the memory allocator properly track freed memory.
    pub(crate) fn synchronize(&self) -> Result<()> {
        self.io_binding
            .synchronize_outputs()
            .map_err(|e| Error::Inference(format!("failed to synchronize outputs: {e}")))?;
        Ok(())
    }

    /// Clear bound inputs to prepare for next batch.
    pub(crate) fn clear_inputs(&mut self) {
        self.io_binding.clear_inputs();
    }

    /// Extract output tensor data from `IoBinding` results.
    pub(crate) fn extract_outputs(
        &self,
        outputs: &ort::session::SessionOutputs<'_>,
        batch_size: usize,
    ) -> Result<(Option<Vec<f32>>, Vec<f32>)> {
        match self.model_type {
            ModelType::BirdNetV24 => {
                let logits =
                    Self::extract_tensor_data(outputs, "output", batch_size * self.num_species)?;
                Ok((None, logits))
            }
            ModelType::BirdNetV30 => {
                let embedding_dim = self.embedding_dim.ok_or_else(|| {
                    Error::Inference("embedding_dim required for BirdNetV30".into())
                })?;
                let embeddings =
                    Self::extract_tensor_data(outputs, "output_0", batch_size * embedding_dim)?;
                let logits =
                    Self::extract_tensor_data(outputs, "output_1", batch_size * self.num_species)?;
                Ok((Some(embeddings), logits))
            }
            ModelType::PerchV2 => Err(Error::Inference("PerchV2 not supported".into())),
        }
    }

    /// Extract tensor data from outputs by name.
    fn extract_tensor_data(
        outputs: &ort::session::SessionOutputs<'_>,
        name: &str,
        expected_len: usize,
    ) -> Result<Vec<f32>> {
        let tensor = outputs
            .get(name)
            .ok_or_else(|| Error::Inference(format!("missing output tensor '{name}'")))?;

        let (_, data) = tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Inference(e.to_string()))?;

        // Convert to vec for processing
        let data_vec = data.to_vec();

        if data_vec.len() < expected_len {
            return Err(Error::Inference(format!(
                "output tensor '{name}' too short: expected at least {expected_len}, got {}",
                data_vec.len()
            )));
        }

        // Take only the data for actual batch size (may be smaller than max)
        Ok(data_vec[..expected_len].to_vec())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_model_type_sample_counts() {
        // Verify model type sample counts are as expected
        assert_eq!(ModelType::BirdNetV24.sample_count(), 144_000);
        assert_eq!(ModelType::BirdNetV30.sample_count(), 160_000);
        assert_eq!(ModelType::PerchV2.sample_count(), 160_000);
    }

    #[test]
    fn test_model_type_has_embeddings() {
        assert!(!ModelType::BirdNetV24.has_embeddings());
        assert!(ModelType::BirdNetV30.has_embeddings());
        assert!(ModelType::PerchV2.has_embeddings());
    }
}
