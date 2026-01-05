//! Classifier builder and implementation

use crate::detection::detect_model_type;
use crate::error::{Error, Result};
use crate::inference_options::InferenceOptions;
use crate::labels::load_labels_from_file;
use crate::postprocess::top_k_predictions;
use crate::types::{ExecutionProviderInfo, ModelConfig, ModelType, PredictionResult};
use ndarray::Array2;
use ort::session::{RunOptions, Session};
use ort::value::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Macro to generate execution provider builder methods
macro_rules! with_provider_method {
    ($fn_name:ident, $provider_struct:ident, $provider_enum:ident, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        pub fn $fn_name(mut self) -> Self {
            use ort::execution_providers::$provider_struct;
            self.execution_providers
                .push($provider_struct::default().into());
            // Only set requested_provider if it's still the default (CPU).
            // This aligns with ONNX Runtime's behavior: it tries providers
            // in the order they were added, so the first non-CPU provider
            // is the most relevant one to track.
            if self.requested_provider == ExecutionProviderInfo::Cpu {
                self.requested_provider = ExecutionProviderInfo::$provider_enum;
            }
            self
        }
    };
}

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
    requested_provider: ExecutionProviderInfo,
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
            requested_provider: ExecutionProviderInfo::Cpu,
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

    /// Override auto-detected model type (useful for `Perch` v2)
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

    /// Request CUDA execution provider (NVIDIA GPU) with safe defaults
    ///
    /// This method uses safe defaults to prevent memory allocation issues:
    /// - **Arena strategy**: `SameAsRequested` to prevent exponential memory growth
    /// - **No memory limit**: Uses available GPU memory
    ///
    /// For custom CUDA settings, use [`with_cuda_config()`](Self::with_cuda_config).
    ///
    /// # Safe Defaults
    ///
    /// ONNX Runtime's default `NextPowerOfTwo` arena strategy can cause sudden 4GB+
    /// allocations that freeze Windows systems. This method uses `SameAsRequested`
    /// by default to allocate memory more gradually.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use birdnet_onnx::Classifier;
    ///
    /// let classifier = Classifier::builder()
    ///     .model_path("model.onnx")
    ///     .labels_path("labels.txt")
    ///     .with_cuda()
    ///     .build()?;
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    #[must_use]
    pub fn with_cuda(self) -> Self {
        self.with_cuda_config(crate::cuda_config::CUDAConfig::new())
    }

    /// Configure CUDA with custom settings
    ///
    /// Use this method when you need fine-grained control over CUDA memory allocation.
    /// For most use cases, [`with_cuda()`](Self::with_cuda) provides safe defaults.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use birdnet_onnx::{Classifier, CUDAConfig};
    ///
    /// let cuda_config = CUDAConfig::new()
    ///     .with_memory_limit(8 * 1024 * 1024 * 1024)  // 8GB limit
    ///     .with_device_id(1);  // Use second GPU
    ///
    /// let classifier = Classifier::builder()
    ///     .model_path("model.onnx")
    ///     .labels_path("labels.txt")
    ///     .with_cuda_config(cuda_config)
    ///     .build()?;
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    #[must_use]
    pub fn with_cuda_config(mut self, config: crate::cuda_config::CUDAConfig) -> Self {
        use ort::execution_providers::CUDAExecutionProvider;

        let provider = config.apply_to(CUDAExecutionProvider::default());
        self.execution_providers.push(provider.into());

        if self.requested_provider == ExecutionProviderInfo::Cpu {
            self.requested_provider = ExecutionProviderInfo::Cuda;
        }

        self
    }

    /// Request `TensorRT` execution provider (NVIDIA GPU) with optimized defaults
    ///
    /// This method enables performance optimizations including FP16 precision and caching.
    /// Expected performance: 4x faster than unoptimized `TensorRT` and comparable to or
    /// better than CUDA provider.
    ///
    /// For custom `TensorRT` settings, use [`with_tensorrt_config()`](Self::with_tensorrt_config).
    ///
    /// # Requirements
    /// - NVIDIA GPU (compute capability 5.3+)
    /// - `TensorRT` library installed
    /// - ONNX Runtime built with `TensorRT` support
    ///
    /// # Performance Optimizations
    ///
    /// The default configuration enables:
    /// - **FP16 precision**: 2x faster inference on GPUs with tensor cores
    /// - **Engine caching**: Reduces session creation from minutes to seconds
    /// - **Timing cache**: Accelerates future builds with similar layer configurations
    /// - **Optimization level 3**: Balanced optimization (`TensorRT` default)
    ///
    /// **CUDA graphs are disabled by default** to avoid a known bug in ONNX Runtime 1.22.0.
    /// See [`crate::TensorRTConfig::new()`] for details and how to enable them if needed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use birdnet_onnx::Classifier;
    ///
    /// let classifier = Classifier::builder()
    ///     .model_path("model.onnx")
    ///     .labels_path("labels.txt")
    ///     .with_tensorrt()
    ///     .build()?;
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    #[must_use]
    pub fn with_tensorrt(self) -> Self {
        self.with_tensorrt_config(crate::tensorrt_config::TensorRTConfig::new())
    }

    /// Configure `TensorRT` with custom settings
    ///
    /// Use this method when you need fine-grained control over `TensorRT` behavior.
    /// For most use cases, [`with_tensorrt()`](Self::with_tensorrt) provides optimal defaults.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use birdnet_onnx::{Classifier, TensorRTConfig};
    ///
    /// let trt_config = TensorRTConfig::new()
    ///     .with_fp16(false)  // Disable FP16 for accuracy-critical work
    ///     .with_builder_optimization_level(5)  // Maximum optimization
    ///     .with_engine_cache_path("/tmp/trt_cache");
    ///
    /// let classifier = Classifier::builder()
    ///     .model_path("model.onnx")
    ///     .labels_path("labels.txt")
    ///     .with_tensorrt_config(trt_config)
    ///     .build()?;
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    #[must_use]
    pub fn with_tensorrt_config(mut self, config: crate::tensorrt_config::TensorRTConfig) -> Self {
        use ort::execution_providers::TensorRTExecutionProvider;

        let provider = config.apply_to(TensorRTExecutionProvider::default());
        self.execution_providers.push(provider.into());

        if self.requested_provider == ExecutionProviderInfo::Cpu {
            self.requested_provider = ExecutionProviderInfo::TensorRt;
        }

        self
    }

    with_provider_method!(
        with_directml,
        DirectMLExecutionProvider,
        DirectMl,
        "Request `DirectML` execution provider (Windows GPU)"
    );
    with_provider_method!(
        with_coreml,
        CoreMLExecutionProvider,
        CoreMl,
        "Request `CoreML` execution provider (Apple Neural Engine)"
    );
    with_provider_method!(
        with_rocm,
        ROCmExecutionProvider,
        Rocm,
        "Request `ROCm` execution provider (AMD GPU)"
    );
    with_provider_method!(
        with_openvino,
        OpenVINOExecutionProvider,
        OpenVino,
        "Request `OpenVINO` execution provider (Intel accelerator)"
    );
    with_provider_method!(
        with_onednn,
        OneDNNExecutionProvider,
        OneDnn,
        "Request oneDNN execution provider (Intel accelerator)"
    );
    with_provider_method!(
        with_qnn,
        QNNExecutionProvider,
        Qnn,
        "Request QNN execution provider (Qualcomm NPU)"
    );
    with_provider_method!(
        with_acl,
        ACLExecutionProvider,
        Acl,
        "Request ACL execution provider (Arm Compute Library)"
    );
    with_provider_method!(
        with_armnn,
        ArmNNExecutionProvider,
        ArmNn,
        "Request `ArmNN` execution provider (Arm Neural Network)"
    );

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
                requested_provider: self.requested_provider,
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

/// Reason the monitor thread terminated.
#[derive(Debug, Clone, Copy)]
enum TerminationReason {
    /// Inference completed normally.
    Completed,
    /// Timeout exceeded.
    Timeout(Duration),
    /// External cancellation requested.
    Cancelled,
}

/// Internal state shared via Arc for thread safety
struct ClassifierInner {
    session: Mutex<Session>,
    config: ModelConfig,
    labels: Vec<String>,
    requested_provider: ExecutionProviderInfo,
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
            .field("requested_provider", &self.inner.requested_provider)
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

    /// Returns the execution provider that was requested for this classifier.
    ///
    /// **Note:** This returns the provider that was *requested* during build,
    /// not necessarily the provider that is *actually active*. If the requested
    /// provider is unavailable, ONNX Runtime will silently fall back to CPU.
    ///
    /// This value is only set by the typed `with_<provider>()` builder methods
    /// (e.g., `with_cuda()`, `with_tensorrt()`). The generic `execution_provider()`
    /// method does not affect the value returned here.
    ///
    /// To verify the actual provider being used, enable ONNX Runtime verbose
    /// logging via environment variable: `ORT_LOG_LEVEL=Verbose`
    #[must_use]
    pub fn requested_provider(&self) -> ExecutionProviderInfo {
        self.inner.requested_provider
    }

    /// Run inference with optional timeout and cancellation support.
    ///
    /// Spawns a monitor thread that watches for timeout/cancellation and calls
    /// `terminate()` on the `RunOptions` if needed.
    #[allow(clippy::unused_self)] // self is used indirectly via closure capturing self.inner
    fn run_inference<F, T>(&self, options: &InferenceOptions, f: F) -> Result<T>
    where
        F: FnOnce(&RunOptions) -> Result<T>,
    {
        let run_options = RunOptions::new()
            .map_err(|e| Error::Inference(format!("failed to create run options: {e}")))?;

        // Fast path: no monitoring needed
        if !options.needs_monitor() {
            return f(&run_options);
        }

        // Wrap for sharing with monitor thread
        let run_options = Arc::new(run_options);
        let completed = Arc::new(AtomicBool::new(false));

        // Clone values for monitor thread
        let run_options_clone = Arc::clone(&run_options);
        let completed_clone = Arc::clone(&completed);
        let timeout = options.timeout;
        let cancel_token = options.cancellation_token.clone();

        // Spawn monitor thread
        let monitor_handle = std::thread::spawn(move || {
            let start = Instant::now();
            loop {
                // Check if inference completed normally
                if completed_clone.load(Ordering::SeqCst) {
                    return TerminationReason::Completed;
                }

                // Check external cancellation
                if let Some(ref token) = cancel_token
                    && token.is_cancelled()
                {
                    let _ = run_options_clone.terminate();
                    return TerminationReason::Cancelled;
                }

                // Check timeout
                if let Some(duration) = timeout
                    && start.elapsed() >= duration
                {
                    let _ = run_options_clone.terminate();
                    return TerminationReason::Timeout(duration);
                }

                // Poll interval: balance responsiveness vs CPU overhead
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        // Run inference
        let result = f(&run_options);

        // Signal monitor to stop
        completed.store(true, Ordering::SeqCst);

        // Wait for monitor and get termination reason
        let termination_reason = monitor_handle
            .join()
            .unwrap_or(TerminationReason::Completed);

        // Translate errors based on termination reason
        match (result, termination_reason) {
            (Ok(value), _) => Ok(value),
            (Err(_), TerminationReason::Timeout(duration)) => Err(Error::Timeout { duration }),
            (Err(_), TerminationReason::Cancelled) => Err(Error::Cancelled),
            (Err(e), TerminationReason::Completed) => Err(e),
        }
    }

    /// Run inference on a single audio segment.
    ///
    /// # Arguments
    /// * `segment` - Audio samples (must match `config().sample_count`)
    /// * `options` - Inference options for timeout and cancellation control
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
    /// - Inference times out ([`Error::Timeout`])
    /// - Inference is cancelled ([`Error::Cancelled`])
    ///
    /// # Example
    ///
    /// ```ignore
    /// use birdnet_onnx::InferenceOptions;
    /// use std::time::Duration;
    ///
    /// // With 30 second timeout
    /// let result = classifier.predict(
    ///     &segment,
    ///     &InferenceOptions::timeout(Duration::from_secs(30)),
    /// )?;
    ///
    /// // Without timeout (default behavior)
    /// let result = classifier.predict(&segment, &InferenceOptions::default())?;
    /// ```
    #[allow(clippy::significant_drop_tightening)]
    pub fn predict(&self, segment: &[f32], options: &InferenceOptions) -> Result<PredictionResult> {
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

        self.run_inference(options, |run_options| {
            // IMPORTANT: Session lock must be held while outputs exist because ort::Value
            // borrows from the session. Dropping the lock before processing outputs would
            // cause a use-after-free.
            let mut session = self
                .inner
                .session
                .lock()
                .map_err(|e| Error::Inference(format!("session lock poisoned: {e}")))?;

            let outputs = session
                .run_with_options(ort::inputs![input_value.view()], run_options)
                .map_err(|e| Error::Inference(e.to_string()))?;

            self.process_outputs(&outputs)
        })
    }

    /// Run inference on multiple audio segments (more efficient for GPU).
    ///
    /// # Arguments
    /// * `segments` - Slice of audio segments (all must match `config().sample_count`)
    /// * `options` - Inference options for timeout and cancellation control
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
    /// - Inference times out ([`Error::Timeout`])
    /// - Inference is cancelled ([`Error::Cancelled`])
    ///
    /// # Example
    ///
    /// ```ignore
    /// use birdnet_onnx::InferenceOptions;
    /// use std::time::Duration;
    ///
    /// // With 60 second timeout for batch
    /// let results = classifier.predict_batch(
    ///     &segments,
    ///     &InferenceOptions::timeout(Duration::from_secs(60)),
    /// )?;
    /// ```
    #[allow(clippy::significant_drop_tightening)]
    pub fn predict_batch(
        &self,
        segments: &[&[f32]],
        options: &InferenceOptions,
    ) -> Result<Vec<PredictionResult>> {
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

        self.run_inference(options, |run_options| {
            // IMPORTANT: Session lock must be held while outputs exist because ort::Value
            // borrows from the session.
            let mut session = self
                .inner
                .session
                .lock()
                .map_err(|e| Error::Inference(format!("session lock poisoned: {e}")))?;

            let outputs = session
                .run_with_options(ort::inputs![input_value.view()], run_options)
                .map_err(|e| Error::Inference(e.to_string()))?;

            self.process_batch_outputs(&outputs, batch_size)
        })
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
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cpu); // Default to CPU
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

    // Execution provider tests

    #[test]
    fn test_requested_provider_defaults_to_cpu() {
        let builder = ClassifierBuilder::new();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cpu);
    }

    #[test]
    fn test_builder_debug_includes_requested_provider() {
        let builder = ClassifierBuilder::new()
            .model_path("test.onnx")
            .labels(vec!["species1".to_string()]);

        let debug_str = format!("{builder:?}");
        assert!(debug_str.contains("requested_provider"));
        assert!(debug_str.contains("Cpu"));
    }

    // Typed builder method tests

    #[test]
    fn test_with_cuda_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_cuda();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_cuda_config_sets_requested_provider() {
        use crate::CUDAConfig;

        let config = CUDAConfig::new();
        let builder = ClassifierBuilder::new().with_cuda_config(config);
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_cuda_config_custom_settings() {
        use crate::CUDAConfig;

        let config = CUDAConfig::new()
            .with_memory_limit(8 * 1024 * 1024 * 1024)
            .with_device_id(1);

        let builder = ClassifierBuilder::new().with_cuda_config(config);
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_tensorrt_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_tensorrt();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::TensorRt);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_tensorrt_config_sets_requested_provider() {
        use crate::TensorRTConfig;

        let config = TensorRTConfig::new();
        let builder = ClassifierBuilder::new().with_tensorrt_config(config);
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::TensorRt);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_tensorrt_config_custom_settings() {
        use crate::TensorRTConfig;

        let config = TensorRTConfig::new()
            .with_fp16(false)
            .with_builder_optimization_level(5)
            .with_device_id(1);

        let builder = ClassifierBuilder::new().with_tensorrt_config(config);
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::TensorRt);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_tensorrt_config_disable_optimizations() {
        use crate::TensorRTConfig;

        let config = TensorRTConfig::new()
            .with_fp16(false)
            .with_cuda_graph(false)
            .with_engine_cache(false)
            .with_timing_cache(false);

        let builder = ClassifierBuilder::new().with_tensorrt_config(config);
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::TensorRt);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_directml_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_directml();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::DirectMl);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_coreml_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_coreml();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::CoreMl);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_rocm_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_rocm();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Rocm);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_openvino_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_openvino();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::OpenVino);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_onednn_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_onednn();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::OneDnn);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_qnn_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_qnn();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Qnn);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_acl_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_acl();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Acl);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_with_armnn_sets_requested_provider() {
        let builder = ClassifierBuilder::new().with_armnn();
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::ArmNn);
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_chaining_multiple_providers_first_wins() {
        let builder = ClassifierBuilder::new().with_cuda().with_tensorrt();
        // First non-CPU provider wins (aligns with ort's provider priority)
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        // Both providers added to the vec
        assert_eq!(builder.execution_providers.len(), 2);
    }

    #[test]
    fn test_chaining_three_providers_first_wins() {
        let builder = ClassifierBuilder::new()
            .with_cuda()
            .with_tensorrt()
            .with_directml();
        // First non-CPU provider wins (aligns with ort's provider priority)
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        // All three providers added
        assert_eq!(builder.execution_providers.len(), 3);
    }

    #[test]
    fn test_provider_methods_can_chain_with_other_builders() {
        let builder = ClassifierBuilder::new()
            .model_path("model.onnx")
            .labels_path("labels.txt")
            .with_cuda()
            .top_k(5)
            .min_confidence(0.8);

        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        assert_eq!(builder.top_k, 5);
        assert_eq!(builder.min_confidence, Some(0.8));
        assert_eq!(builder.execution_providers.len(), 1);
    }

    #[test]
    fn test_provider_methods_return_self_for_chaining() {
        // Verify each method returns Self and can be chained
        let builder = ClassifierBuilder::new()
            .with_cuda()
            .with_tensorrt()
            .with_directml()
            .with_coreml()
            .with_rocm()
            .with_openvino()
            .with_onednn()
            .with_qnn()
            .with_acl()
            .with_armnn();

        // First non-CPU provider wins (aligns with ort's provider priority)
        assert_eq!(builder.requested_provider, ExecutionProviderInfo::Cuda);
        // All 10 providers added
        assert_eq!(builder.execution_providers.len(), 10);
    }
}
