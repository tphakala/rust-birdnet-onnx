use thiserror::Error;

/// Errors that can occur during classifier operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Audio segment has wrong number of samples.
    #[error("input size mismatch: expected {expected} samples, got {got}")]
    InputSize {
        /// Expected sample count.
        expected: usize,
        /// Actual sample count.
        got: usize,
    },

    /// One segment in a batch has wrong number of samples.
    #[error("batch input size mismatch: segment {index} has {got} samples, expected {expected}")]
    BatchInputSize {
        /// Index of the problematic segment.
        index: usize,
        /// Expected sample count.
        expected: usize,
        /// Actual sample count.
        got: usize,
    },

    /// Failed to detect model type from ONNX structure.
    #[error("model detection failed: {reason}")]
    ModelDetection {
        /// Reason for detection failure.
        reason: String,
    },

    /// Number of labels doesn't match model output size.
    #[error("label count mismatch: model expects {expected}, got {got}")]
    LabelCount {
        /// Expected label count.
        expected: usize,
        /// Actual label count.
        got: usize,
    },

    /// Model path was not provided to builder.
    #[error("model path required")]
    ModelPathRequired,

    /// Labels were not provided to builder.
    #[error("labels required (provide path or vec)")]
    LabelsRequired,

    /// Failed to load ONNX model.
    #[error("failed to load model: {0}")]
    ModelLoad(#[from] ort::Error),

    /// Failed to load labels from file.
    #[error("failed to load labels from {path}: {reason}")]
    LabelLoad {
        /// Path that failed to load.
        path: String,
        /// Reason for failure.
        reason: String,
    },

    /// Failed to parse label file content.
    #[error("failed to parse labels: {0}")]
    LabelParse(String),

    /// Inference execution failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Invalid geographic coordinates provided.
    #[error("invalid coordinates: latitude: {latitude}, longitude: {longitude}, reason: {reason}")]
    InvalidCoordinates {
        /// Latitude value.
        latitude: f32,
        /// Longitude value.
        longitude: f32,
        /// Reason for invalidity.
        reason: String,
    },

    /// Range filter inference failed.
    #[error("range filter inference failed: {0}")]
    RangeFilterInference(String),

    /// Failed to initialize ONNX Runtime.
    #[error("failed to initialize ONNX Runtime: {0}")]
    RuntimeInit(String),

    /// Audio file format is not supported.
    #[error("unsupported audio format: {reason}")]
    AudioFormat {
        /// Reason for format rejection.
        reason: String,
    },

    /// Failed to read audio file.
    #[error("failed to read audio file {path}: {reason}")]
    AudioRead {
        /// Path to the audio file.
        path: String,
        /// Reason for failure.
        reason: String,
    },
}

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_input_size_error_display() {
        let err = Error::InputSize {
            expected: 144_000,
            got: 100_000,
        };
        assert_eq!(
            err.to_string(),
            "input size mismatch: expected 144000 samples, got 100000"
        );
    }

    #[test]
    fn test_batch_input_size_error_display() {
        let err = Error::BatchInputSize {
            index: 3,
            expected: 144_000,
            got: 50_000,
        };
        assert_eq!(
            err.to_string(),
            "batch input size mismatch: segment 3 has 50000 samples, expected 144000"
        );
    }

    #[test]
    fn test_model_detection_error_display() {
        let err = Error::ModelDetection {
            reason: "unsupported model".to_string(),
        };
        assert_eq!(err.to_string(), "model detection failed: unsupported model");
    }

    #[test]
    fn test_label_count_error_display() {
        let err = Error::LabelCount {
            expected: 6522,
            got: 1000,
        };
        assert_eq!(
            err.to_string(),
            "label count mismatch: model expects 6522, got 1000"
        );
    }

    #[test]
    fn test_audio_format_error_display() {
        let err = Error::AudioFormat {
            reason: "WAV must be mono".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "unsupported audio format: WAV must be mono"
        );
    }

    #[test]
    fn test_audio_read_error_display() {
        let err = Error::AudioRead {
            path: "/path/to/file.wav".to_string(),
            reason: "file not found".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "failed to read audio file /path/to/file.wav: file not found"
        );
    }

    #[test]
    fn test_invalid_coordinates_error() {
        let err = Error::InvalidCoordinates {
            latitude: 95.0,
            longitude: 200.0,
            reason: "latitude out of range".to_string(),
        };
        assert!(err.to_string().contains("latitude: 95"));
        assert!(err.to_string().contains("longitude: 200"));
    }

    #[test]
    fn test_range_filter_inference_error() {
        let err = Error::RangeFilterInference("model invoke failed".to_string());
        assert_eq!(
            err.to_string(),
            "range filter inference failed: model invoke failed"
        );
    }
}
