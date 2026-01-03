/// Supported model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// `BirdNET` v2.4 - 48kHz, 3s segments, no embeddings.
    BirdNetV24,
    /// `BirdNET` v3.0 - 32kHz, 5s segments, 1024-dim embeddings.
    BirdNetV30,
    /// Google Perch v2 - 32kHz, 5s segments, variable embeddings.
    PerchV2,
}

impl ModelType {
    /// Sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        match self {
            Self::BirdNetV24 => 48_000,
            Self::BirdNetV30 | Self::PerchV2 => 32_000,
        }
    }

    /// Segment duration in seconds.
    #[must_use]
    pub const fn segment_duration(&self) -> f32 {
        match self {
            Self::BirdNetV24 => 3.0,
            Self::BirdNetV30 | Self::PerchV2 => 5.0,
        }
    }

    /// Expected sample count per segment.
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        match self {
            Self::BirdNetV24 => 144_000,
            Self::BirdNetV30 | Self::PerchV2 => 160_000,
        }
    }

    /// Whether this model produces embeddings.
    #[must_use]
    pub const fn has_embeddings(&self) -> bool {
        match self {
            Self::BirdNetV24 => false,
            Self::BirdNetV30 | Self::PerchV2 => true,
        }
    }

    /// Expected label file format for this model type.
    #[must_use]
    pub const fn expected_label_format(&self) -> LabelFormat {
        match self {
            Self::BirdNetV24 => LabelFormat::Text,
            Self::BirdNetV30 | Self::PerchV2 => LabelFormat::Csv,
        }
    }
}

/// Expected label format per model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelFormat {
    /// One label per line.
    Text,
    /// CSV with first column as label.
    Csv,
    /// JSON array or object.
    Json,
}

/// Information about execution providers (hardware backends).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionProviderInfo {
    /// CPU execution (always available, fallback).
    Cpu,
    /// NVIDIA CUDA GPU acceleration.
    Cuda,
    /// NVIDIA `TensorRT` optimized inference.
    TensorRt,
    /// Windows `DirectML` (DirectX 12).
    DirectMl,
    /// Apple `CoreML` (macOS/iOS).
    CoreMl,
    /// AMD `ROCm` GPU acceleration.
    Rocm,
    /// Intel `OpenVINO`.
    OpenVino,
    /// Intel oneDNN.
    OneDnn,
    /// Qualcomm QNN (NPU).
    Qnn,
    /// ARM Compute Library.
    Acl,
    /// ARM NN.
    ArmNn,
}

impl ExecutionProviderInfo {
    /// Get the execution provider identifier string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::TensorRt => "TensorRT",
            Self::DirectMl => "DirectML",
            Self::CoreMl => "CoreML",
            Self::Rocm => "ROCm",
            Self::OpenVino => "OpenVINO",
            Self::OneDnn => "oneDNN",
            Self::Qnn => "QNN",
            Self::Acl => "ACL",
            Self::ArmNn => "ArmNN",
        }
    }

    /// Get the category of this execution provider.
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda | Self::TensorRt | Self::Rocm | Self::DirectMl => "GPU",
            Self::CoreMl => "Neural Engine",
            Self::Qnn => "NPU",
            Self::OpenVino | Self::OneDnn | Self::Acl | Self::ArmNn => "Accelerator",
        }
    }
}

impl std::fmt::Display for ExecutionProviderInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Model configuration derived from detected model type.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Detected or overridden model type.
    pub model_type: ModelType,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Segment duration in seconds.
    pub segment_duration: f32,
    /// Expected sample count per segment.
    pub sample_count: usize,
    /// Number of species classes in model output.
    pub num_species: usize,
    /// Embedding dimension (None for v2.4).
    pub embedding_dim: Option<usize>,
}

/// Single species prediction.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Species name from labels.
    pub species: String,
    /// Confidence score (0.0 - 1.0, after sigmoid).
    pub confidence: f32,
    /// Index in model output.
    pub index: usize,
}

/// Complete inference result.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Model type that produced this result.
    pub model_type: ModelType,
    /// Top predictions sorted by confidence (descending).
    pub predictions: Vec<Prediction>,
    /// Feature embeddings (None for `BirdNET` v2.4).
    pub embeddings: Option<Vec<f32>>,
    /// Raw logits from model output.
    pub raw_scores: Vec<f32>,
}

/// Species probability score from meta model based on location and date.
#[derive(Debug, Clone)]
pub struct LocationScore {
    /// Species name from labels.
    pub species: String,
    /// Probability score (0.0 - 1.0) for this species at given location/time.
    pub score: f32,
    /// Index in model output.
    pub index: usize,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::float_cmp)]
    #![allow(clippy::cast_precision_loss)]
    use super::*;

    #[test]
    fn test_birdnet_v24_properties() {
        let model = ModelType::BirdNetV24;
        assert_eq!(model.sample_rate(), 48_000);
        assert_eq!(model.segment_duration(), 3.0);
        assert_eq!(model.sample_count(), 144_000);
        assert!(!model.has_embeddings());
        assert_eq!(model.expected_label_format(), LabelFormat::Text);
    }

    #[test]
    fn test_birdnet_v30_properties() {
        let model = ModelType::BirdNetV30;
        assert_eq!(model.sample_rate(), 32_000);
        assert_eq!(model.segment_duration(), 5.0);
        assert_eq!(model.sample_count(), 160_000);
        assert!(model.has_embeddings());
        assert_eq!(model.expected_label_format(), LabelFormat::Csv);
    }

    #[test]
    fn test_perch_v2_properties() {
        let model = ModelType::PerchV2;
        assert_eq!(model.sample_rate(), 32_000);
        assert_eq!(model.segment_duration(), 5.0);
        assert_eq!(model.sample_count(), 160_000);
        assert!(model.has_embeddings());
        assert_eq!(model.expected_label_format(), LabelFormat::Csv);
    }

    #[test]
    fn test_sample_count_matches_rate_times_duration() {
        for model in [
            ModelType::BirdNetV24,
            ModelType::BirdNetV30,
            ModelType::PerchV2,
        ] {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let expected = (model.sample_rate() as f32 * model.segment_duration()) as usize;
            assert_eq!(model.sample_count(), expected);
        }
    }

    #[test]
    fn test_location_score_creation() {
        let score = LocationScore {
            species: "Turdus merula_Common Blackbird".to_string(),
            score: 0.85,
            index: 42,
        };
        assert_eq!(score.species, "Turdus merula_Common Blackbird");
        assert_eq!(score.score, 0.85);
        assert_eq!(score.index, 42);
    }

    #[test]
    fn test_execution_provider_info_display() {
        assert_eq!(ExecutionProviderInfo::Cpu.as_str(), "CPU");
        assert_eq!(ExecutionProviderInfo::Cuda.as_str(), "CUDA");
        assert_eq!(ExecutionProviderInfo::TensorRt.as_str(), "TensorRT");
        assert_eq!(ExecutionProviderInfo::DirectMl.as_str(), "DirectML");
        assert_eq!(ExecutionProviderInfo::CoreMl.as_str(), "CoreML");
    }

    #[test]
    fn test_execution_provider_info_category() {
        assert_eq!(ExecutionProviderInfo::Cpu.category(), "CPU");
        assert_eq!(ExecutionProviderInfo::Cuda.category(), "GPU");
        assert_eq!(ExecutionProviderInfo::TensorRt.category(), "GPU");
        assert_eq!(ExecutionProviderInfo::DirectMl.category(), "GPU");
        assert_eq!(ExecutionProviderInfo::CoreMl.category(), "Neural Engine");
    }
}
