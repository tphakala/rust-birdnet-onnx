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
            Self::BirdNetV30 => LabelFormat::Csv,
            Self::PerchV2 => LabelFormat::Json,
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
        assert_eq!(model.expected_label_format(), LabelFormat::Json);
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
}
