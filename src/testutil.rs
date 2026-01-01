//! Test utilities for creating mock data
//!
//! This module is only compiled in test builds.

#![allow(clippy::unwrap_used)] // Tests can use unwrap
#![allow(clippy::cast_precision_loss)] // Sample rate/index casts are fine for test utilities

use crate::types::{ModelConfig, ModelType, Prediction, PredictionResult};

/// Create a mock `ModelConfig` for testing
pub fn mock_config(model_type: ModelType) -> ModelConfig {
    ModelConfig {
        model_type,
        sample_rate: model_type.sample_rate(),
        segment_duration: model_type.segment_duration(),
        sample_count: model_type.sample_count(),
        num_species: match model_type {
            ModelType::BirdNetV24 => 6522,
            ModelType::BirdNetV30 => 1000,
            ModelType::PerchV2 => 500,
        },
        embedding_dim: match model_type {
            ModelType::BirdNetV24 => None,
            ModelType::BirdNetV30 => Some(1024),
            ModelType::PerchV2 => Some(512),
        },
    }
}

/// Create mock audio segment with correct size for model type
pub fn mock_audio_segment(model_type: ModelType) -> Vec<f32> {
    vec![0.0f32; model_type.sample_count()]
}

/// Create mock audio segment with sine wave
pub fn mock_sine_wave(model_type: ModelType, frequency: f32) -> Vec<f32> {
    let sample_rate = model_type.sample_rate() as f32;
    let sample_count = model_type.sample_count();

    (0..sample_count)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect()
}

/// Create mock labels
pub fn mock_labels(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("Species_{i}")).collect()
}

/// Create a mock `PredictionResult` for testing downstream code
pub fn mock_prediction_result(model_type: ModelType) -> PredictionResult {
    let embeddings = match model_type {
        ModelType::BirdNetV24 => None,
        ModelType::BirdNetV30 => Some(vec![0.1f32; 1024]),
        ModelType::PerchV2 => Some(vec![0.1f32; 512]),
    };

    PredictionResult {
        model_type,
        predictions: vec![
            Prediction {
                species: "American Robin".to_string(),
                confidence: 0.95,
                index: 0,
            },
            Prediction {
                species: "Northern Cardinal".to_string(),
                confidence: 0.85,
                index: 1,
            },
            Prediction {
                species: "Blue Jay".to_string(),
                confidence: 0.75,
                index: 2,
            },
        ],
        embeddings,
        raw_scores: vec![3.0, 2.0, 1.5], // Pre-sigmoid values
    }
}

/// Generate random logits for testing top-K selection
pub fn random_logits(count: usize, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducible "random" values
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            // Extract 16 bits and map to range [-5, 5]
            let bits = ((state >> 16) & 0xFFFF) as f32;
            bits.mul_add(10.0 / 65535.0, -5.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    #![allow(clippy::cast_precision_loss)]
    use super::*;

    #[test]
    fn test_mock_config() {
        let config = mock_config(ModelType::BirdNetV24);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.sample_count, 144_000);
        assert_eq!(config.embedding_dim, None);

        let config = mock_config(ModelType::BirdNetV30);
        assert_eq!(config.sample_rate, 32000);
        assert_eq!(config.embedding_dim, Some(1024));
    }

    #[test]
    fn test_mock_audio_segment_size() {
        let segment = mock_audio_segment(ModelType::BirdNetV24);
        assert_eq!(segment.len(), 144_000);

        let segment = mock_audio_segment(ModelType::BirdNetV30);
        assert_eq!(segment.len(), 160_000);
    }

    #[test]
    fn test_mock_sine_wave() {
        let wave = mock_sine_wave(ModelType::BirdNetV24, 440.0);
        assert_eq!(wave.len(), 144_000);

        // Check it's actually oscillating
        let max = wave.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = wave.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max > 0.9);
        assert!(min < -0.9);
    }

    #[test]
    fn test_mock_labels() {
        let labels = mock_labels(100);
        assert_eq!(labels.len(), 100);
        assert_eq!(labels[0], "Species_0");
        assert_eq!(labels[99], "Species_99");
    }

    #[test]
    fn test_mock_prediction_result() {
        let result = mock_prediction_result(ModelType::BirdNetV24);
        assert_eq!(result.model_type, ModelType::BirdNetV24);
        assert_eq!(result.predictions.len(), 3);
        assert!(result.embeddings.is_none());

        let result = mock_prediction_result(ModelType::BirdNetV30);
        assert!(result.embeddings.is_some());
        assert_eq!(result.embeddings.unwrap().len(), 1024);
    }

    #[test]
    fn test_random_logits() {
        let logits1 = random_logits(100, 12345);
        let logits2 = random_logits(100, 12345);

        // Same seed should produce same values
        assert_eq!(logits1, logits2);

        // Different seed should produce different values
        let logits3 = random_logits(100, 54321);
        assert_ne!(logits1, logits3);

        // Check range
        for &v in &logits1 {
            assert!((-5.0..=5.0).contains(&v));
        }
    }
}
