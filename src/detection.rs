//! Model type detection from ONNX tensor shapes.

use crate::error::{Error, Result};
use crate::types::{ModelConfig, ModelType};

/// Detects model type from ONNX input/output tensor shapes.
///
/// # Arguments
/// * `input_shape` - Input tensor shape, expected `[batch, samples]` or `[batch, 1, samples]`
/// * `output_shapes` - Output tensor shapes
/// * `override_type` - Optional user override for ambiguous models (v3.0 vs Perch)
///
/// # Errors
/// Returns [`Error::ModelDetection`] if the model structure is not recognized.
pub fn detect_model_type(
    input_shape: &[i64],
    output_shapes: &[Vec<i64>],
    override_type: Option<ModelType>,
) -> Result<ModelConfig> {
    let sample_count = extract_sample_count(input_shape)?;
    let num_outputs = output_shapes.len();

    // If user provided override, validate and use it
    if let Some(model_type) = override_type {
        return build_config_with_override(model_type, sample_count, output_shapes);
    }

    // Auto-detection based on sample count and output count
    match (sample_count, num_outputs) {
        // BirdNET v2.4: 144,000 samples, 1 output (predictions)
        (144_000, 1) => {
            let num_species = extract_last_dim(&output_shapes[0])?;
            Ok(ModelConfig {
                model_type: ModelType::BirdNetV24,
                sample_rate: 48_000,
                segment_duration: 3.0,
                sample_count: 144_000,
                num_species,
                embedding_dim: None,
            })
        }

        // BirdNET v3.0 / Perch v2: 160,000 samples, 2 outputs (embeddings, predictions)
        (160_000, 2) => {
            let embedding_dim = extract_last_dim(&output_shapes[0])?;
            let num_species = extract_last_dim(&output_shapes[1])?;

            // Default to BirdNET v3.0, user can override for Perch
            Ok(ModelConfig {
                model_type: ModelType::BirdNetV30,
                sample_rate: 32_000,
                segment_duration: 5.0,
                sample_count: 160_000,
                num_species,
                embedding_dim: Some(embedding_dim),
            })
        }

        _ => Err(Error::ModelDetection {
            reason: format!(
                "unsupported model: {sample_count} samples, {num_outputs} outputs \
                 (expected 144000/1 or 160000/2)"
            ),
        }),
    }
}

/// Build config with user-specified model type, validating against actual shapes.
fn build_config_with_override(
    model_type: ModelType,
    sample_count: usize,
    output_shapes: &[Vec<i64>],
) -> Result<ModelConfig> {
    let expected_samples = model_type.sample_count();
    if sample_count != expected_samples {
        return Err(Error::ModelDetection {
            reason: format!(
                "model type {model_type:?} expects {expected_samples} samples, \
                 but model has {sample_count}"
            ),
        });
    }

    let (embedding_dim, num_species) = match model_type {
        ModelType::BirdNetV24 => {
            if output_shapes.len() != 1 {
                return Err(Error::ModelDetection {
                    reason: format!(
                        "BirdNET v2.4 expects 1 output, got {}",
                        output_shapes.len()
                    ),
                });
            }
            (None, extract_last_dim(&output_shapes[0])?)
        }
        ModelType::BirdNetV30 | ModelType::PerchV2 => {
            if output_shapes.len() != 2 {
                return Err(Error::ModelDetection {
                    reason: format!(
                        "{model_type:?} expects 2 outputs, got {}",
                        output_shapes.len()
                    ),
                });
            }
            (
                Some(extract_last_dim(&output_shapes[0])?),
                extract_last_dim(&output_shapes[1])?,
            )
        }
    };

    Ok(ModelConfig {
        model_type,
        sample_rate: model_type.sample_rate(),
        segment_duration: model_type.segment_duration(),
        sample_count,
        num_species,
        embedding_dim,
    })
}

/// Extract sample count from input shape.
/// Handles `[batch, samples]` or `[batch, 1, samples]`.
fn extract_sample_count(shape: &[i64]) -> Result<usize> {
    let value = match shape.len() {
        2 => shape[1],
        3 => shape[2],
        _ => {
            return Err(Error::ModelDetection {
                reason: format!("unexpected input shape: {shape:?}"),
            })
        }
    };

    usize::try_from(value).map_err(|_| Error::ModelDetection {
        reason: format!("invalid sample count: {value}"),
    })
}

/// Extract last dimension from output shape.
fn extract_last_dim(shape: &[i64]) -> Result<usize> {
    let value = shape.last().copied().ok_or_else(|| Error::ModelDetection {
        reason: "empty output shape".to_string(),
    })?;

    usize::try_from(value).map_err(|_| Error::ModelDetection {
        reason: format!("invalid dimension: {value}"),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::disallowed_methods)]
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn test_detect_birdnet_v24() {
        let input_shape = vec![1, 144_000];
        let output_shapes = vec![vec![1, 6522]];

        let config = detect_model_type(&input_shape, &output_shapes, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.sample_rate, 48_000);
        assert_eq!(config.segment_duration, 3.0);
        assert_eq!(config.sample_count, 144_000);
        assert_eq!(config.num_species, 6522);
        assert_eq!(config.embedding_dim, None);
    }

    #[test]
    fn test_detect_birdnet_v30() {
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 1024], vec![1, 1000]];

        let config = detect_model_type(&input_shape, &output_shapes, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV30);
        assert_eq!(config.sample_rate, 32_000);
        assert_eq!(config.segment_duration, 5.0);
        assert_eq!(config.sample_count, 160_000);
        assert_eq!(config.num_species, 1000);
        assert_eq!(config.embedding_dim, Some(1024));
    }

    #[test]
    fn test_detect_with_perch_override() {
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 512], vec![1, 500]];

        let config =
            detect_model_type(&input_shape, &output_shapes, Some(ModelType::PerchV2)).unwrap();

        assert_eq!(config.model_type, ModelType::PerchV2);
        assert_eq!(config.embedding_dim, Some(512));
        assert_eq!(config.num_species, 500);
    }

    #[test]
    fn test_detect_with_invalid_override() {
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 1024], vec![1, 1000]];

        // BirdNET v2.4 expects 144,000 samples, not 160,000
        let result = detect_model_type(&input_shape, &output_shapes, Some(ModelType::BirdNetV24));

        assert!(result.is_err());
    }

    #[test]
    fn test_detect_unsupported_model() {
        let input_shape = vec![1, 100_000]; // Wrong sample count
        let output_shapes = vec![vec![1, 1000]];

        let result = detect_model_type(&input_shape, &output_shapes, None);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unsupported model"));
    }

    #[test]
    fn test_extract_sample_count_2d() {
        assert_eq!(extract_sample_count(&[1, 144_000]).unwrap(), 144_000);
    }

    #[test]
    fn test_extract_sample_count_3d() {
        assert_eq!(extract_sample_count(&[1, 1, 144_000]).unwrap(), 144_000);
    }
}
