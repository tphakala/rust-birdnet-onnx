//! Model type detection from ONNX tensor shapes.

use crate::error::{Error, Result};
use crate::types::{ModelConfig, ModelType};

/// Output names that identify an embeddings tensor, lowercased.
///
/// Deliberately keyed on the embeddings output rather than the class-score one.
/// Across the supported exports the embeddings name is stable while the class
/// scores answer to at least three different names: `BirdNET` v3.0 fp32 calls
/// them `predictions`, its fp16 export of the SAME weights calls them `output`,
/// and `Perch` v2 calls them `label`.
const EMBEDDING_OUTPUT_NAMES: &[&str] = &["embeddings", "embedding"];

/// Which output carries what, for a two-output model.
///
/// Resolved from names where they say so, because output ORDER is not a
/// property of a model family. Every published `BirdNET` v3.0 export puts the
/// embeddings second, which is the opposite of what this crate assumed until
/// the models were tested against it, and reading the embedding tensor as class
/// scores is silent nonsense rather than an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutputRoles {
    /// Index of the class-score tensor.
    predictions: usize,
    /// Index of the embeddings tensor.
    embeddings: usize,
}

/// Locate the embeddings and class-score outputs of a two-output model.
///
/// Falls back to "class scores first, embeddings second" when the names say
/// nothing, which is the layout of every published two-output model this crate
/// has been tested against, `BirdNET` v3.0 and v2.4-with-embeddings alike.
fn resolve_two_output_roles(output_names: &[String]) -> OutputRoles {
    const FALLBACK: OutputRoles = OutputRoles {
        predictions: 0,
        embeddings: 1,
    };

    let embedding_positions: Vec<usize> = output_names
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            let name = name.to_ascii_lowercase();
            EMBEDDING_OUTPUT_NAMES.contains(&name.as_str())
        })
        .map(|(index, _)| index)
        .collect();

    // Exactly one match, or the name told us nothing useful. Two outputs both
    // claiming to be embeddings is a model we do not understand, and guessing
    // between them would be worse than the documented default.
    match embedding_positions.as_slice() {
        [embeddings] if output_names.len() == 2 => OutputRoles {
            predictions: 1 - embeddings,
            embeddings: *embeddings,
        },
        _ => FALLBACK,
    }
}

/// Detects model type from ONNX input/output tensor shapes.
///
/// # Arguments
/// * `input_shape` - Input tensor shape, expected `[batch, samples]` or `[batch, 1, samples]`
/// * `output_shapes` - Output tensor shapes
/// * `output_names` - Output tensor names, in the same order as `output_shapes`.
///   Pass an empty slice when they are unavailable, and detection falls back to
///   positional layout.
/// * `override_type` - Optional user override for ambiguous models (v3.0 vs `Perch`)
///
/// # Errors
/// Returns [`Error::ModelDetection`] if the model structure is not recognized.
pub fn detect_model_type(
    input_shape: &[i64],
    output_shapes: &[Vec<i64>],
    output_names: &[String],
    override_type: Option<ModelType>,
) -> Result<ModelConfig> {
    let sample_count_opt = extract_sample_count(input_shape);
    let num_outputs = output_shapes.len();

    // If user provided override, validate and use it
    if let Some(model_type) = override_type {
        let sample_count = sample_count_opt.unwrap_or_else(|| model_type.sample_count());
        return build_config_with_override(model_type, sample_count, output_shapes, output_names);
    }

    // If sample count is known, use it for detection
    if let Some(sample_count) = sample_count_opt {
        return detect_from_sample_count(sample_count, num_outputs, output_shapes, output_names);
    }

    // Input has dynamic dimensions - infer from output shapes
    detect_from_outputs(num_outputs, output_shapes, output_names)
}

/// Detect model type from known sample count and output patterns.
fn detect_from_sample_count(
    sample_count: usize,
    num_outputs: usize,
    output_shapes: &[Vec<i64>],
    output_names: &[String],
) -> Result<ModelConfig> {
    // Auto-detection based on sample count and output count
    match (sample_count, num_outputs) {
        // `BirdNET` v2.4: 144,000 samples, 1 output (predictions)
        (144_000, 1) => {
            let num_species = extract_last_dim(&output_shapes[0])?;
            Ok(ModelConfig {
                model_type: ModelType::BirdNetV24,
                sample_rate: 48_000,
                segment_duration: 3.0,
                sample_count: 144_000,
                num_species,
                embedding_dim: None,
                predictions_index: 0,
                embeddings_index: None,
            })
        }

        // BirdNET v2.4 with embeddings: 144,000 samples, 2 outputs.
        (144_000, 2) => {
            let roles = resolve_two_output_roles(output_names);
            let num_species = extract_last_dim(&output_shapes[roles.predictions])?;
            let embedding_dim = extract_last_dim(&output_shapes[roles.embeddings])?;
            Ok(ModelConfig {
                model_type: ModelType::BirdNetV24,
                sample_rate: 48_000,
                segment_duration: 3.0,
                sample_count: 144_000,
                num_species,
                embedding_dim: Some(embedding_dim),
                predictions_index: roles.predictions,
                embeddings_index: Some(roles.embeddings),
            })
        }

        // `BirdNET` v3.0: 160,000 samples, 2 outputs.
        (160_000, 2) => {
            let roles = resolve_two_output_roles(output_names);
            let embedding_dim = extract_last_dim(&output_shapes[roles.embeddings])?;
            let num_species = extract_last_dim(&output_shapes[roles.predictions])?;

            Ok(ModelConfig {
                model_type: ModelType::BirdNetV30,
                sample_rate: 32_000,
                segment_duration: 5.0,
                sample_count: 160_000,
                num_species,
                embedding_dim: Some(embedding_dim),
                predictions_index: roles.predictions,
                embeddings_index: Some(roles.embeddings),
            })
        }

        // `Perch` v2: 160,000 samples, 4 outputs (embedding, spatial_embedding, spectrogram, predictions)
        (160_000, 4) => {
            let embedding_dim = extract_last_dim(&output_shapes[0])?;
            let num_species = extract_last_dim(&output_shapes[3])?; // predictions at index 3

            Ok(ModelConfig {
                model_type: ModelType::PerchV2,
                sample_rate: 32_000,
                segment_duration: 5.0,
                sample_count: 160_000,
                num_species,
                embedding_dim: Some(embedding_dim),
                predictions_index: 3,
                embeddings_index: Some(0),
            })
        }

        _ => Err(Error::ModelDetection {
            reason: format!(
                "unsupported model: {sample_count} samples, {num_outputs} outputs \
                 (expected 144000/1, 144000/2, 160000/2, or 160000/4)"
            ),
        }),
    }
}

/// Embedding width of `BirdNET` v3.0, the one two-output family this crate can
/// name from its embedding tensor alone. v2.4's 1024 is not listed because it
/// is the default rather than a signal.
const V30_EMBEDDING_DIM: usize = 1280;

/// Detect model type from output shapes when input dimensions are dynamic.
fn detect_from_outputs(
    num_outputs: usize,
    output_shapes: &[Vec<i64>],
    output_names: &[String],
) -> Result<ModelConfig> {
    match num_outputs {
        // `BirdNET` v2.4: 1 output
        1 => {
            let num_species = extract_last_dim(&output_shapes[0])?;
            Ok(ModelConfig {
                model_type: ModelType::BirdNetV24,
                sample_rate: 48_000,
                segment_duration: 3.0,
                sample_count: 144_000,
                num_species,
                embedding_dim: None,
                predictions_index: 0,
                embeddings_index: None,
            })
        }

        // 2 outputs: either v2.4-with-embeddings or v3.0.
        //
        // Which output is which comes from the names; which FAMILY it is comes
        // from the embedding width, 1024 for v2.4 and 1280 for v3.0. The two
        // questions are separate, and conflating them is what made this branch
        // wrong: it decided the family by "is the first output bigger", which
        // is true for a v3.0 export that puts its 11,560 class scores ahead of
        // its 1,280-wide embeddings, so every such model was detected as v2.4
        // and run at 48 kHz with a 3 second window.
        2 => {
            let roles = resolve_two_output_roles(output_names);
            let num_species = extract_last_dim(&output_shapes[roles.predictions])?;
            let embedding_dim = extract_last_dim(&output_shapes[roles.embeddings])?;

            // Only v3.0's width is a positive signal. v2.4 is the default for
            // its own width and for anything unrecognised alike, because an
            // embedding tensor narrower than the class scores is the shape both
            // families share and so distinguishes nothing.
            let model_type = if embedding_dim == V30_EMBEDDING_DIM {
                ModelType::BirdNetV30
            } else {
                ModelType::BirdNetV24
            };

            Ok(ModelConfig {
                model_type,
                sample_rate: model_type.sample_rate(),
                segment_duration: model_type.segment_duration(),
                sample_count: model_type.sample_count(),
                num_species,
                embedding_dim: Some(embedding_dim),
                predictions_index: roles.predictions,
                embeddings_index: Some(roles.embeddings),
            })
        }

        // `Perch` v2: 4 outputs
        4 => {
            let embedding_dim = extract_last_dim(&output_shapes[0])?;
            let num_species = extract_last_dim(&output_shapes[3])?;

            Ok(ModelConfig {
                model_type: ModelType::PerchV2,
                sample_rate: 32_000,
                segment_duration: 5.0,
                sample_count: 160_000,
                num_species,
                embedding_dim: Some(embedding_dim),
                predictions_index: 3,
                embeddings_index: Some(0),
            })
        }

        _ => Err(Error::ModelDetection {
            reason: format!(
                "unsupported model with dynamic input: {num_outputs} outputs \
                 (expected 1, 2, or 4)"
            ),
        }),
    }
}

/// Build config with user-specified model type, validating against actual shapes.
fn build_config_with_override(
    model_type: ModelType,
    sample_count: usize,
    output_shapes: &[Vec<i64>],
    output_names: &[String],
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

    // An override says which FAMILY this is. It says nothing about which output
    // is which, so the roles are resolved from the names exactly as they are on
    // the auto-detection path. Assuming a layout here was the reason `--model
    // v30` failed on the published models in the same way auto-detection did.
    let (embedding_dim, num_species, roles) = match model_type {
        ModelType::BirdNetV24 => match output_shapes.len() {
            1 => (
                None,
                extract_last_dim(&output_shapes[0])?,
                OutputRoles {
                    predictions: 0,
                    embeddings: 0,
                },
            ),
            2 => {
                let roles = resolve_two_output_roles(output_names);
                (
                    Some(extract_last_dim(&output_shapes[roles.embeddings])?),
                    extract_last_dim(&output_shapes[roles.predictions])?,
                    roles,
                )
            }
            n => {
                return Err(Error::ModelDetection {
                    reason: format!("BirdNET v2.4 expects 1 or 2 outputs, got {n}"),
                });
            }
        },
        ModelType::BirdNetV30 => {
            if output_shapes.len() != 2 {
                return Err(Error::ModelDetection {
                    reason: format!(
                        "`BirdNET` v3.0 expects 2 outputs, got {}",
                        output_shapes.len()
                    ),
                });
            }
            let roles = resolve_two_output_roles(output_names);
            (
                Some(extract_last_dim(&output_shapes[roles.embeddings])?),
                extract_last_dim(&output_shapes[roles.predictions])?,
                roles,
            )
        }
        ModelType::PerchV2 => {
            if output_shapes.len() != 4 {
                return Err(Error::ModelDetection {
                    reason: format!("`Perch` v2 expects 4 outputs, got {}", output_shapes.len()),
                });
            }
            (
                Some(extract_last_dim(&output_shapes[0])?),
                extract_last_dim(&output_shapes[3])?, // predictions at index 3
                OutputRoles {
                    predictions: 3,
                    embeddings: 0,
                },
            )
        }
        ModelType::BsgFinland => {
            if output_shapes.len() != 1 {
                return Err(Error::ModelDetection {
                    reason: format!("BSG Finland expects 1 output, got {}", output_shapes.len()),
                });
            }
            (
                None,
                extract_last_dim(&output_shapes[0])?,
                OutputRoles {
                    predictions: 0,
                    embeddings: 0,
                },
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
        predictions_index: roles.predictions,
        embeddings_index: embedding_dim.map(|_| roles.embeddings),
    })
}

/// Extract sample count from input shape.
/// Handles `[batch, samples]` or `[batch, 1, samples]`.
/// Returns `None` if the dimension is dynamic (-1).
fn extract_sample_count(shape: &[i64]) -> Option<usize> {
    let value = match shape.len() {
        2 => shape[1],
        3 => shape[2],
        _ => return None, // Unexpected shape
    };

    // Handle dynamic dimensions (-1) or invalid values
    if value <= 0 {
        return None;
    }

    usize::try_from(value).ok()
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

    /// Output names as the session reports them.
    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    #[test]
    fn test_roles_find_the_embeddings_output_in_either_position() {
        assert_eq!(
            resolve_two_output_roles(&names(&["predictions", "embeddings"])),
            OutputRoles {
                predictions: 0,
                embeddings: 1
            }
        );
        assert_eq!(
            resolve_two_output_roles(&names(&["embeddings", "predictions"])),
            OutputRoles {
                predictions: 1,
                embeddings: 0
            }
        );
    }

    #[test]
    fn test_roles_accept_the_singular_embedding_spelling() {
        // Perch spells it `embedding`. Supporting both costs nothing and
        // avoids a fix that works for one publisher's habit only.
        assert_eq!(
            resolve_two_output_roles(&names(&["embedding", "label"])),
            OutputRoles {
                predictions: 1,
                embeddings: 0
            }
        );
    }

    #[test]
    fn test_roles_ignore_case_in_output_names() {
        assert_eq!(
            resolve_two_output_roles(&names(&["Output", "Embeddings"])),
            OutputRoles {
                predictions: 0,
                embeddings: 1
            }
        );
    }

    #[test]
    fn test_roles_fall_back_when_no_name_identifies_the_embeddings() {
        // BirdNET v2.4's embedding export, whose second output is named
        // `model/GLOBAL_AVG_POOL/Mean_reduced_0`. The fallback is the layout it
        // actually has, so this keeps working.
        let roles =
            resolve_two_output_roles(&names(&["output", "model/GLOBAL_AVG_POOL/Mean_reduced_0"]));

        assert_eq!(
            roles,
            OutputRoles {
                predictions: 0,
                embeddings: 1
            }
        );
    }

    #[test]
    fn test_roles_fall_back_when_both_outputs_claim_to_be_embeddings() {
        // Guessing between two equally plausible candidates would be worse
        // than the documented default.
        let roles = resolve_two_output_roles(&names(&["embeddings", "embedding"]));

        assert_eq!(
            roles,
            OutputRoles {
                predictions: 0,
                embeddings: 1
            }
        );
    }

    #[test]
    fn test_roles_fall_back_on_an_empty_name_list() {
        assert_eq!(
            resolve_two_output_roles(&[]),
            OutputRoles {
                predictions: 0,
                embeddings: 1
            }
        );
    }

    #[test]
    fn test_every_detected_config_indexes_an_output_that_exists() {
        // Inference now reads whatever these point at, for every family, so an
        // index outside the model's outputs would be an out-of-bounds read
        // rather than the loud shape mismatch it used to be.
        /// One model's shape: input shape, output shapes, output names.
        type Case = (Vec<i64>, Vec<Vec<i64>>, Vec<String>);

        let cases: Vec<Case> = vec![
            (vec![1, 144_000], vec![vec![1, 6522]], names(&["output"])),
            (
                vec![1, 144_000],
                vec![vec![1, 6522], vec![1, 1024]],
                names(&["output", "model/GLOBAL_AVG_POOL/Mean_reduced_0"]),
            ),
            (
                vec![1, 160_000],
                vec![vec![1, 11_560], vec![1, 1280]],
                names(&["predictions", "embeddings"]),
            ),
            (
                vec![1, 160_000],
                vec![vec![1, 422], vec![1, 1280]],
                names(&["output", "embeddings"]),
            ),
            (
                vec![1, 160_000],
                vec![
                    vec![1, 1536],
                    vec![1, 16, 4, 1536],
                    vec![1, 500, 128],
                    vec![1, 638],
                ],
                names(&["embedding", "spatial_embedding", "spectrogram", "label"]),
            ),
        ];

        for (input_shape, output_shapes, output_names) in cases {
            let config =
                detect_model_type(&input_shape, &output_shapes, &output_names, None).unwrap();

            assert!(
                config.predictions_index < output_shapes.len(),
                "{:?}: class-score index {} is out of range",
                output_names,
                config.predictions_index
            );
            assert_eq!(
                extract_last_dim(&output_shapes[config.predictions_index]).unwrap(),
                config.num_species,
                "{output_names:?}: class-score index does not point at the class scores"
            );

            if let Some(index) = config.embeddings_index {
                assert!(index < output_shapes.len());
                assert_eq!(
                    extract_last_dim(&output_shapes[index]).unwrap(),
                    config.embedding_dim.unwrap(),
                    "{output_names:?}: embeddings index does not point at the embeddings"
                );
            }
        }
    }

    #[test]
    fn test_an_override_still_resolves_roles_from_names() {
        // Passing --model v30 used to bypass name resolution and assume the
        // layout, so an explicit override failed on exactly the models an
        // override exists to rescue.
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 422], vec![1, 1280]];
        let names = names(&["output", "embeddings"]);

        let config = detect_model_type(
            &input_shape,
            &output_shapes,
            &names,
            Some(ModelType::BirdNetV30),
        )
        .unwrap();

        assert_eq!(config.num_species, 422);
        assert_eq!(config.embedding_dim, Some(1280));
        assert_eq!(config.predictions_index, 0);
        assert_eq!(config.embeddings_index, Some(1));
    }

    #[test]
    fn test_detect_birdnet_v24() {
        let input_shape = vec![1, 144_000];
        let output_shapes = vec![vec![1, 6522]];

        let config = detect_model_type(&input_shape, &output_shapes, &[], None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.sample_rate, 48_000);
        assert_eq!(config.segment_duration, 3.0);
        assert_eq!(config.sample_count, 144_000);
        assert_eq!(config.num_species, 6522);
        assert_eq!(config.embedding_dim, None);
    }

    #[test]
    fn test_detect_birdnet_v30() {
        // The published fp32 layout: class scores first, named `predictions`.
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 11_560], vec![1, 1280]];
        let names = names(&["predictions", "embeddings"]);

        let config = detect_model_type(&input_shape, &output_shapes, &names, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV30);
        assert_eq!(config.sample_rate, 32_000);
        assert_eq!(config.segment_duration, 5.0);
        assert_eq!(config.sample_count, 160_000);
        assert_eq!(config.num_species, 11_560);
        assert_eq!(config.embedding_dim, Some(1280));
        assert_eq!(config.predictions_index, 0);
        assert_eq!(config.embeddings_index, Some(1));
    }

    #[test]
    fn test_detect_birdnet_v30_fp16_names_its_scores_output() {
        // Same weights, different export: the fp16 file calls its class scores
        // `output`, not `predictions`. Keying on the class-score name would fix
        // one variant of a model and leave the other broken, which is why the
        // embeddings name is what identifies the pair.
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 422], vec![1, 1280]];
        let names = names(&["output", "embeddings"]);

        let config = detect_model_type(&input_shape, &output_shapes, &names, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV30);
        assert_eq!(config.num_species, 422);
        assert_eq!(config.embedding_dim, Some(1280));
        assert_eq!(config.predictions_index, 0);
        assert_eq!(config.embeddings_index, Some(1));
    }

    #[test]
    fn test_detect_birdnet_v30_with_embeddings_first() {
        // Order must not matter once the names are known. This is the layout
        // the crate assumed for every v3.0 model, and getting it from the names
        // rather than the position is the whole point.
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 1280], vec![1, 11_560]];
        let names = names(&["embeddings", "predictions"]);

        let config = detect_model_type(&input_shape, &output_shapes, &names, None).unwrap();

        assert_eq!(config.num_species, 11_560);
        assert_eq!(config.embedding_dim, Some(1280));
        assert_eq!(config.predictions_index, 1);
        assert_eq!(config.embeddings_index, Some(0));
    }

    #[test]
    fn test_detect_two_output_model_without_names_assumes_scores_first() {
        // The documented fallback. Every published two-output model puts its
        // class scores first, so that is what an unnamed pair is read as.
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 11_560], vec![1, 1280]];

        let config = detect_model_type(&input_shape, &output_shapes, &[], None).unwrap();

        assert_eq!(config.num_species, 11_560);
        assert_eq!(config.embedding_dim, Some(1280));
        assert_eq!(config.predictions_index, 0);
    }

    #[test]
    fn test_detect_perch_v2() {
        let input_shape = vec![1, 160_000];
        // `Perch` v2 has 4 outputs: embedding, spatial_embedding, spectrogram, predictions
        let output_shapes = vec![
            vec![1, 1536],        // embedding
            vec![1, 16, 4, 1536], // spatial_embedding
            vec![1, 500, 128],    // spectrogram
            vec![1, 14795],       // predictions
        ];

        let config = detect_model_type(&input_shape, &output_shapes, &[], None).unwrap();

        assert_eq!(config.model_type, ModelType::PerchV2);
        assert_eq!(config.sample_rate, 32_000);
        assert_eq!(config.segment_duration, 5.0);
        assert_eq!(config.sample_count, 160_000);
        assert_eq!(config.num_species, 14795);
        assert_eq!(config.embedding_dim, Some(1536));
    }

    #[test]
    fn test_detect_with_perch_override() {
        let input_shape = vec![1, 160_000];
        // `Perch` v2 has 4 outputs: embedding, spatial_embedding, spectrogram, predictions
        let output_shapes = vec![
            vec![1, 512],        // embedding
            vec![1, 16, 4, 512], // spatial_embedding
            vec![1, 500, 128],   // spectrogram
            vec![1, 500],        // predictions
        ];

        let config =
            detect_model_type(&input_shape, &output_shapes, &[], Some(ModelType::PerchV2)).unwrap();

        assert_eq!(config.model_type, ModelType::PerchV2);
        assert_eq!(config.embedding_dim, Some(512));
        assert_eq!(config.num_species, 500);
    }

    #[test]
    fn test_detect_with_invalid_override() {
        let input_shape = vec![1, 160_000];
        let output_shapes = vec![vec![1, 1024], vec![1, 1000]];

        // `BirdNET` v2.4 expects 144,000 samples, not 160,000
        let result = detect_model_type(
            &input_shape,
            &output_shapes,
            &[],
            Some(ModelType::BirdNetV24),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_detect_unsupported_model() {
        let input_shape = vec![1, 100_000]; // Wrong sample count
        let output_shapes = vec![vec![1, 1000]];

        let result = detect_model_type(&input_shape, &output_shapes, &[], None);

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

    #[test]
    fn test_extract_sample_count_dynamic() {
        // Dynamic dimensions should return None
        assert_eq!(extract_sample_count(&[-1, -1]), None);
        assert_eq!(extract_sample_count(&[1, -1]), None);
        assert_eq!(extract_sample_count(&[-1, 160_000]), Some(160_000));
    }

    #[test]
    fn test_detect_birdnet_v30_dynamic_input() {
        // Input shape with dynamic dimensions
        let input_shape = vec![-1, -1];
        let output_shapes = vec![vec![-1, 1280], vec![-1, 11560]];
        let names = names(&["embeddings", "predictions"]);

        let config = detect_model_type(&input_shape, &output_shapes, &names, None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV30);
        assert_eq!(config.sample_rate, 32_000);
        assert_eq!(config.segment_duration, 5.0);
        assert_eq!(config.sample_count, 160_000);
        assert_eq!(config.num_species, 11560);
        assert_eq!(config.embedding_dim, Some(1280));
    }

    #[test]
    fn test_detect_perch_v2_dynamic_input() {
        // Input shape with dynamic dimensions
        let input_shape = vec![-1, -1];
        let output_shapes = vec![
            vec![-1, 1536],        // embedding
            vec![-1, 16, 4, 1536], // spatial_embedding
            vec![-1, 500, 128],    // spectrogram
            vec![-1, 14795],       // predictions
        ];

        let config = detect_model_type(&input_shape, &output_shapes, &[], None).unwrap();

        assert_eq!(config.model_type, ModelType::PerchV2);
        assert_eq!(config.sample_count, 160_000);
        assert_eq!(config.num_species, 14795);
        assert_eq!(config.embedding_dim, Some(1536));
    }

    #[test]
    fn test_detect_birdnet_v24_with_embeddings() {
        let input_shape = vec![1, 144_000];
        // v2.4 with embeddings: predictions at 0, embeddings at 1
        let output_shapes = vec![vec![1, 6522], vec![1, 1024]];

        let config = detect_model_type(&input_shape, &output_shapes, &[], None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.sample_rate, 48_000);
        assert_eq!(config.segment_duration, 3.0);
        assert_eq!(config.sample_count, 144_000);
        assert_eq!(config.num_species, 6522);
        assert_eq!(config.embedding_dim, Some(1024));
    }

    #[test]
    fn test_detect_birdnet_v24_with_embeddings_dynamic_input() {
        let input_shape = vec![-1, -1];
        let output_shapes = vec![vec![-1, 6522], vec![-1, 1024]];

        let config = detect_model_type(&input_shape, &output_shapes, &[], None).unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.embedding_dim, Some(1024));
    }

    #[test]
    fn test_detect_birdnet_v24_with_embeddings_override() {
        let input_shape = vec![1, 144_000];
        let output_shapes = vec![vec![1, 6522], vec![1, 1024]];

        let config = detect_model_type(
            &input_shape,
            &output_shapes,
            &[],
            Some(ModelType::BirdNetV24),
        )
        .unwrap();

        assert_eq!(config.model_type, ModelType::BirdNetV24);
        assert_eq!(config.embedding_dim, Some(1024));
    }
}
