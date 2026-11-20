//! Engine-neutral decoding of raw model output into [`PredictionResult`]s.
//!
//! Which activation a model family needs, how class scores and embeddings are
//! sliced per batch row, and how the top-K and confidence threshold apply are
//! properties of the model, not of the runtime that executed it. They live
//! here so that [`Classifier`](crate::Classifier) and any other engine that
//! runs the same models share one implementation.

use crate::error::{Error, Result};
use crate::postprocess::{
    top_k_predictions, top_k_predictions_presigmoid, top_k_predictions_softmax,
};
use crate::types::{ModelConfig, Prediction, PredictionResult};

/// Turns flat output buffers from a model run into [`PredictionResult`]s.
///
/// The decoder holds no inference state. It applies the activation the model
/// family needs (sigmoid, softmax for `Perch` v2, none for the pre-sigmoided
/// BSG Finland head), selects the top-K species, applies the confidence
/// threshold and slices embeddings per batch row.
///
/// # Example
///
/// ```
/// use birdnet_onnx::{ModelConfig, ModelType, PredictionDecoder};
///
/// let config = ModelConfig {
///     model_type: ModelType::BirdNetV24,
///     sample_rate: 48_000,
///     segment_duration: 3.0,
///     sample_count: 144_000,
///     num_species: 3,
///     embedding_dim: None,
///     predictions_index: 0,
///     embeddings_index: None,
/// };
/// let labels = vec!["Robin".to_string(), "Wren".to_string(), "Owl".to_string()];
/// let decoder = PredictionDecoder::new(config, labels, 1, None)?;
///
/// // Two batch rows of three raw scores each.
/// let results = decoder.decode_batch(2, &[0.0, 2.0, -2.0, 3.0, 0.0, 0.0], None)?;
/// assert_eq!(results[0].predictions[0].species, "Wren");
/// assert_eq!(results[1].predictions[0].species, "Robin");
/// # Ok::<(), birdnet_onnx::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct PredictionDecoder {
    config: ModelConfig,
    labels: Vec<String>,
    top_k: usize,
    min_confidence: Option<f32>,
}

impl PredictionDecoder {
    /// Create a decoder for a detected model.
    ///
    /// # Arguments
    /// * `config` - Model configuration, as returned by
    ///   [`detect_model_type`](crate::detect_model_type)
    /// * `labels` - Species labels, one per model output class
    /// * `top_k` - Number of predictions to keep per segment
    /// * `min_confidence` - Optional confidence threshold applied after the
    ///   activation; predictions below it are dropped
    ///
    /// # Errors
    ///
    /// Returns [`Error::LabelCount`] if `labels.len()` differs from
    /// `config.num_species`.
    ///
    /// # Example
    ///
    /// ```
    /// use birdnet_onnx::{Error, ModelConfig, ModelType, PredictionDecoder};
    ///
    /// let config = ModelConfig {
    ///     model_type: ModelType::BirdNetV24,
    ///     sample_rate: 48_000,
    ///     segment_duration: 3.0,
    ///     sample_count: 144_000,
    ///     num_species: 3,
    ///     embedding_dim: None,
    ///     predictions_index: 0,
    ///     embeddings_index: None,
    /// };
    /// let err = PredictionDecoder::new(config, vec!["Robin".to_string()], 5, None).unwrap_err();
    /// assert!(matches!(err, Error::LabelCount { expected: 3, got: 1 }));
    /// ```
    pub fn new(
        config: ModelConfig,
        labels: Vec<String>,
        top_k: usize,
        min_confidence: Option<f32>,
    ) -> Result<Self> {
        if labels.len() != config.num_species {
            return Err(Error::LabelCount {
                expected: config.num_species,
                got: labels.len(),
            });
        }
        Ok(Self {
            config,
            labels,
            top_k,
            min_confidence,
        })
    }

    /// The model configuration this decoder was built with.
    ///
    /// # Example
    ///
    /// ```
    /// use birdnet_onnx::{ModelConfig, ModelType, PredictionDecoder};
    ///
    /// let config = ModelConfig {
    ///     model_type: ModelType::BsgFinland,
    ///     sample_rate: 48_000,
    ///     segment_duration: 3.0,
    ///     sample_count: 144_000,
    ///     num_species: 1,
    ///     embedding_dim: None,
    ///     predictions_index: 0,
    ///     embeddings_index: None,
    /// };
    /// let decoder = PredictionDecoder::new(config, vec!["Robin".to_string()], 1, None)?;
    /// assert_eq!(decoder.config().model_type, ModelType::BsgFinland);
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The species labels this decoder maps class indices to.
    ///
    /// # Example
    ///
    /// ```
    /// use birdnet_onnx::{ModelConfig, ModelType, PredictionDecoder};
    ///
    /// let config = ModelConfig {
    ///     model_type: ModelType::BirdNetV24,
    ///     sample_rate: 48_000,
    ///     segment_duration: 3.0,
    ///     sample_count: 144_000,
    ///     num_species: 1,
    ///     embedding_dim: None,
    ///     predictions_index: 0,
    ///     embeddings_index: None,
    /// };
    /// let decoder = PredictionDecoder::new(config, vec!["Robin".to_string()], 1, None)?;
    /// assert_eq!(decoder.labels(), ["Robin".to_string()]);
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Number of predictions kept per segment.
    pub(crate) const fn top_k(&self) -> usize {
        self.top_k
    }

    /// Confidence threshold applied after the activation, if any.
    pub(crate) const fn min_confidence(&self) -> Option<f32> {
        self.min_confidence
    }

    /// Decode the flat output buffers of one model run.
    ///
    /// # Arguments
    /// * `batch_size` - Number of segments in the run
    /// * `scores` - Raw class scores, row-major, `batch_size * num_species`
    ///   values
    /// * `embeddings` - Embeddings, row-major, `batch_size * embedding_dim`
    ///   values, or `None` when the caller has none. `None` yields
    ///   `embeddings: None` in every result.
    ///
    /// # Returns
    /// One [`PredictionResult`] per batch row, in row order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Inference`] if `scores.len()` is not
    /// `batch_size * num_species`, if `embeddings` is given and its length is
    /// not `batch_size * embedding_dim`, or if `embeddings` is given for a
    /// model config that has no `embedding_dim`. It never panics on a buffer
    /// of the wrong length.
    ///
    /// # Example
    ///
    /// ```
    /// use birdnet_onnx::{ModelConfig, ModelType, PredictionDecoder};
    ///
    /// let config = ModelConfig {
    ///     model_type: ModelType::BirdNetV30,
    ///     sample_rate: 32_000,
    ///     segment_duration: 5.0,
    ///     sample_count: 160_000,
    ///     num_species: 2,
    ///     embedding_dim: Some(2),
    ///     predictions_index: 0,
    ///     embeddings_index: Some(1),
    /// };
    /// let labels = vec!["Robin".to_string(), "Wren".to_string()];
    /// let decoder = PredictionDecoder::new(config, labels, 2, None)?;
    ///
    /// let scores = [1.0, -1.0, -1.0, 1.0];
    /// let embeddings = [0.1, 0.2, 0.3, 0.4];
    /// let results = decoder.decode_batch(2, &scores, Some(&embeddings))?;
    /// assert_eq!(results[1].embeddings, Some(vec![0.3, 0.4]));
    ///
    /// // A short buffer is an error, not a panic.
    /// assert!(decoder.decode_batch(2, &scores[..3], None).is_err());
    /// # Ok::<(), birdnet_onnx::Error>(())
    /// ```
    pub fn decode_batch(
        &self,
        batch_size: usize,
        scores: &[f32],
        embeddings: Option<&[f32]>,
    ) -> Result<Vec<PredictionResult>> {
        let num_species = self.config.num_species;
        check_len("scores", batch_size, num_species, scores.len())?;

        let embedding_dim = match (embeddings, self.config.embedding_dim) {
            (Some(emb), Some(dim)) => {
                check_len("embeddings", batch_size, dim, emb.len())?;
                Some(dim)
            }
            (Some(_), None) => {
                return Err(Error::Inference(
                    "embeddings supplied but the model config has no embedding_dim".into(),
                ));
            }
            (None, _) => None,
        };

        Ok((0..batch_size)
            .map(|row| {
                let row_scores = &scores[row * num_species..(row + 1) * num_species];
                let row_embeddings = embeddings.zip(embedding_dim).map(|(emb, dim)| {
                    // In range: `check_len` established emb.len() == batch_size * dim.
                    emb[row * dim..(row + 1) * dim].to_vec()
                });

                PredictionResult {
                    model_type: self.config.model_type,
                    predictions: self.select_top_k(row_scores),
                    embeddings: row_embeddings,
                    raw_scores: row_scores.to_vec(),
                }
            })
            .collect())
    }

    /// Select top-K predictions, choosing the correct score transform for the model type.
    fn select_top_k(&self, scores: &[f32]) -> Vec<Prediction> {
        if self.config.model_type.output_is_sigmoid() {
            top_k_predictions_presigmoid(scores, &self.labels, self.top_k, self.min_confidence)
        } else if self.config.model_type.output_needs_softmax() {
            top_k_predictions_softmax(scores, &self.labels, self.top_k, self.min_confidence)
        } else {
            top_k_predictions(scores, &self.labels, self.top_k, self.min_confidence)
        }
    }
}

/// Check that a flat buffer holds exactly `batch_size * width` values.
fn check_len(what: &str, batch_size: usize, width: usize, got: usize) -> Result<()> {
    let expected = batch_size.checked_mul(width).ok_or_else(|| {
        Error::Inference(format!(
            "{what} length overflows: batch size {batch_size} x width {width}"
        ))
    })?;
    if got == expected {
        Ok(())
    } else {
        Err(Error::Inference(format!(
            "{what} length mismatch: expected {expected} values (batch size {batch_size} x width {width}), got {got}"
        )))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use crate::types::ModelType;

    const TOLERANCE: f32 = 1e-6;

    fn config(
        model_type: ModelType,
        num_species: usize,
        embedding_dim: Option<usize>,
    ) -> ModelConfig {
        ModelConfig {
            model_type,
            sample_rate: model_type.sample_rate(),
            segment_duration: model_type.segment_duration(),
            sample_count: model_type.sample_count(),
            num_species,
            embedding_dim,
            predictions_index: 0,
            embeddings_index: embedding_dim.map(|_| 1),
        }
    }

    fn labels() -> Vec<String> {
        ["Robin", "Wren", "Owl"].map(String::from).to_vec()
    }

    /// One expected prediction: species, class index, confidence.
    type Expected = (&'static str, usize, f32);

    fn assert_row(
        actual: &PredictionResult,
        model_type: ModelType,
        predictions: &[Expected],
        embeddings: Option<&[f32]>,
        raw_scores: &[f32],
    ) {
        assert_eq!(actual.model_type, model_type);
        assert_eq!(actual.raw_scores, raw_scores);
        assert_eq!(actual.embeddings.as_deref(), embeddings);

        let got: Vec<(&str, usize)> = actual
            .predictions
            .iter()
            .map(|p| (p.species.as_str(), p.index))
            .collect();
        let want: Vec<(&str, usize)> = predictions.iter().map(|&(s, i, _)| (s, i)).collect();
        assert_eq!(got, want);
        for (p, &(species, _, confidence)) in actual.predictions.iter().zip(predictions) {
            assert!(
                (p.confidence - confidence).abs() < TOLERANCE,
                "{species}: confidence {} differs from {confidence}",
                p.confidence
            );
        }
    }

    /// One row of the activation table: the same two-row buffer decoded under
    /// one model family.
    struct Case {
        name: &'static str,
        model_type: ModelType,
        embedding_dim: Option<usize>,
        top_k: usize,
        min_confidence: Option<f32>,
        scores: [f32; 6],
        embeddings: Option<[f32; 4]>,
        row0: Vec<Expected>,
        row1: Vec<Expected>,
    }

    #[test]
    fn decode_batch_applies_the_family_activation_per_row() {
        let cases = [
            Case {
                name: "sigmoid, BirdNET v2.4 with embeddings",
                model_type: ModelType::BirdNetV24,
                embedding_dim: Some(2),
                top_k: 2,
                min_confidence: None,
                scores: [0.0, 2.0, -2.0, 3.0, -1.0, 1.0],
                embeddings: Some([0.1, 0.2, 0.3, 0.4]),
                row0: vec![("Wren", 1, 0.880_797_1), ("Robin", 0, 0.5)],
                row1: vec![("Robin", 0, 0.952_574_1), ("Owl", 2, 0.731_058_6)],
            },
            Case {
                name: "softmax, Perch v2 with embeddings",
                model_type: ModelType::PerchV2,
                embedding_dim: Some(2),
                top_k: 2,
                min_confidence: None,
                scores: [0.0, 2.0, -2.0, 1.0, -1.0, 3.0],
                embeddings: Some([0.1, 0.2, 0.3, 0.4]),
                row0: vec![("Wren", 1, 0.866_813_3), ("Robin", 0, 0.117_310_43)],
                row1: vec![("Owl", 2, 0.866_813_3), ("Robin", 0, 0.117_310_43)],
            },
            Case {
                name: "pre-sigmoid, BSG Finland without embeddings",
                model_type: ModelType::BsgFinland,
                embedding_dim: None,
                top_k: 3,
                min_confidence: Some(0.4),
                scores: [0.9, 0.1, 0.5, 0.2, 0.3, 0.45],
                embeddings: None,
                row0: vec![("Robin", 0, 0.9), ("Owl", 2, 0.5)],
                row1: vec![("Owl", 2, 0.45)],
            },
        ];

        for case in cases {
            let decoder = PredictionDecoder::new(
                config(case.model_type, 3, case.embedding_dim),
                labels(),
                case.top_k,
                case.min_confidence,
            )
            .unwrap();

            let results = decoder
                .decode_batch(
                    2,
                    &case.scores,
                    case.embeddings.as_ref().map(<[f32; 4]>::as_slice),
                )
                .unwrap();

            assert_eq!(results.len(), 2, "{}", case.name);
            let (emb0, emb1) = case.embeddings.map_or((None, None), |e| {
                (Some(e[..2].to_vec()), Some(e[2..].to_vec()))
            });
            assert_row(
                &results[0],
                case.model_type,
                &case.row0,
                emb0.as_deref(),
                &case.scores[..3],
            );
            assert_row(
                &results[1],
                case.model_type,
                &case.row1,
                emb1.as_deref(),
                &case.scores[3..],
            );
        }
    }

    #[test]
    fn decode_batch_without_embeddings_leaves_them_unset() {
        let decoder =
            PredictionDecoder::new(config(ModelType::BirdNetV30, 3, Some(2)), labels(), 1, None)
                .unwrap();

        let results = decoder
            .decode_batch(2, &[0.0, 2.0, -2.0, 3.0, -1.0, 1.0], None)
            .unwrap();

        assert!(results.iter().all(|r| r.embeddings.is_none()));
    }

    #[test]
    fn decode_batch_of_zero_rows_is_empty() {
        let decoder =
            PredictionDecoder::new(config(ModelType::BirdNetV24, 3, None), labels(), 1, None)
                .unwrap();

        assert!(decoder.decode_batch(0, &[], None).unwrap().is_empty());
    }

    #[test]
    fn decode_batch_rejects_buffers_of_the_wrong_length() {
        // (batch size, scores length, embeddings length, expected message)
        let cases: [(usize, usize, Option<usize>, &str); 6] = [
            (
                2,
                5,
                None,
                "inference failed: scores length mismatch: expected 6 values (batch size 2 x width 3), got 5",
            ),
            (
                2,
                7,
                None,
                "inference failed: scores length mismatch: expected 6 values (batch size 2 x width 3), got 7",
            ),
            (
                2,
                6,
                Some(3),
                "inference failed: embeddings length mismatch: expected 4 values (batch size 2 x width 2), got 3",
            ),
            (
                2,
                6,
                Some(5),
                "inference failed: embeddings length mismatch: expected 4 values (batch size 2 x width 2), got 5",
            ),
            (
                usize::MAX,
                6,
                None,
                "inference failed: scores length overflows: batch size 18446744073709551615 x width 3",
            ),
            (
                2,
                6,
                Some(0),
                "inference failed: embeddings length mismatch: expected 4 values (batch size 2 x width 2), got 0",
            ),
        ];
        let decoder =
            PredictionDecoder::new(config(ModelType::BirdNetV30, 3, Some(2)), labels(), 1, None)
                .unwrap();

        for (batch_size, scores_len, embeddings_len, message) in cases {
            let scores = vec![0.5; scores_len];
            let embeddings = embeddings_len.map(|n| vec![0.5; n]);

            let err = decoder
                .decode_batch(batch_size, &scores, embeddings.as_deref())
                .unwrap_err();

            assert_eq!(err.to_string(), message);
        }
    }

    #[test]
    fn decode_batch_rejects_embeddings_for_a_model_without_embedding_dim() {
        let decoder =
            PredictionDecoder::new(config(ModelType::BirdNetV24, 3, None), labels(), 1, None)
                .unwrap();

        let err = decoder
            .decode_batch(1, &[0.0, 1.0, 2.0], Some(&[0.1, 0.2]))
            .unwrap_err();

        assert_eq!(
            err.to_string(),
            "inference failed: embeddings supplied but the model config has no embedding_dim"
        );
    }

    #[test]
    fn new_rejects_a_label_count_that_differs_from_the_model() {
        // (label count, expected message) for a model with three species
        let cases = [
            (2, "label count mismatch: model expects 3, got 2"),
            (4, "label count mismatch: model expects 3, got 4"),
        ];

        for (label_count, message) in cases {
            let labels = vec!["Robin".to_string(); label_count];

            let result =
                PredictionDecoder::new(config(ModelType::BirdNetV24, 3, None), labels, 1, None);

            assert_eq!(
                result.err().map(|e| e.to_string()),
                Some(message.to_string())
            );
        }
    }
}
