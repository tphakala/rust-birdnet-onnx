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
/// family needs (sigmoid, softmax for `Perch` v2, none for `BirdNET` v3.0 and
/// the BSG Finland head, whose graphs already apply the sigmoid), selects the
/// top-K species, applies the confidence threshold and slices embeddings per
/// batch row.
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
    /// One [`PredictionResult`] per batch row, in row order. A NaN score (and, for
    /// models whose output is already a probability, an infinite one) ranks as
    /// the lowest possible score, so it never displaces a usable class and does
    /// not turn a softmax row into NaN. Such a class, and any prediction whose
    /// confidence is not finite, is never listed in `predictions`. `raw_scores`
    /// always holds the row as the model returned it.
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
    /// // BirdNET v3.0 scores are already probabilities and pass through unchanged.
    /// let scores = [0.9, 0.1, 0.2, 0.8];
    /// let embeddings = [0.1, 0.2, 0.3, 0.4];
    /// let results = decoder.decode_batch(2, &scores, Some(&embeddings))?;
    /// assert_eq!(results[1].predictions[0].confidence, 0.8);
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

        let mut results = Vec::new();
        results.try_reserve_exact(batch_size).map_err(|_| {
            Error::Inference(format!(
                "cannot allocate results for batch size {batch_size}"
            ))
        })?;
        for row in 0..batch_size {
            let row_scores = &scores[row * num_species..(row + 1) * num_species];
            let row_embeddings = embeddings.zip(embedding_dim).map(|(emb, dim)| {
                // In range: `check_len` established emb.len() == batch_size * dim.
                emb[row * dim..(row + 1) * dim].to_vec()
            });

            results.push(PredictionResult {
                model_type: self.config.model_type,
                predictions: self.select_top_k(row_scores),
                embeddings: row_embeddings,
                raw_scores: row_scores.to_vec(),
            });
        }
        Ok(results)
    }

    /// Select top-K predictions, choosing the correct score transform for the model type.
    fn select_top_k(&self, scores: &[f32]) -> Vec<Prediction> {
        let presigmoid = self.config.model_type.output_is_sigmoid();
        // Ranking uses `total_cmp`, which puts NaN above every number, so a NaN
        // score would take a top-K slot, and one NaN in a softmax row turns
        // every probability into NaN. Rank such a score as the lowest possible
        // value instead. For already-sigmoided output an infinite value is not
        // a probability either, so it is ranked lowest too.
        let unusable = |score: f32| score.is_nan() || (presigmoid && score.is_infinite());
        let original = scores;
        let unusable_count = scores.iter().filter(|&&score| unusable(score)).count();
        let cleaned: Vec<f32>;
        let scores = if unusable_count > 0 {
            cleaned = scores
                .iter()
                .map(|&score| {
                    if unusable(score) {
                        f32::NEG_INFINITY
                    } else {
                        score
                    }
                })
                .collect();
            &cleaned[..]
        } else {
            scores
        };

        // An unusable score ties with a genuine negative-infinity score, so ask
        // for one extra place per unusable score: every usable class that
        // belongs in the top K is then in the list before they are removed.
        let wanted = self.top_k.saturating_add(unusable_count);
        let mut predictions = if presigmoid {
            top_k_predictions_presigmoid(scores, &self.labels, wanted, self.min_confidence)
        } else if self.config.model_type.output_needs_softmax() {
            top_k_predictions_softmax(scores, &self.labels, wanted, self.min_confidence)
        } else {
            top_k_predictions(scores, &self.labels, wanted, self.min_confidence)
        };
        // Leave out a class whose own score was unusable and anything whose
        // confidence is not finite (a softmax row with an infinite score).
        predictions.retain(|prediction| {
            prediction.confidence.is_finite()
                && original
                    .get(prediction.index)
                    .is_some_and(|&score| !unusable(score))
        });
        predictions.truncate(self.top_k);
        predictions
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
            Case {
                name: "pre-sigmoid, BirdNET v3.0 with embeddings",
                model_type: ModelType::BirdNetV30,
                embedding_dim: Some(2),
                top_k: 2,
                min_confidence: None,
                scores: [0.02, 0.97, 0.001, 0.6, 0.04, 0.3],
                embeddings: Some([0.1, 0.2, 0.3, 0.4]),
                row0: vec![("Wren", 1, 0.97), ("Robin", 0, 0.02)],
                row1: vec![("Robin", 0, 0.6), ("Owl", 2, 0.3)],
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

    /// Compare only the predictions; `raw_scores` may hold NaN, which never equals itself.
    fn assert_predictions(actual: &PredictionResult, want: &[Expected]) {
        let got: Vec<(&str, usize)> = actual
            .predictions
            .iter()
            .map(|p| (p.species.as_str(), p.index))
            .collect();
        let expected: Vec<(&str, usize)> = want.iter().map(|&(s, i, _)| (s, i)).collect();
        assert_eq!(got, expected);
        for (p, &(_, _, confidence)) in actual.predictions.iter().zip(want) {
            assert!(
                (p.confidence - confidence).abs() < TOLERANCE,
                "{}: confidence {} != {confidence}",
                p.species,
                p.confidence
            );
        }
    }

    #[test]
    fn decode_batch_ranks_a_nan_score_below_every_usable_class() {
        // Softmax: one NaN would make every probability NaN; the other class
        // keeps its probability instead.
        let perch = PredictionDecoder::new(
            config(ModelType::PerchV2, 2, None),
            labels()[..2].to_vec(),
            1,
            None,
        )
        .unwrap();
        let results = perch.decode_batch(1, &[0.0, f32::NAN], None).unwrap();
        assert_predictions(&results[0], &[("Robin", 0, 1.0)]);
        assert!(results[0].raw_scores[1].is_nan());

        // Sigmoid: the NaN must not take one of the two slots.
        let v24 = PredictionDecoder::new(config(ModelType::BirdNetV24, 3, None), labels(), 2, None)
            .unwrap();
        let results = v24.decode_batch(1, &[f32::NAN, 1.0, 2.0], None).unwrap();
        assert_predictions(
            &results[0],
            &[("Owl", 2, 0.880_797_1), ("Wren", 1, 0.731_058_6)],
        );

        // Already-sigmoided output: an infinite value is not a probability.
        let bsg = PredictionDecoder::new(config(ModelType::BsgFinland, 3, None), labels(), 2, None)
            .unwrap();
        let results = bsg
            .decode_batch(1, &[f32::INFINITY, 0.25, 0.5], None)
            .unwrap();
        assert_predictions(&results[0], &[("Owl", 2, 0.5), ("Wren", 1, 0.25)]);

        // Fewer usable classes than top_k: the unusable one is left out.
        let results = v24
            .decode_batch(1, &[f32::NAN, 1.0, f32::NAN], None)
            .unwrap();
        assert_predictions(&results[0], &[("Wren", 1, 0.731_058_6)]);

        // A NaN ties with a genuine negative-infinity logit; the logit must win.
        let one = PredictionDecoder::new(
            config(ModelType::BirdNetV24, 2, None),
            labels()[..2].to_vec(),
            1,
            None,
        )
        .unwrap();
        let results = one
            .decode_batch(1, &[f32::NEG_INFINITY, f32::NAN], None)
            .unwrap();
        assert_predictions(&results[0], &[("Robin", 0, 0.0)]);
        let results = one
            .decode_batch(1, &[f32::NAN, f32::NEG_INFINITY], None)
            .unwrap();
        assert_predictions(&results[0], &[("Wren", 1, 0.0)]);
    }

    #[test]
    fn decode_batch_reports_a_result_vector_that_cannot_be_allocated() {
        // A zero-width model accepts an empty scores buffer for any batch size,
        // so only the result allocation can fail.
        let decoder =
            PredictionDecoder::new(config(ModelType::BirdNetV24, 0, None), Vec::new(), 1, None)
                .unwrap();

        let error = decoder.decode_batch(usize::MAX, &[], None).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "inference failed: cannot allocate results for batch size {}",
                usize::MAX
            )
        );
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
        // `usize::MAX` differs between 32-bit and 64-bit targets.
        let overflow_message = format!(
            "inference failed: scores length overflows: batch size {} x width 3",
            usize::MAX
        );
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
            (usize::MAX, 6, None, overflow_message.as_str()),
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
