//! Prediction post-processing: top-K selection with sigmoid

use crate::types::Prediction;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Entry for min-heap based top-K selection
struct ScoreEntry {
    index: usize,
    score: f32,
}

impl PartialEq for ScoreEntry {
    fn eq(&self, other: &Self) -> bool {
        // Use total ordering for consistent behavior with NaN values.
        // This ensures PartialEq and Ord are consistent for heap operations.
        self.score.total_cmp(&other.score) == Ordering::Equal
    }
}

impl Eq for ScoreEntry {}

impl PartialOrd for ScoreEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (smallest at top, gets popped first)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Select top-K predictions with sigmoid activation
///
/// Uses min-heap for O(n log k) selection, then applies sigmoid only to top-K.
pub fn top_k_predictions(
    logits: &[f32],
    labels: &[String],
    top_k: usize,
    min_confidence: Option<f32>,
) -> Vec<Prediction> {
    if logits.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let k = top_k.min(logits.len());

    // Use min-heap to find top K logits efficiently
    let mut heap: BinaryHeap<ScoreEntry> = BinaryHeap::with_capacity(k + 1);

    for (index, &score) in logits.iter().enumerate() {
        heap.push(ScoreEntry { index, score });
        if heap.len() > k {
            heap.pop(); // Remove smallest
        }
    }

    // Convert to predictions with sigmoid
    let mut predictions: Vec<Prediction> = heap
        .into_iter()
        .map(|entry| {
            let confidence = sigmoid(entry.score);
            Prediction {
                species: labels
                    .get(entry.index)
                    .cloned()
                    .unwrap_or_else(|| format!("unknown_{}", entry.index)),
                confidence,
                index: entry.index,
            }
        })
        .filter(|p| min_confidence.is_none_or(|min| p.confidence >= min))
        .collect();

    // Sort by confidence descending
    predictions.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    predictions
}

/// Sigmoid activation function
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.0001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_top_k_predictions_basic() {
        let logits = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let labels: Vec<String> = (0..5).map(|i| format!("species_{i}")).collect();

        let predictions = top_k_predictions(&logits, &labels, 3, None);

        assert_eq!(predictions.len(), 3);
        // Should be sorted by confidence (descending)
        assert!(predictions[0].confidence >= predictions[1].confidence);
        assert!(predictions[1].confidence >= predictions[2].confidence);
        // Highest logit (0.9) should be first
        assert_eq!(predictions[0].species, "species_2");
    }

    #[test]
    fn test_top_k_with_min_confidence() {
        let logits = vec![-5.0, 0.0, 5.0]; // sigmoid: ~0.007, 0.5, ~0.993
        let labels: Vec<String> = (0..3).map(|i| format!("species_{i}")).collect();

        let predictions = top_k_predictions(&logits, &labels, 10, Some(0.4));

        // Only species with confidence >= 0.4 should be included
        assert_eq!(predictions.len(), 2);
        for p in &predictions {
            assert!(p.confidence >= 0.4);
        }
    }

    #[test]
    fn test_top_k_larger_than_input() {
        let logits = vec![0.1, 0.2];
        let labels = vec!["a".to_string(), "b".to_string()];

        let predictions = top_k_predictions(&logits, &labels, 100, None);

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_top_k_empty_input() {
        let predictions = top_k_predictions(&[], &[], 10, None);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_top_k_zero_k() {
        let logits = vec![0.1, 0.2, 0.3];
        let labels: Vec<String> = (0..3).map(|i| format!("s{i}")).collect();

        let predictions = top_k_predictions(&logits, &labels, 0, None);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predictions_have_correct_indices() {
        let logits = vec![0.1, 0.9, 0.5];
        let labels = vec!["zero".to_string(), "one".to_string(), "two".to_string()];

        let predictions = top_k_predictions(&logits, &labels, 3, None);

        // Find the prediction for "one" (highest logit)
        let one_pred = predictions.iter().find(|p| p.species == "one").unwrap();
        assert_eq!(one_pred.index, 1);
    }
}
