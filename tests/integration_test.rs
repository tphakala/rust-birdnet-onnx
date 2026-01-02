//! Integration tests for birdnet-onnx
//!
//! These tests require actual ONNX model files.
//! Run with: cargo test -- --ignored

#![allow(clippy::unwrap_used)] // Tests can use unwrap
#![allow(clippy::disallowed_methods)] // Tests can use unwrap
#![allow(clippy::cast_precision_loss)] // Sample rate/count casts are fine
#![allow(clippy::print_stderr)] // Tests can print skip messages
#![allow(clippy::items_after_statements)] // Allow use imports in tests
#![allow(clippy::unnecessary_wraps)] // Tests can return Result for consistency

use birdnet_onnx::{Classifier, ModelType, RangeFilter, Result, calculate_week, init_runtime};
use std::path::Path;

const FIXTURES_DIR: &str = "tests/fixtures";

/// Check if test fixtures are available
fn fixtures_available() -> bool {
    Path::new(FIXTURES_DIR).join("birdnet_v24.onnx").exists()
}

/// Create silent audio segment
fn silent_segment(model_type: ModelType) -> Vec<f32> {
    vec![0.0f32; model_type.sample_count()]
}

/// Create sine wave audio segment
fn sine_wave_segment(model_type: ModelType, frequency: f32) -> Vec<f32> {
    let sample_rate = model_type.sample_rate() as f32;
    let sample_count = model_type.sample_count();

    (0..sample_count)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect()
}

// ============================================================================
// BirdNET v2.4 Tests
// ============================================================================

#[test]
#[ignore = "requires model fixtures"]
fn test_birdnet_v24_load() -> Result<()> {
    if !fixtures_available() {
        eprintln!("Skipping: fixtures not available");
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()?;

    let config = classifier.config();
    assert_eq!(config.model_type, ModelType::BirdNetV24);
    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.sample_count, 144_000);
    assert!(config.embedding_dim.is_none());

    Ok(())
}

#[test]
#[ignore = "requires model fixtures"]
fn test_birdnet_v24_predict() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .top_k(10)
        .build()?;

    let segment = silent_segment(ModelType::BirdNetV24);
    let result = classifier.predict(&segment)?;

    assert_eq!(result.model_type, ModelType::BirdNetV24);
    assert!(result.predictions.len() <= 10);
    assert!(result.embeddings.is_none());
    assert!(!result.raw_scores.is_empty());

    // Predictions should be sorted by confidence
    for window in result.predictions.windows(2) {
        assert!(window[0].confidence >= window[1].confidence);
    }

    Ok(())
}

#[test]
#[ignore = "requires model fixtures"]
fn test_birdnet_v24_predict_batch() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()?;

    let seg1 = silent_segment(ModelType::BirdNetV24);
    let seg2 = sine_wave_segment(ModelType::BirdNetV24, 440.0);
    let seg3 = sine_wave_segment(ModelType::BirdNetV24, 1000.0);

    let segments: Vec<&[f32]> = vec![&seg1, &seg2, &seg3];
    let results = classifier.predict_batch(&segments)?;

    assert_eq!(results.len(), 3);

    for result in &results {
        assert_eq!(result.model_type, ModelType::BirdNetV24);
        assert!(result.embeddings.is_none());
    }

    Ok(())
}

#[test]
#[ignore = "requires model fixtures"]
fn test_birdnet_v24_wrong_input_size() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()?;

    let wrong_size = vec![0.0f32; 100_000]; // Wrong size
    let result = classifier.predict(&wrong_size);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("input size mismatch"));

    Ok(())
}

// ============================================================================
// BirdNET v3.0 Tests
// ============================================================================

#[test]
#[ignore = "requires model fixtures"]
fn test_birdnet_v30_load() -> Result<()> {
    let model_path = format!("{FIXTURES_DIR}/birdnet_v30.onnx");
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping: BirdNET v3.0 model not available");
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v30_labels.csv"))
        .build()?;

    let config = classifier.config();
    assert_eq!(config.model_type, ModelType::BirdNetV30);
    assert_eq!(config.sample_rate, 32000);
    assert_eq!(config.sample_count, 160_000);
    assert!(config.embedding_dim.is_some());

    Ok(())
}

#[test]
#[ignore = "requires model fixtures"]
fn test_birdnet_v30_predict_with_embeddings() -> Result<()> {
    let model_path = format!("{FIXTURES_DIR}/birdnet_v30.onnx");
    if !Path::new(&model_path).exists() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v30_labels.csv"))
        .build()?;

    let segment = silent_segment(ModelType::BirdNetV30);
    let result = classifier.predict(&segment)?;

    assert_eq!(result.model_type, ModelType::BirdNetV30);
    assert!(result.embeddings.is_some());

    let embeddings = result.embeddings.unwrap();
    assert_eq!(embeddings.len(), 1024); // Expected embedding dimension

    Ok(())
}

// ============================================================================
// Perch v2 Tests
// ============================================================================

#[test]
#[ignore = "requires model fixtures"]
fn test_perch_v2_load_with_override() -> Result<()> {
    let model_path = format!("{FIXTURES_DIR}/perch_v2.onnx");
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping: Perch v2 model not available");
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(format!("{FIXTURES_DIR}/perch_v2_labels.json"))
        .model_type(ModelType::PerchV2) // Override detection
        .build()?;

    let config = classifier.config();
    assert_eq!(config.model_type, ModelType::PerchV2);
    assert_eq!(config.sample_rate, 32000);
    assert!(config.embedding_dim.is_some());

    Ok(())
}

#[test]
#[ignore = "requires model fixtures"]
fn test_perch_v2_predict() -> Result<()> {
    let model_path = format!("{FIXTURES_DIR}/perch_v2.onnx");
    if !Path::new(&model_path).exists() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(format!("{FIXTURES_DIR}/perch_v2_labels.json"))
        .model_type(ModelType::PerchV2)
        .build()?;

    let segment = silent_segment(ModelType::PerchV2);
    let result = classifier.predict(&segment)?;

    assert_eq!(result.model_type, ModelType::PerchV2);
    assert!(result.embeddings.is_some());

    Ok(())
}

// ============================================================================
// In-Memory Labels Tests
// ============================================================================

#[test]
#[ignore = "requires model fixtures"]
fn test_in_memory_labels() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    // Load model and get expected label count
    let temp_classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()?;

    let label_count = temp_classifier.config().num_species;
    drop(temp_classifier);

    // Create in-memory labels
    let labels: Vec<String> = (0..label_count).map(|i| format!("Species_{i}")).collect();

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels(labels)
        .build()?;

    let segment = silent_segment(ModelType::BirdNetV24);
    let result = classifier.predict(&segment)?;

    // Check that our custom labels are used
    for pred in &result.predictions {
        assert!(pred.species.starts_with("Species_"));
    }

    Ok(())
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
#[ignore = "requires model fixtures"]
fn test_top_k_configuration() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .top_k(5)
        .build()?;

    let segment = silent_segment(ModelType::BirdNetV24);
    let result = classifier.predict(&segment)?;

    assert!(result.predictions.len() <= 5);

    Ok(())
}

#[test]
#[ignore = "requires model fixtures"]
fn test_min_confidence_configuration() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    let classifier = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .min_confidence(0.5)
        .build()?;

    let segment = silent_segment(ModelType::BirdNetV24);
    let result = classifier.predict(&segment)?;

    // All returned predictions should meet threshold
    for pred in &result.predictions {
        assert!(pred.confidence >= 0.5);
    }

    Ok(())
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
#[ignore = "requires model fixtures"]
fn test_classifier_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Classifier>();
}

#[test]
#[ignore = "requires model fixtures"]
fn test_concurrent_predictions() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    use std::sync::Arc;
    use std::thread;

    let classifier = Arc::new(
        Classifier::builder()
            .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
            .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
            .build()?,
    );

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let clf = Arc::clone(&classifier);
            thread::spawn(move || {
                let segment = vec![0.0f32; 144_000];
                for _ in 0..10 {
                    let result = clf.predict(&segment).unwrap();
                    assert_eq!(result.model_type, ModelType::BirdNetV24);
                }
                i
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_missing_model_path() {
    let result = Classifier::builder().labels_path("labels.txt").build();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("model path required")
    );
}

#[test]
fn test_missing_labels() {
    let result = Classifier::builder().model_path("model.onnx").build();

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("labels required"));
}

#[test]
#[ignore = "requires model fixtures"]
fn test_nonexistent_model_file() {
    let result = Classifier::builder()
        .model_path("/nonexistent/model.onnx")
        .labels_path("labels.txt")
        .build();

    assert!(result.is_err());
}

#[test]
#[ignore = "requires model fixtures"]
fn test_label_count_mismatch() -> Result<()> {
    if !fixtures_available() {
        return Ok(());
    }

    // Provide wrong number of labels
    let labels = vec!["only_one".to_string()];

    let result = Classifier::builder()
        .model_path(format!("{FIXTURES_DIR}/birdnet_v24.onnx"))
        .labels(labels)
        .build();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("label count mismatch")
    );

    Ok(())
}

#[test]
#[allow(clippy::expect_used, clippy::print_stdout)]
fn test_range_filter_with_real_model() {
    init_runtime().expect("failed to init runtime");

    // Get model path from environment variable (for local testing only)
    // CI will skip this test since BIRDNET_META_MODEL won't be set
    let Ok(model_path) = std::env::var("BIRDNET_META_MODEL") else {
        eprintln!("Skipping test: BIRDNET_META_MODEL environment variable not set");
        eprintln!(
            "To run this test locally: export BIRDNET_META_MODEL=/path/to/birdnet_data_model.onnx"
        );
        return;
    };

    // Skip if model doesn't exist
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping test: meta model not found at {model_path}");
        return;
    }

    // Load labels from BirdNET v2.4 test fixture
    let labels_path = "tests/fixtures/birdnet_v24/labels.txt";
    let labels_content = std::fs::read_to_string(labels_path).expect("failed to read labels");
    let labels: Vec<String> = labels_content.lines().map(String::from).collect();

    let range_filter = RangeFilter::builder()
        .model_path(&model_path)
        .labels(labels)
        .threshold(0.01)
        .build()
        .expect("failed to build range filter");

    // Test prediction for Helsinki, Finland in June
    let scores = range_filter
        .predict(60.1695, 24.9354, 6, 15)
        .expect("prediction failed");

    assert!(!scores.is_empty(), "should return some species");

    // Verify scores are sorted descending
    for i in 1..scores.len() {
        assert!(
            scores[i - 1].score >= scores[i].score,
            "scores should be sorted descending"
        );
    }

    // Verify all scores are above threshold
    for score in &scores {
        assert!(score.score >= 0.01, "all scores should be >= threshold");
    }

    println!("Range filter returned {} species", scores.len());
    if !scores.is_empty() {
        println!(
            "Top species: {} (score: {:.4})",
            scores[0].species, scores[0].score
        );
    }
}

#[test]
#[allow(clippy::expect_used)]
fn test_range_filter_invalid_coordinates() {
    init_runtime().expect("failed to init runtime");

    // Get model path from environment variable
    let Ok(model_path) = std::env::var("BIRDNET_META_MODEL") else {
        return;
    };

    if !std::path::Path::new(&model_path).exists() {
        return;
    }

    let labels = vec!["Test_Species".to_string()];
    let range_filter = RangeFilter::builder()
        .model_path(&model_path)
        .labels(labels)
        .build()
        .expect("failed to build");

    // Test invalid latitude
    let result = range_filter.predict(95.0, 0.0, 1, 1);
    assert!(result.is_err());

    // Test invalid longitude
    let result = range_filter.predict(0.0, 190.0, 1, 1);
    assert!(result.is_err());
}

#[test]
#[allow(clippy::float_cmp)]
fn test_calculate_week_values() {
    // January 1st = week 1
    assert_eq!(calculate_week(1, 1), 1.0);

    // February 1st = week 5 (4 weeks in Jan + 1)
    assert_eq!(calculate_week(2, 1), 5.0);

    // December 1st = week 45 (44 weeks + 1)
    assert_eq!(calculate_week(12, 1), 45.0);
}
