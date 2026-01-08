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

#[cfg(feature = "load-dynamic")]
use birdnet_onnx::init_runtime;
use birdnet_onnx::{Classifier, InferenceOptions, ModelType, RangeFilter, Result};
use std::path::Path;

const FIXTURES_DIR: &str = "tests/fixtures";

/// Check if test fixtures are available
fn fixtures_available() -> bool {
    Path::new(FIXTURES_DIR).join("birdnet_v24.onnx").exists()
}

/// Get Perch v2 test assets from environment variables
fn get_perch_v2_test_assets() -> Option<(String, String)> {
    let Ok(model_path) = std::env::var("PERCH_V2_MODEL") else {
        eprintln!("Skipping test: PERCH_V2_MODEL environment variable not set");
        eprintln!("To run this test locally: export PERCH_V2_MODEL=/path/to/perch-v2.onnx");
        return None;
    };

    let Ok(labels_path) = std::env::var("PERCH_V2_LABELS") else {
        eprintln!("Skipping test: PERCH_V2_LABELS environment variable not set");
        eprintln!("To run this test locally: export PERCH_V2_LABELS=/path/to/perch-v2.csv");
        return None;
    };

    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return None;
    }
    if !Path::new(&labels_path).exists() {
        eprintln!("Skipping test: labels not found at {labels_path}");
        return None;
    }

    Some((model_path, labels_path))
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
    let result = classifier.predict(&segment, &InferenceOptions::default())?;

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
    let results = classifier.predict_batch(&segments, &InferenceOptions::default())?;

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
    let result = classifier.predict(&wrong_size, &InferenceOptions::default());

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
    let result = classifier.predict(&segment, &InferenceOptions::default())?;

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
        .labels_path(format!("{FIXTURES_DIR}/perch_v2_labels.csv"))
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
        .labels_path(format!("{FIXTURES_DIR}/perch_v2_labels.csv"))
        .model_type(ModelType::PerchV2)
        .build()?;

    let segment = silent_segment(ModelType::PerchV2);
    let result = classifier.predict(&segment, &InferenceOptions::default())?;

    assert_eq!(result.model_type, ModelType::PerchV2);
    assert!(result.embeddings.is_some());

    Ok(())
}

#[test]
#[allow(clippy::expect_used, clippy::print_stdout, clippy::float_cmp)]
fn test_perch_v2_auto_detection() {
    #[cfg(feature = "load-dynamic")]
    init_runtime().expect("failed to init runtime");

    let Some((model_path, labels_path)) = get_perch_v2_test_assets() else {
        return;
    };

    // Load classifier WITHOUT override - should auto-detect Perch v2
    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(&labels_path)
        .build()
        .expect("failed to build classifier");

    let config = classifier.config();
    assert_eq!(
        config.model_type,
        ModelType::PerchV2,
        "should auto-detect Perch v2"
    );
    assert_eq!(config.sample_rate, 32_000);
    assert_eq!(config.segment_duration, 5.0);
    assert_eq!(config.sample_count, 160_000);
    assert!(config.embedding_dim.is_some());

    println!("✓ Perch v2 auto-detection successful");
    println!("  Model type: {:?}", config.model_type);
    println!("  Species count: {}", config.num_species);
    println!("  Embedding dim: {:?}", config.embedding_dim);
}

#[test]
#[allow(clippy::expect_used, clippy::print_stdout)]
fn test_perch_v2_predict_real_model() {
    #[cfg(feature = "load-dynamic")]
    init_runtime().expect("failed to init runtime");

    let Some((model_path, labels_path)) = get_perch_v2_test_assets() else {
        return;
    };

    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(&labels_path)
        .top_k(10)
        .min_confidence(0.1)
        .build()
        .expect("failed to build classifier");

    // Test with silent audio
    let segment = silent_segment(ModelType::PerchV2);
    let result = classifier
        .predict(&segment, &InferenceOptions::default())
        .expect("prediction failed");

    assert_eq!(result.model_type, ModelType::PerchV2);
    assert!(result.embeddings.is_some(), "should have embeddings");
    assert!(result.predictions.len() <= 10, "should respect top_k");
    assert!(!result.raw_scores.is_empty(), "should have raw scores");

    // Verify predictions are sorted by confidence
    for window in result.predictions.windows(2) {
        assert!(window[0].confidence >= window[1].confidence);
    }

    // Verify all predictions meet min_confidence
    for pred in &result.predictions {
        assert!(pred.confidence >= 0.1);
    }

    println!("✓ Perch v2 prediction successful");
    println!("  Predictions: {}", result.predictions.len());
    if let Some(embeddings) = &result.embeddings {
        println!("  Embedding dim: {}", embeddings.len());
    }
}

#[test]
#[allow(clippy::expect_used)]
fn test_perch_v2_batch_predict() {
    #[cfg(feature = "load-dynamic")]
    init_runtime().expect("failed to init runtime");

    let Some((model_path, labels_path)) = get_perch_v2_test_assets() else {
        return;
    };

    let classifier = Classifier::builder()
        .model_path(&model_path)
        .labels_path(&labels_path)
        .build()
        .expect("failed to build classifier");

    // Create multiple segments
    let seg1 = silent_segment(ModelType::PerchV2);
    let seg2 = sine_wave_segment(ModelType::PerchV2, 440.0);
    let seg3 = sine_wave_segment(ModelType::PerchV2, 1000.0);

    let segments: Vec<&[f32]> = vec![&seg1, &seg2, &seg3];
    let results = classifier
        .predict_batch(&segments, &InferenceOptions::default())
        .expect("batch prediction failed");

    assert_eq!(results.len(), 3, "should return 3 results");

    for result in &results {
        assert_eq!(result.model_type, ModelType::PerchV2);
        assert!(
            result.embeddings.is_some(),
            "all results should have embeddings"
        );
    }
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
    let result = classifier.predict(&segment, &InferenceOptions::default())?;

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
    let result = classifier.predict(&segment, &InferenceOptions::default())?;

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
    let result = classifier.predict(&segment, &InferenceOptions::default())?;

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
                    let result = clf.predict(&segment, &InferenceOptions::default()).unwrap();
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
    #[cfg(feature = "load-dynamic")]
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

    // Load labels from BirdNET v2.4 labels file
    let labels_path = "data/labels/birdnet_v2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt";
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
fn test_range_filter_invalid_inputs() {
    #[cfg(feature = "load-dynamic")]
    init_runtime().expect("failed to init runtime");

    // Get model path from environment variable
    let Ok(model_path) = std::env::var("BIRDNET_META_MODEL") else {
        return;
    };

    if !std::path::Path::new(&model_path).exists() {
        return;
    }

    // Load actual labels to match model output dimension
    let labels_path = format!("{FIXTURES_DIR}/birdnet_v24_labels.txt");
    let labels_content = std::fs::read_to_string(&labels_path).expect("failed to read labels");
    let labels: Vec<String> = labels_content.lines().map(String::from).collect();

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

    // Test invalid month (zero)
    let result = range_filter.predict(0.0, 0.0, 0, 1);
    assert!(result.is_err());

    // Test invalid month (> 12)
    let result = range_filter.predict(0.0, 0.0, 13, 1);
    assert!(result.is_err());

    // Test invalid day (zero)
    let result = range_filter.predict(0.0, 0.0, 1, 0);
    assert!(result.is_err());

    // Test invalid day (> 31)
    let result = range_filter.predict(0.0, 0.0, 1, 32);
    assert!(result.is_err());
}

#[test]
#[allow(clippy::expect_used)]
fn test_range_filter_from_classifier_labels() {
    #[cfg(feature = "load-dynamic")]
    init_runtime().expect("failed to init runtime");

    // Get model paths from environment variables
    let Ok(model_path) = std::env::var("BIRDNET_META_MODEL") else {
        eprintln!("Skipping test: BIRDNET_META_MODEL environment variable not set");
        return;
    };

    // Skip if meta model doesn't exist
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping test: meta model not found at {model_path}");
        return;
    }

    // Load classifier with actual model (if available)
    let classifier_model_path = format!("{FIXTURES_DIR}/birdnet_v24.onnx");
    if !std::path::Path::new(&classifier_model_path).exists() {
        eprintln!("Skipping test: classifier model not found");
        return;
    }

    let classifier = Classifier::builder()
        .model_path(&classifier_model_path)
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()
        .expect("failed to build classifier");

    // Build range filter using classifier labels
    let range_filter = RangeFilter::builder()
        .model_path(&model_path)
        .from_classifier_labels(classifier.labels())
        .threshold(0.01)
        .build()
        .expect("failed to build range filter");

    // Verify it works
    let scores = range_filter
        .predict(60.1695, 24.9354, 6, 15)
        .expect("prediction failed");

    assert!(!scores.is_empty());
}

#[test]
#[allow(clippy::expect_used, clippy::print_stdout)]
fn test_range_filter_complete_workflow() {
    #[cfg(feature = "load-dynamic")]
    init_runtime().expect("failed to init runtime");

    // Get model path from environment variable
    let Ok(model_path) = std::env::var("BIRDNET_META_MODEL") else {
        eprintln!("Skipping test: BIRDNET_META_MODEL environment variable not set");
        eprintln!(
            "To run this test locally: export BIRDNET_META_MODEL=/path/to/birdnet_data_model.onnx"
        );
        return;
    };

    // Skip if meta model doesn't exist
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: meta model not found at {model_path}");
        return;
    }

    // Load classifier with actual model (if available)
    let classifier_model_path = format!("{FIXTURES_DIR}/birdnet_v24.onnx");
    if !Path::new(&classifier_model_path).exists() {
        eprintln!("Skipping test: classifier model not found");
        return;
    }

    let classifier = Classifier::builder()
        .model_path(&classifier_model_path)
        .labels_path(format!("{FIXTURES_DIR}/birdnet_v24_labels.txt"))
        .build()
        .expect("failed to build classifier");

    // Build range filter from classifier labels
    let range_filter = RangeFilter::builder()
        .model_path(&model_path)
        .from_classifier_labels(classifier.labels())
        .threshold(0.01)
        .build()
        .expect("failed to build range filter");

    // Create mock predictions
    use birdnet_onnx::Prediction;
    let predictions = vec![
        Prediction {
            species: classifier.labels()[0].clone(),
            confidence: 0.8,
            index: 0,
        },
        Prediction {
            species: classifier.labels()[1].clone(),
            confidence: 0.6,
            index: 1,
        },
    ];

    // Get location scores (Helsinki, June 15)
    let location_scores = range_filter
        .predict(60.1695, 24.9354, 6, 15)
        .expect("prediction failed");

    // Filter predictions
    let filtered = range_filter.filter_predictions(&predictions, &location_scores, false);

    // Should have some results (exact count depends on location scores)
    println!(
        "Original: {}, Filtered: {}",
        predictions.len(),
        filtered.len()
    );

    // Test batch filtering
    let batch = vec![predictions.clone(), predictions];
    let filtered_batch = range_filter.filter_batch_predictions(
        batch,
        &location_scores,
        true, // with reranking
    );

    assert_eq!(filtered_batch.len(), 2);
}
