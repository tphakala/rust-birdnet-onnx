//! Range filter for location and date-based species filtering

use crate::error::{Error, Result};
use crate::types::LocationScore;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::{Arc, Mutex};

/// Calculate week number for `BirdNET` meta model (48-week year, 4 weeks per month).
///
/// `BirdNET` assumes each month has exactly 4 weeks, creating a 48-week year.
/// Week calculation: weeksFromMonths = (month - 1) * 4; weekInMonth = (day - 1) / 7 + 1
///
/// # Arguments
/// * `month` - Month number (1-12)
/// * `day` - Day of month (1-31)
///
/// # Returns
/// Week number as f32 (typically 1-48, but can exceed 48 for days 29-31)
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub const fn calculate_week(month: u32, day: u32) -> f32 {
    let weeks_from_months = (month - 1) * 4;
    let week_in_month = (day - 1) / 7 + 1;
    (weeks_from_months + week_in_month) as f32
}

/// Validate geographic coordinates.
///
/// # Arguments
/// * `latitude` - Latitude in degrees (-90 to 90)
/// * `longitude` - Longitude in degrees (-180 to 180)
///
/// # Errors
/// Returns `Error::InvalidCoordinates` if values are out of range
pub fn validate_coordinates(latitude: f32, longitude: f32) -> Result<()> {
    if !(-90.0..=90.0).contains(&latitude) {
        return Err(Error::InvalidCoordinates {
            latitude,
            longitude,
            reason: format!("latitude must be in range [-90, 90], got {latitude}"),
        });
    }
    if !(-180.0..=180.0).contains(&longitude) {
        return Err(Error::InvalidCoordinates {
            latitude,
            longitude,
            reason: format!("longitude must be in range [-180, 180], got {longitude}"),
        });
    }
    Ok(())
}

/// Validate date parameters for `BirdNET` calendar.
///
/// # Arguments
/// * `month` - Month number (1-12)
/// * `day` - Day of month (1-31)
///
/// # Errors
/// Returns `Error::RangeFilterInference` if values are out of range
pub fn validate_date(month: u32, day: u32) -> Result<()> {
    if !(1..=12).contains(&month) {
        return Err(Error::RangeFilterInference(format!(
            "month must be in range [1, 12], got {month}"
        )));
    }
    if !(1..=31).contains(&day) {
        return Err(Error::RangeFilterInference(format!(
            "day must be in range [1, 31], got {day}"
        )));
    }
    Ok(())
}

/// Builder for constructing a `RangeFilter`
#[derive(Debug)]
pub struct RangeFilterBuilder {
    model_path: Option<String>,
    labels: Option<Vec<String>>,
    execution_providers: Vec<ort::execution_providers::ExecutionProviderDispatch>,
    threshold: f32,
}

impl Default for RangeFilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeFilterBuilder {
    /// Create a new range filter builder
    #[must_use]
    pub const fn new() -> Self {
        Self {
            model_path: None,
            labels: None,
            execution_providers: Vec::new(),
            threshold: 0.01,
        }
    }

    /// Set the path to the ONNX meta model file (required)
    #[must_use]
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set species labels (required, must match model output size)
    #[must_use]
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add an execution provider (GPU, CPU, etc.)
    #[must_use]
    pub fn execution_provider(
        mut self,
        provider: impl Into<ort::execution_providers::ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers.push(provider.into());
        self
    }

    /// Set minimum score threshold (default: 0.01)
    #[must_use]
    pub const fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Build the range filter
    ///
    /// # Errors
    /// Returns error if model path or labels not set, or if model loading fails
    pub fn build(self) -> Result<RangeFilter> {
        let model_path = self.model_path.ok_or(Error::ModelPathRequired)?;
        let labels = self.labels.ok_or(Error::LabelsRequired)?;

        // Build session with execution providers
        let mut session_builder = Session::builder().map_err(Error::ModelLoad)?;

        if !self.execution_providers.is_empty() {
            session_builder = session_builder
                .with_execution_providers(self.execution_providers)
                .map_err(Error::ModelLoad)?;
        }

        let session = session_builder
            .commit_from_file(&model_path)
            .map_err(Error::ModelLoad)?;

        // Validate label count matches model output
        let output_shapes = extract_output_shapes(&session)?;

        // Meta model should have exactly one output
        if output_shapes.len() != 1 {
            return Err(Error::ModelDetection {
                reason: format!("meta model expects 1 output, got {}", output_shapes.len()),
            });
        }

        let expected = extract_last_dim(&output_shapes[0])?;
        if labels.len() != expected {
            return Err(Error::LabelCount {
                expected,
                got: labels.len(),
            });
        }

        Ok(RangeFilter {
            inner: Arc::new(RangeFilterInner {
                session: Mutex::new(session),
                labels,
                threshold: self.threshold,
            }),
        })
    }
}

/// Extract output tensor shapes from session
fn extract_output_shapes(session: &Session) -> Result<Vec<Vec<i64>>> {
    session
        .outputs
        .iter()
        .map(|output| {
            let shape = output
                .output_type
                .tensor_shape()
                .ok_or_else(|| Error::ModelDetection {
                    reason: "output is not a tensor".to_string(),
                })?;
            Ok(shape.iter().copied().collect())
        })
        .collect()
}

/// Extract last dimension from output shape
fn extract_last_dim(shape: &[i64]) -> Result<usize> {
    let value = shape.last().copied().ok_or_else(|| Error::ModelDetection {
        reason: "empty output shape".to_string(),
    })?;

    usize::try_from(value).map_err(|_| Error::ModelDetection {
        reason: format!("invalid dimension: {value}"),
    })
}

/// Internal state for `RangeFilter`
struct RangeFilterInner {
    session: Mutex<Session>,
    labels: Vec<String>,
    threshold: f32,
}

/// Thread-safe range filter for location-based species filtering
#[derive(Clone)]
pub struct RangeFilter {
    inner: Arc<RangeFilterInner>,
}

impl std::fmt::Debug for RangeFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RangeFilter")
            .field("labels_count", &self.inner.labels.len())
            .field("threshold", &self.inner.threshold)
            .finish_non_exhaustive()
    }
}

impl RangeFilter {
    /// Create a new range filter builder
    #[must_use]
    pub const fn builder() -> RangeFilterBuilder {
        RangeFilterBuilder::new()
    }

    /// Get species probability scores for given location and date
    ///
    /// # Arguments
    /// * `latitude` - Latitude in degrees (-90 to 90)
    /// * `longitude` - Longitude in degrees (-180 to 180)
    /// * `month` - Month number (1-12)
    /// * `day` - Day of month (1-31)
    ///
    /// # Returns
    /// Vector of `LocationScore` sorted by score (descending)
    ///
    /// # Errors
    /// Returns error if:
    /// - Coordinates are invalid (latitude not in [-90, 90] or longitude not in [-180, 180])
    /// - Date parameters are invalid (month not in [1, 12] or day not in [1, 31])
    /// - Session lock is poisoned
    /// - ONNX inference fails
    #[allow(clippy::significant_drop_tightening)]
    pub fn predict(
        &self,
        latitude: f32,
        longitude: f32,
        month: u32,
        day: u32,
    ) -> Result<Vec<LocationScore>> {
        // Validate coordinates
        validate_coordinates(latitude, longitude)?;

        // Validate date parameters
        validate_date(month, day)?;

        // Calculate week number
        let week = calculate_week(month, day);

        // Create input tensor [1, 3] with [latitude, longitude, week]
        let input_data = vec![latitude, longitude, week];
        let input_array = Array2::from_shape_vec((1, 3), input_data).map_err(|e| {
            Error::RangeFilterInference(format!("failed to create input array: {e}"))
        })?;

        let input_value = Value::from_array(input_array).map_err(|e| {
            Error::RangeFilterInference(format!("failed to create input tensor: {e}"))
        })?;

        // Run inference with locked session
        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|e| Error::RangeFilterInference(format!("session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| Error::RangeFilterInference(e.to_string()))?;

        // Extract output tensor
        let output_names: Vec<_> = outputs.keys().collect();
        let name = output_names
            .first()
            .ok_or_else(|| Error::RangeFilterInference("missing output tensor".to_string()))?;

        let tensor = outputs.get(*name).ok_or_else(|| {
            Error::RangeFilterInference(format!("missing output tensor '{name}'"))
        })?;

        let (_, data) = tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::RangeFilterInference(e.to_string()))?;

        // Build scores above threshold
        let mut scores: Vec<LocationScore> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &score)| {
                if score >= self.inner.threshold && i < self.inner.labels.len() {
                    Some(LocationScore {
                        species: self.inner.labels[i].clone(),
                        score,
                        index: i,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn test_calculate_week_january_first() {
        // January 1st = month 1, day 1
        // weeksFromMonths = (1 - 1) * 4 = 0
        // weekInMonth = (1 - 1) / 7 + 1 = 1
        // week = 0 + 1 = 1
        let week = calculate_week(1, 1);
        assert_eq!(week, 1.0);
    }

    #[test]
    fn test_calculate_week_january_eighth() {
        // January 8th = month 1, day 8
        // weeksFromMonths = 0
        // weekInMonth = (8 - 1) / 7 + 1 = 2
        // week = 0 + 2 = 2
        let week = calculate_week(1, 8);
        assert_eq!(week, 2.0);
    }

    #[test]
    fn test_calculate_week_february_first() {
        // February 1st = month 2, day 1
        // weeksFromMonths = (2 - 1) * 4 = 4
        // weekInMonth = (1 - 1) / 7 + 1 = 1
        // week = 4 + 1 = 5
        let week = calculate_week(2, 1);
        assert_eq!(week, 5.0);
    }

    #[test]
    fn test_calculate_week_december_last() {
        // December 31st = month 12, day 31
        // weeksFromMonths = (12 - 1) * 4 = 44
        // weekInMonth = (31 - 1) / 7 + 1 = 5
        // week = 44 + 5 = 49
        // Note: BirdNET uses 48-week year, so this wraps
        let week = calculate_week(12, 31);
        assert_eq!(week, 49.0);
    }

    #[test]
    fn test_validate_coordinates_valid() {
        assert!(validate_coordinates(45.0, -122.0).is_ok());
        assert!(validate_coordinates(0.0, 0.0).is_ok());
        assert!(validate_coordinates(-90.0, -180.0).is_ok());
        assert!(validate_coordinates(90.0, 180.0).is_ok());
    }

    #[test]
    fn test_validate_coordinates_invalid_latitude() {
        let result = validate_coordinates(91.0, 0.0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::InvalidCoordinates { .. }
        ));
    }

    #[test]
    fn test_validate_coordinates_invalid_longitude() {
        let result = validate_coordinates(0.0, 181.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_date_valid() {
        assert!(validate_date(1, 1).is_ok());
        assert!(validate_date(6, 15).is_ok());
        assert!(validate_date(12, 31).is_ok());
    }

    #[test]
    fn test_validate_date_invalid_month_zero() {
        let result = validate_date(0, 1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("month must be in range")
        );
    }

    #[test]
    fn test_validate_date_invalid_month_thirteen() {
        let result = validate_date(13, 1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("month must be in range")
        );
    }

    #[test]
    fn test_validate_date_invalid_day_zero() {
        let result = validate_date(1, 0);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("day must be in range")
        );
    }

    #[test]
    fn test_validate_date_invalid_day_thirty_two() {
        let result = validate_date(1, 32);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("day must be in range")
        );
    }

    #[test]
    fn test_range_filter_builder_missing_model_path() {
        let result = RangeFilter::builder().build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ModelPathRequired));
    }

    #[test]
    fn test_range_filter_builder_missing_labels() {
        let result = RangeFilter::builder().model_path("/tmp/model.onnx").build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::LabelsRequired));
    }
}
