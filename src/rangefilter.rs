//! Range filter for location and date-based species filtering

use crate::error::{Error, Result};
use crate::types::LocationScore;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::{Arc, Mutex};

/// Calculate week number for BirdNET meta model (48-week year, 4 weeks per month).
///
/// BirdNET assumes each month has exactly 4 weeks, creating a 48-week year.
/// Week calculation: weeksFromMonths = (month - 1) * 4; weekInMonth = (day - 1) / 7 + 1
///
/// # Arguments
/// * `month` - Month number (1-12)
/// * `day` - Day of month (1-31)
///
/// # Returns
/// Week number as f32 (typically 1-48, but can exceed 48 for days 29-31)
#[must_use]
pub fn calculate_week(month: u32, day: u32) -> f32 {
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

/// Builder for constructing a RangeFilter
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

        for provider in self.execution_providers {
            session_builder = session_builder
                .with_execution_providers([provider])
                .map_err(Error::ModelLoad)?;
        }

        let session = session_builder
            .commit_from_file(&model_path)
            .map_err(Error::ModelLoad)?;

        Ok(RangeFilter {
            inner: Arc::new(RangeFilterInner {
                session: Mutex::new(session),
                labels,
                threshold: self.threshold,
            }),
        })
    }
}

/// Internal state for RangeFilter
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
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
