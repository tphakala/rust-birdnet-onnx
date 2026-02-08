//! BSG Finland post-processor for calibration and species distribution model (SDM) adjustment.
//!
//! The BSG (Bird Survey Group) Finland pipeline applies two post-processing steps
//! to raw model predictions:
//!
//! 1. **Per-species logistic calibration** — corrects confidence values using
//!    empirically fitted parameters so that scores match real-world detection accuracy.
//! 2. **Species Distribution Model (SDM)** — adjusts scores based on geographic location,
//!    seasonal migration curves, and distribution maps.
//!
//! # Usage
//!
//! ```ignore
//! use birdnet_onnx::{Classifier, ModelType, InferenceOptions, BsgPostProcessor};
//!
//! let classifier = Classifier::builder()
//!     .model_path("BSG_fused.onnx")
//!     .labels_path("BSG_labels.txt")
//!     .model_type(ModelType::BsgFinland)
//!     .build()?;
//!
//! let bsg = BsgPostProcessor::builder()
//!     .labels_path("BSG_labels.txt")
//!     .calibration_path("calibration.csv")
//!     .migration_path("migration.csv")
//!     .distribution_maps_path("distribution_maps.bin")
//!     .build()?;
//!
//! let result = classifier.predict(&segment, &InferenceOptions::default())?;
//!
//! // Calibration only
//! let calibrated = bsg.calibrate(&result)?;
//!
//! // Calibration + SDM
//! let adjusted = bsg.process(&result, 60.17, 24.94, 150)?;
//! ```

use crate::error::{Error, Result};
use crate::labels::load_labels_from_file;
use crate::rangefilter::validate_coordinates;
use crate::types::{ModelType, Prediction, PredictionResult};
use std::cmp::Ordering;

// ── Calibration ──────────────────────────────────────────────────────────────

/// Per-species logistic calibration parameters.
struct CalibrationParams {
    /// Per-species intercept (β₀).
    intercept: Vec<f32>,
    /// Per-species slope (β₁).
    slope: Vec<f32>,
}

impl CalibrationParams {
    fn load(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::BsgCalibrationLoad(format!("failed to read {path}: {e}")))?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(content.as_bytes());

        let mut intercept = Vec::new();
        let mut slope = Vec::new();

        for (i, record) in reader.records().enumerate() {
            let record = record.map_err(|e| {
                Error::BsgCalibrationLoad(format!("CSV parse error at row {i}: {e}"))
            })?;

            if record.len() < 2 {
                return Err(Error::BsgCalibrationLoad(format!(
                    "row {i}: expected 2 columns, got {}",
                    record.len()
                )));
            }

            let b0: f32 = record[0]
                .parse()
                .map_err(|e| Error::BsgCalibrationLoad(format!("row {i} intercept: {e}")))?;
            let b1: f32 = record[1]
                .parse()
                .map_err(|e| Error::BsgCalibrationLoad(format!("row {i} slope: {e}")))?;

            intercept.push(b0);
            slope.push(b1);
        }

        if intercept.is_empty() {
            return Err(Error::BsgCalibrationLoad(
                "calibration file is empty".to_string(),
            ));
        }

        Ok(Self { intercept, slope })
    }

    /// Apply per-species logistic calibration: `1 / (1 + exp(-(β₀ + β₁ * score)))`.
    /// For species with β₀=0 and β₁=0 (identity), returns the score unchanged.
    #[inline]
    fn calibrate(&self, index: usize, score: f32) -> f32 {
        if index >= self.intercept.len() {
            return score;
        }
        let b0 = self.intercept[index];
        let b1 = self.slope[index];
        if b0 == 0.0 && b1 == 0.0 {
            return score;
        }
        1.0 / (1.0 + (-b1.mul_add(score, b0)).exp())
    }
}

// ── Migration / Normal CDF ───────────────────────────────────────────────────

/// Per-species migration parameters for temporal filtering.
struct MigrationParams {
    arrival_intercept: Vec<f32>,
    arrival_slope: Vec<f32>,
    departure_intercept: Vec<f32>,
    departure_slope: Vec<f32>,
    arrival_std: Vec<f32>,
    departure_std: Vec<f32>,
    seasonal_start: Vec<f32>,
    seasonal_end: Vec<f32>,
}

impl MigrationParams {
    fn load(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            Error::BsgMapsLoad(format!("failed to read migration file {path}: {e}"))
        })?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(content.as_bytes());

        let mut params = Self {
            arrival_intercept: Vec::new(),
            arrival_slope: Vec::new(),
            departure_intercept: Vec::new(),
            departure_slope: Vec::new(),
            arrival_std: Vec::new(),
            departure_std: Vec::new(),
            seasonal_start: Vec::new(),
            seasonal_end: Vec::new(),
        };

        for (i, record) in reader.records().enumerate() {
            let record = record.map_err(|e| {
                Error::BsgMapsLoad(format!("migration CSV parse error at row {i}: {e}"))
            })?;

            if record.len() < 8 {
                return Err(Error::BsgMapsLoad(format!(
                    "migration row {i}: expected 8 columns, got {}",
                    record.len()
                )));
            }

            let parse = |col: usize| -> Result<f32> {
                record[col]
                    .parse()
                    .map_err(|e| Error::BsgMapsLoad(format!("migration row {i} col {col}: {e}")))
            };

            params.arrival_intercept.push(parse(0)?);
            params.arrival_slope.push(parse(1)?);
            params.departure_intercept.push(parse(2)?);
            params.departure_slope.push(parse(3)?);
            params.arrival_std.push(parse(4)?);
            params.departure_std.push(parse(5)?);
            params.seasonal_start.push(parse(6)?);
            params.seasonal_end.push(parse(7)?);
        }

        Ok(params)
    }
}

/// Normal CDF approximation (Abramowitz & Stegun, formula 7.1.26).
///
/// Maximum error < 1.5×10⁻⁷. Sufficient for migration curve calculations.
#[allow(clippy::excessive_precision)]
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let abs_x = x.abs();
    let t = 1.0 / 0.231_641_9f64.mul_add(abs_x, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    // Horner-like evaluation of the polynomial
    let poly = 1.330_274_429f64.mul_add(
        t5,
        (-1.821_255_978f64).mul_add(
            t4,
            1.781_477_937f64.mul_add(t3, (-0.356_563_782f64).mul_add(t2, 0.319_381_530 * t)),
        ),
    );

    // 1 / sqrt(2π) ≈ 0.398_942_280_401_432_7
    let frac_1_sqrt_2pi: f64 = (2.0 * std::f64::consts::PI).sqrt().recip();
    let pdf = (-0.5 * abs_x * abs_x).exp() * frac_1_sqrt_2pi;
    let cdf = 1.0 - pdf * poly;

    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

/// Normal CDF with mean and standard deviation: `Φ((x - mean) / std)`.
fn normal_cdf_with_params(x: f64, mean: f64, std: f64) -> f64 {
    if std <= 0.0 {
        return if x >= mean { 1.0 } else { 0.0 };
    }
    normal_cdf((x - mean) / std)
}

// ── Distribution Maps ────────────────────────────────────────────────────────

/// Magic bytes identifying a BSG distribution maps binary file.
const BSDM_MAGIC: &[u8; 4] = b"BSDM";

/// Packed distribution maps for geographic presence sampling.
struct DistributionMaps {
    width: usize,
    height: usize,
    origin_lon: f64,
    origin_lat: f64,
    pixel_scale_lon: f64,
    pixel_scale_lat: f64,
    /// Flat data: `[species_offset][map_a | map_b][height * width]`, all f32.
    data: Vec<f32>,
}

impl DistributionMaps {
    fn load(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::BsgMapsLoad(format!("failed to read {path}: {e}")))?;

        if bytes.len() < 64 {
            return Err(Error::BsgMapsLoad("file too small for header".to_string()));
        }

        // Parse header
        let magic = &bytes[0..4];
        if magic != BSDM_MAGIC {
            return Err(Error::BsgMapsLoad(format!(
                "invalid magic: expected BSDM, got {magic:?}"
            )));
        }

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 1 {
            return Err(Error::BsgMapsLoad(format!(
                "unsupported version: {version}"
            )));
        }

        let num_species = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let width = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let height = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;

        let origin_lon = f64::from_le_bytes(
            bytes[20..28]
                .try_into()
                .map_err(|_| Error::BsgMapsLoad("failed to parse origin_lon".to_string()))?,
        );
        let origin_lat = f64::from_le_bytes(
            bytes[28..36]
                .try_into()
                .map_err(|_| Error::BsgMapsLoad("failed to parse origin_lat".to_string()))?,
        );
        let pixel_scale_lon = f64::from_le_bytes(
            bytes[36..44]
                .try_into()
                .map_err(|_| Error::BsgMapsLoad("failed to parse pixel_scale_lon".to_string()))?,
        );
        let pixel_scale_lat = f64::from_le_bytes(
            bytes[44..52]
                .try_into()
                .map_err(|_| Error::BsgMapsLoad("failed to parse pixel_scale_lat".to_string()))?,
        );

        let map_pixels = width * height;
        let expected_data_size = num_species * 2 * map_pixels * 4; // 2 maps per species, f32
        let data_bytes = &bytes[64..];

        if data_bytes.len() != expected_data_size {
            return Err(Error::BsgMapsLoad(format!(
                "data size mismatch: expected {expected_data_size}, got {}",
                data_bytes.len()
            )));
        }

        // Convert bytes to f32 vec
        let data: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Self {
            width,
            height,
            origin_lon,
            origin_lat,
            pixel_scale_lon,
            pixel_scale_lat,
            data,
        })
    }

    /// Sample a map value at (lat, lon) for a given species and map type.
    /// `species_offset` is 0-based (species class index 2 → offset 0).
    /// `map_idx` is 0 for `map_a`, 1 for `map_b`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn sample(&self, species_offset: usize, map_idx: usize, lat: f64, lon: f64) -> f32 {
        let col_f = ((lon - self.origin_lon) / self.pixel_scale_lon).floor();
        let row_f = ((self.origin_lat - lat) / self.pixel_scale_lat).floor();

        if col_f < 0.0 || row_f < 0.0 || col_f >= self.width as f64 || row_f >= self.height as f64 {
            return 1.0; // Outside map bounds: assume species possible
        }

        let col = col_f as usize;
        let row = row_f as usize;
        let map_pixels = self.width * self.height;
        let offset = (species_offset * 2 + map_idx) * map_pixels + row * self.width + col;

        self.data.get(offset).copied().unwrap_or(1.0)
    }
}

// ── SDM Engine ───────────────────────────────────────────────────────────────

/// SDM data bundle (migration params + distribution maps).
struct SdmData {
    migration: MigrationParams,
    maps: DistributionMaps,
}

impl SdmData {
    /// Calculate SDM-adjusted score for a single species prediction.
    #[allow(clippy::cast_precision_loss)]
    fn adjust_score(
        &self,
        species_index: usize,
        score: f32,
        latitude: f32,
        longitude: f32,
        day_of_year: u32,
    ) -> f32 {
        // SDM only applies to bird species (indices 2+).
        // BSG model indices 0-1 are non-bird classes (background/noise) that lack
        // migration and distribution data.
        if species_index < 2 || species_index >= self.migration.arrival_intercept.len() {
            return score;
        }

        let lat = f64::from(latitude);
        let day = f64::from(day_of_year);

        // Clamp day_of_year to max 365
        let day = day.min(365.0);

        // 1. Migration probability
        let arrival_mean = f64::from(self.migration.arrival_slope[species_index]).mul_add(
            lat,
            f64::from(self.migration.arrival_intercept[species_index]),
        );
        let departure_mean = f64::from(self.migration.departure_slope[species_index]).mul_add(
            lat,
            f64::from(self.migration.departure_intercept[species_index]),
        );

        let arrival_std = f64::from(self.migration.arrival_std[species_index]) / 2.0;
        let departure_std = f64::from(self.migration.departure_std[species_index]) / 2.0;

        let arrival_prob = normal_cdf_with_params(day, arrival_mean, arrival_std);
        let departure_prob = 1.0 - normal_cdf_with_params(day, departure_mean, departure_std);
        let migration_prob = arrival_prob.min(departure_prob);

        // 2. Geographic presence (map_a)
        let species_offset = species_index - 2;
        let geo_prob = f64::from(
            self.maps
                .sample(species_offset, 0, lat, f64::from(longitude)),
        );

        // 3. Seasonal map (map_b)
        let seasonal_start = f64::from(self.migration.seasonal_start[species_index]);
        let seasonal_end = f64::from(self.migration.seasonal_end[species_index]);

        let use_map_b = if seasonal_start < seasonal_end {
            day >= seasonal_start && day <= seasonal_end
        } else if seasonal_start > seasonal_end {
            // Year wraps around (e.g., Nov-Feb)
            day >= seasonal_start || day <= seasonal_end
        } else {
            false
        };

        let time_prob = if use_map_b {
            f64::from(
                self.maps
                    .sample(species_offset, 1, lat, f64::from(longitude)),
            )
        } else {
            0.0
        };

        // 4. Combined presence probability
        let use_b = if use_map_b { 1.0 } else { 0.0 };
        let presence = migration_prob * ((1.0 - geo_prob) * use_b).mul_add(time_prob, geo_prob);

        // 5. Score adjustment (from BSG Python reference: functions.py)
        // The 0.25 scaling factor controls SDM influence on the final score.
        if presence <= 0.0 {
            return 0.0;
        }

        let adj = (presence.log10() + 1.0).clamp(-10.0, 0.0);
        let score_f64 = f64::from(score);
        let numerator = (score_f64 + adj * 0.25).max(0.0);
        let denominator = (1.0 + adj * 0.25).max(0.0001);

        #[allow(clippy::cast_possible_truncation)]
        let result = (numerator / denominator) as f32;
        result
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Builder for constructing a [`BsgPostProcessor`].
#[derive(Debug, Default)]
#[allow(clippy::struct_field_names)]
pub struct BsgPostProcessorBuilder {
    labels_path: Option<String>,
    calibration_path: Option<String>,
    migration_path: Option<String>,
    distribution_maps_path: Option<String>,
}

impl BsgPostProcessorBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the path to the BSG labels file (required).
    ///
    /// One label per line, matching the model's output order.
    /// Used to name species in post-processed predictions.
    #[must_use]
    pub fn labels_path(mut self, path: impl Into<String>) -> Self {
        self.labels_path = Some(path.into());
        self
    }

    /// Set the path to the calibration parameters CSV file (required).
    ///
    /// CSV format: `intercept,slope` header, one row per species.
    #[must_use]
    pub fn calibration_path(mut self, path: impl Into<String>) -> Self {
        self.calibration_path = Some(path.into());
        self
    }

    /// Set the path to the migration parameters CSV file (optional, needed for SDM).
    ///
    /// CSV format: 8-column header with per-species migration curve parameters.
    #[must_use]
    pub fn migration_path(mut self, path: impl Into<String>) -> Self {
        self.migration_path = Some(path.into());
        self
    }

    /// Set the path to the packed distribution maps binary file (optional, needed for SDM).
    ///
    /// Binary format with BSDM header + packed float32 map data.
    #[must_use]
    pub fn distribution_maps_path(mut self, path: impl Into<String>) -> Self {
        self.distribution_maps_path = Some(path.into());
        self
    }

    /// Build the post-processor.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Labels path was not set
    /// - Calibration path was not set
    /// - Calibration file cannot be loaded or parsed
    /// - Migration or distribution map files cannot be loaded (if provided)
    /// - Only one of migration/maps path provided (both needed for SDM)
    pub fn build(self) -> Result<BsgPostProcessor> {
        let labels_path = self
            .labels_path
            .ok_or_else(|| Error::BsgCalibrationLoad("labels path required".to_string()))?;

        let labels = load_labels_from_file(&labels_path, ModelType::BsgFinland)?;

        let calibration_path = self
            .calibration_path
            .ok_or_else(|| Error::BsgCalibrationLoad("calibration path required".to_string()))?;

        let calibration = CalibrationParams::load(&calibration_path)?;

        let sdm = match (self.migration_path, self.distribution_maps_path) {
            (Some(mig_path), Some(maps_path)) => {
                let migration = MigrationParams::load(&mig_path)?;
                let maps = DistributionMaps::load(&maps_path)?;
                Some(SdmData { migration, maps })
            }
            (None, None) => None,
            (Some(_), None) => {
                return Err(Error::BsgMapsLoad(
                    "migration_path set but distribution_maps_path missing (both required for SDM)"
                        .to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(Error::BsgMapsLoad(
                    "distribution_maps_path set but migration_path missing (both required for SDM)"
                        .to_string(),
                ));
            }
        };

        Ok(BsgPostProcessor {
            labels,
            calibration,
            sdm,
        })
    }
}

/// BSG Finland post-processor for calibration and SDM adjustment.
///
/// Use [`BsgPostProcessor::builder()`] to construct.
pub struct BsgPostProcessor {
    labels: Vec<String>,
    calibration: CalibrationParams,
    sdm: Option<SdmData>,
}

impl std::fmt::Debug for BsgPostProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BsgPostProcessor")
            .field("num_labels", &self.labels.len())
            .field("calibration_species", &self.calibration.intercept.len())
            .field("sdm_enabled", &self.sdm.is_some())
            .finish_non_exhaustive()
    }
}

impl BsgPostProcessor {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> BsgPostProcessorBuilder {
        BsgPostProcessorBuilder::new()
    }

    /// Apply per-species logistic calibration to predictions.
    ///
    /// Rebuilds all predictions from `raw_scores` with calibrated confidence values,
    /// filtered to `confidence > 0.0`. The classifier's `top_k` and `min_confidence`
    /// are intentionally bypassed because calibration changes the score ranking —
    /// callers should apply their own filtering to the returned result.
    ///
    /// # Errors
    ///
    /// Returns an error if the prediction result cannot be processed.
    pub fn calibrate(&self, result: &PredictionResult) -> Result<PredictionResult> {
        let mut predictions: Vec<Prediction> = result
            .raw_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| {
                let confidence = self.calibration.calibrate(i, score);
                let species = self
                    .labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("unknown_{i}"));
                Prediction {
                    species,
                    confidence,
                    index: i,
                }
            })
            .filter(|p| p.confidence > 0.0)
            .collect();

        predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
        });

        Ok(PredictionResult {
            model_type: result.model_type,
            predictions,
            embeddings: result.embeddings.clone(),
            raw_scores: result.raw_scores.clone(),
        })
    }

    /// Apply calibration + SDM adjustment to predictions.
    ///
    /// Requires migration and distribution map data (set at build time).
    ///
    /// # Arguments
    /// * `result` - Prediction result from classifier
    /// * `latitude` - Latitude in degrees (-90 to 90)
    /// * `longitude` - Longitude in degrees (-180 to 180)
    /// * `day_of_year` - Day of year (1-366)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - SDM data was not loaded at build time
    /// - Coordinates are invalid
    /// - Day of year is invalid
    pub fn process(
        &self,
        result: &PredictionResult,
        latitude: f32,
        longitude: f32,
        day_of_year: u32,
    ) -> Result<PredictionResult> {
        validate_coordinates(latitude, longitude)?;

        if !(1..=366).contains(&day_of_year) {
            return Err(Error::InvalidDayOfYear { day_of_year });
        }

        let sdm = self.sdm.as_ref().ok_or_else(|| {
            Error::BsgProcessing(
                "SDM data not loaded: set migration_path and distribution_maps_path in builder"
                    .to_string(),
            )
        })?;

        // First calibrate, then apply SDM
        let calibrated_scores: Vec<f32> = result
            .raw_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| self.calibration.calibrate(i, score))
            .collect();

        let adjusted_scores: Vec<f32> = calibrated_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| sdm.adjust_score(i, score, latitude, longitude, day_of_year))
            .collect();

        // Build predictions from adjusted scores using BSG labels
        let mut predictions: Vec<Prediction> = adjusted_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| {
                let species = self
                    .labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("unknown_{i}"));
                Prediction {
                    species,
                    confidence: score,
                    index: i,
                }
            })
            .filter(|p| p.confidence > 0.0)
            .collect();

        predictions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(Ordering::Equal)
        });

        Ok(PredictionResult {
            model_type: result.model_type,
            predictions,
            embeddings: result.embeddings.clone(),
            raw_scores: result.raw_scores.clone(),
        })
    }

    /// Apply calibration to a batch of prediction results.
    ///
    /// # Errors
    ///
    /// Returns an error if any result cannot be processed.
    pub fn calibrate_batch(&self, results: &[PredictionResult]) -> Result<Vec<PredictionResult>> {
        results.iter().map(|r| self.calibrate(r)).collect()
    }

    /// Apply calibration + SDM to a batch of prediction results.
    ///
    /// All results are assumed to be from the same location and date.
    ///
    /// # Errors
    ///
    /// Returns an error if SDM data is not loaded, coordinates are invalid,
    /// or day of year is invalid.
    pub fn process_batch(
        &self,
        results: &[PredictionResult],
        latitude: f32,
        longitude: f32,
        day_of_year: u32,
    ) -> Result<Vec<PredictionResult>> {
        results
            .iter()
            .map(|r| self.process(r, latitude, longitude, day_of_year))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::disallowed_methods)]
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn test_normal_cdf_standard() {
        // Φ(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // Φ(1) ≈ 0.8413
        assert!((normal_cdf(1.0) - 0.841_344_746_068_543).abs() < 1e-6);

        // Φ(-1) ≈ 0.1587
        assert!((normal_cdf(-1.0) - 0.158_655_253_931_457).abs() < 1e-6);

        // Φ(2) ≈ 0.9772
        assert!((normal_cdf(2.0) - 0.977_249_868_051_821).abs() < 1e-5);
    }

    #[test]
    fn test_normal_cdf_extremes() {
        assert!(normal_cdf(-10.0) < 1e-6);
        assert!((normal_cdf(10.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_with_params_shifted() {
        // CDF at the mean should be 0.5
        assert!((normal_cdf_with_params(100.0, 100.0, 10.0) - 0.5).abs() < 1e-6);

        // 1 std above mean ≈ 0.8413
        assert!((normal_cdf_with_params(110.0, 100.0, 10.0) - 0.841_344_746_068_543).abs() < 1e-5);
    }

    #[test]
    fn test_calibration_identity() {
        let cal = CalibrationParams {
            intercept: vec![0.0],
            slope: vec![0.0],
        };

        // With β₀=0 and β₁=0, score should be returned unchanged
        assert_eq!(cal.calibrate(0, 0.5), 0.5);
        assert_eq!(cal.calibrate(0, 0.9), 0.9);
    }

    #[test]
    fn test_calibration_transform() {
        let cal = CalibrationParams {
            intercept: vec![-8.055],
            slope: vec![10.845],
        };

        // For these params, raw 0.9 should give a specific calibrated value
        let result = cal.calibrate(0, 0.9);
        // sigmoid(-8.055 + 10.845 * 0.9) = sigmoid(1.7055) ≈ 0.846
        assert!((result - 0.846).abs() < 0.01);
    }

    #[test]
    fn test_calibration_out_of_bounds() {
        let cal = CalibrationParams {
            intercept: vec![1.0],
            slope: vec![2.0],
        };

        // Index out of bounds returns score unchanged
        assert_eq!(cal.calibrate(5, 0.7), 0.7);
    }

    #[test]
    fn test_builder_missing_labels() {
        let result = BsgPostProcessorBuilder::new()
            .calibration_path("dummy.csv")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_calibration() {
        let dir = std::env::temp_dir().join("bsg_test_missing_cal");
        std::fs::create_dir_all(&dir).unwrap();

        let labels_path = dir.join("labels.txt");
        std::fs::write(&labels_path, "species_a\nspecies_b\n").unwrap();

        let result = BsgPostProcessorBuilder::new()
            .labels_path(labels_path.to_str().unwrap())
            .build();
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_builder_partial_sdm_migration_only() {
        // Create temp files
        let dir = std::env::temp_dir().join("bsg_test_partial_mig");
        std::fs::create_dir_all(&dir).unwrap();

        let labels_path = dir.join("labels.txt");
        std::fs::write(&labels_path, "species_a\n").unwrap();

        let cal_path = dir.join("cal.csv");
        std::fs::write(&cal_path, "intercept,slope\n0.0,0.0\n").unwrap();

        let mig_path = dir.join("mig.csv");
        std::fs::write(
            &mig_path,
            "a,b,c,d,e,f,g,h\n0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0\n",
        )
        .unwrap();

        let result = BsgPostProcessorBuilder::new()
            .labels_path(labels_path.to_str().unwrap())
            .calibration_path(cal_path.to_str().unwrap())
            .migration_path(mig_path.to_str().unwrap())
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("distribution_maps_path missing"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_builder_calibration_only() {
        let dir = std::env::temp_dir().join("bsg_test_cal_only");
        std::fs::create_dir_all(&dir).unwrap();

        let labels_path = dir.join("labels.txt");
        std::fs::write(&labels_path, "species_a\nspecies_b\n").unwrap();

        let cal_path = dir.join("cal.csv");
        std::fs::write(&cal_path, "intercept,slope\n0.0,0.0\n-8.0,10.0\n").unwrap();

        let processor = BsgPostProcessorBuilder::new()
            .labels_path(labels_path.to_str().unwrap())
            .calibration_path(cal_path.to_str().unwrap())
            .build()
            .unwrap();

        assert!(processor.sdm.is_none());
        assert_eq!(processor.labels.len(), 2);
        assert_eq!(processor.calibration.intercept.len(), 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
