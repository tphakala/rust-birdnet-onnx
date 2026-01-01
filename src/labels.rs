//! Label loading from various file formats.

use crate::error::{Error, Result};
use crate::types::{LabelFormat, ModelType};
use std::path::Path;

/// Helper struct for JSON object with "labels" key.
#[derive(serde::Deserialize)]
struct LabelsObject {
    labels: Vec<String>,
}

/// Helper struct for JSON array of objects.
#[derive(serde::Deserialize)]
struct LabelEntry {
    name: Option<String>,
    label: Option<String>,
    species: Option<String>,
}

/// Load labels from file using format expected by model type.
pub fn load_labels_from_file(path: impl AsRef<Path>, model_type: ModelType) -> Result<Vec<String>> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| Error::LabelLoad {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;

    parse_labels(&content, model_type.expected_label_format())
}

/// Parse labels from content string according to format.
pub fn parse_labels(content: &str, format: LabelFormat) -> Result<Vec<String>> {
    match format {
        LabelFormat::Text => Ok(parse_text_labels(content)),
        LabelFormat::Csv => parse_csv_labels(content),
        LabelFormat::Json => parse_json_labels(content),
    }
}

/// Parse text format: one label per line.
fn parse_text_labels(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// Parse CSV format: first column is label, skip header if detected.
fn parse_csv_labels(content: &str) -> Result<Vec<String>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(content.as_bytes());

    let mut labels = Vec::new();
    let mut first_row = true;

    for result in reader.records() {
        let record = result.map_err(|e| Error::LabelParse(e.to_string()))?;

        if let Some(first_col) = record.get(0) {
            let label = first_col.trim().to_string();

            // Skip header row if it looks like a header
            if first_row && looks_like_header(&label) {
                first_row = false;
                continue;
            }
            first_row = false;

            if !label.is_empty() {
                labels.push(label);
            }
        }
    }

    Ok(labels)
}

/// Check if a value looks like a CSV header.
fn looks_like_header(value: &str) -> bool {
    let lower = value.to_lowercase();
    lower == "label"
        || lower == "species"
        || lower == "name"
        || lower == "class"
        || lower == "common_name"
        || lower == "scientific_name"
}

/// Parse JSON format: supports multiple structures.
fn parse_json_labels(content: &str) -> Result<Vec<String>> {
    // Try parsing as array of strings: ["label1", "label2"]
    if let Ok(labels) = serde_json::from_str::<Vec<String>>(content) {
        return Ok(labels);
    }

    // Try parsing as object with "labels" key: {"labels": ["label1", "label2"]}
    if let Ok(obj) = serde_json::from_str::<LabelsObject>(content) {
        return Ok(obj.labels);
    }

    // Try parsing as array of objects: [{"name": "label1"}, {"name": "label2"}]
    if let Ok(entries) = serde_json::from_str::<Vec<LabelEntry>>(content) {
        let labels: Vec<String> = entries
            .into_iter()
            .filter_map(|e| e.name.or(e.label).or(e.species))
            .collect();
        if !labels.is_empty() {
            return Ok(labels);
        }
    }

    Err(Error::LabelParse(
        "unrecognized JSON format: expected array of strings, {labels: [...]}, or [{name: ...}]"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn test_parse_text_labels() {
        let content = "American Robin\nNorthern Cardinal\n\nBlue Jay\n";
        let labels = parse_text_labels(content);
        assert_eq!(
            labels,
            vec!["American Robin", "Northern Cardinal", "Blue Jay"]
        );
    }

    #[test]
    fn test_parse_text_labels_with_whitespace() {
        let content = "  American Robin  \n  Northern Cardinal  ";
        let labels = parse_text_labels(content);
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_csv_labels_simple() {
        let content = "American Robin\nNorthern Cardinal\nBlue Jay";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(
            labels,
            vec!["American Robin", "Northern Cardinal", "Blue Jay"]
        );
    }

    #[test]
    fn test_parse_csv_labels_with_header() {
        let content = "label,scientific_name\nAmerican Robin,Turdus migratorius\nNorthern Cardinal,Cardinalis cardinalis";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_csv_labels_species_header() {
        let content = "species\nAmerican Robin\nNorthern Cardinal";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_json_array() {
        let content = r#"["American Robin", "Northern Cardinal", "Blue Jay"]"#;
        let labels = parse_json_labels(content).unwrap();
        assert_eq!(
            labels,
            vec!["American Robin", "Northern Cardinal", "Blue Jay"]
        );
    }

    #[test]
    fn test_parse_json_object_with_labels() {
        let content = r#"{"labels": ["American Robin", "Northern Cardinal"]}"#;
        let labels = parse_json_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_json_array_of_objects() {
        let content = r#"[{"name": "American Robin"}, {"name": "Northern Cardinal"}]"#;
        let labels = parse_json_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_json_array_of_objects_label_key() {
        let content = r#"[{"label": "American Robin"}, {"label": "Northern Cardinal"}]"#;
        let labels = parse_json_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_json_invalid() {
        let content = r#"{"invalid": "format"}"#;
        let result = parse_json_labels(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_labels_by_format() {
        let text_content = "American Robin\nNorthern Cardinal";
        let labels = parse_labels(text_content, LabelFormat::Text).unwrap();
        assert_eq!(labels.len(), 2);

        let json_content = r#"["American Robin", "Northern Cardinal"]"#;
        let labels = parse_labels(json_content, LabelFormat::Json).unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_load_labels_file_not_found() {
        let result = load_labels_from_file("/nonexistent/path.txt", ModelType::BirdNetV24);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("failed to load labels"));
    }
}
