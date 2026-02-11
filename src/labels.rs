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

/// Parse CSV format: auto-detect delimiter, intelligently select label column.
fn parse_csv_labels(content: &str) -> Result<Vec<String>> {
    let delimiter = detect_csv_delimiter(content);
    parse_csv_with_delimiter(content, delimiter)
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
        || lower == "sci_name" // `BirdNET` v3
        || lower == "com_name" // `BirdNET` v3
        || lower == "idx" // `BirdNET` v3 index column
        || lower == "id"
        || lower.starts_with("inat") // `Perch` v2 dataset identifier (e.g., "inat2024_fsd50k")
        || lower.ends_with("_fsd50k") // `Perch` v2 dataset identifier
}

/// Detect the delimiter used in CSV content (comma or semicolon).
fn detect_csv_delimiter(content: &str) -> u8 {
    let first_line = content.lines().next().unwrap_or("");
    let comma_count = first_line.matches(',').count();
    let semicolon_count = first_line.matches(';').count();

    if semicolon_count > comma_count {
        b';'
    } else {
        b','
    }
}

/// Find the best column index for labels based on header row.
fn find_label_column(header: &csv::StringRecord) -> usize {
    let priority_headers = [
        "sci_name",        // `BirdNET` v3 scientific name
        "com_name",        // `BirdNET` v3 common name
        "scientific_name", // Common variant
        "common_name",     // Common variant
        "species",         // Generic species
        "name",            // Generic name
        "label",           // Generic label
    ];

    // Check each priority in order to ensure highest priority is selected
    for priority in &priority_headers {
        if let Some(index) = header
            .iter()
            .position(|field| field.trim().to_lowercase() == *priority)
        {
            return index;
        }
    }

    0 // Default to first column if no recognized header
}

/// Check if a record's first column appears to be a numeric index.
fn is_numeric_index(record: &csv::StringRecord) -> bool {
    record.get(0).is_some_and(|first_col| {
        let trimmed = first_col.trim();
        !trimmed.is_empty() && trimmed.chars().all(|c| c.is_ascii_digit())
    })
}

/// Find the first column that doesn't appear to be numeric.
fn find_first_non_numeric_column(record: &csv::StringRecord) -> usize {
    for (index, field) in record.iter().enumerate() {
        let trimmed = field.trim();
        if !trimmed.is_empty() && !trimmed.chars().all(|c| c.is_ascii_digit()) {
            return index;
        }
    }
    0 // Fallback to first column
}

/// Parse CSV content with specified delimiter and intelligent column selection.
fn parse_csv_with_delimiter(content: &str, delimiter: u8) -> Result<Vec<String>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .flexible(true)
        .from_reader(content.as_bytes());

    let mut labels = Vec::new();
    let mut first_row = true;
    let mut label_column_index = 0;

    for result in reader.records() {
        let record = result.map_err(|e| Error::LabelParse(e.to_string()))?;

        if first_row {
            // Check if first row is a header
            if let Some(first_col) = record.get(0) {
                let first_col_trimmed = first_col.trim();
                if looks_like_header(first_col_trimmed) {
                    label_column_index = find_label_column(&record);
                    first_row = false;
                    continue; // Skip header row
                }
            }

            // Not a header - check if first column is numeric index
            if is_numeric_index(&record) {
                label_column_index = find_first_non_numeric_column(&record);
            }

            first_row = false;
        }

        // Extract label from selected column
        if let Some(label_text) = record.get(label_column_index) {
            let label = label_text.trim().to_string();
            if !label.is_empty() {
                labels.push(label);
            }
        }
    }

    Ok(labels)
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
        // scientific_name has higher priority than label, so should be selected
        assert_eq!(labels, vec!["Turdus migratorius", "Cardinalis cardinalis"]);
    }

    #[test]
    fn test_parse_csv_labels_species_header() {
        let content = "species\nAmerican Robin\nNorthern Cardinal";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_csv_labels_perch_v2_inat_header() {
        let content = "inat2024_fsd50k\nAmerican Robin\nNorthern Cardinal";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_csv_labels_perch_v2_fsd50k_header() {
        let content = "dataset_fsd50k\nAmerican Robin\nNorthern Cardinal";
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

    // Edge case tests

    #[test]
    fn test_parse_text_labels_empty_lines() {
        let content = "Species 1\n\nSpecies 2\n\n\nSpecies 3";
        let labels = parse_text_labels(content);
        // Empty lines should be skipped
        assert_eq!(labels.len(), 3);
        assert_eq!(labels, vec!["Species 1", "Species 2", "Species 3"]);
    }

    #[test]
    fn test_parse_text_labels_with_unicode() {
        let content = "Pingüino Emperador\n鸟类\nПтица\n🐦";
        let labels = parse_text_labels(content);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], "Pingüino Emperador");
        assert_eq!(labels[1], "鸟类");
        assert_eq!(labels[2], "Птица");
        assert_eq!(labels[3], "🐦");
    }

    #[test]
    fn test_parse_text_labels_with_special_chars() {
        let content = "Species (Common)\nSpecies-rare\nSpecies_variant\nSpecies's";
        let labels = parse_text_labels(content);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], "Species (Common)");
        assert_eq!(labels[1], "Species-rare");
        assert_eq!(labels[2], "Species_variant");
        assert_eq!(labels[3], "Species's");
    }

    #[test]
    fn test_parse_csv_labels_inconsistent_columns() {
        let content = "label,scientific\nSpecies 1,Name1,Extra\nSpecies 2,Name2";
        // CSV parsing should handle rows with different column counts
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels, vec!["Species 1", "Species 2"]);
    }

    #[test]
    fn test_parse_csv_labels_empty_values() {
        let content = "label\n\nSpecies 1\n\nSpecies 2";
        let labels = parse_csv_labels(content).unwrap();
        // Empty rows are filtered out by the parser (line 73: if !label.is_empty())
        assert_eq!(labels, vec!["Species 1", "Species 2"]);
    }

    #[test]
    fn test_parse_json_array_empty() {
        let content = "[]";
        let labels = parse_json_labels(content).unwrap();
        assert!(labels.is_empty());
    }

    #[test]
    fn test_parse_json_array_with_unicode() {
        let content = r#"["Pingüino", "鸟类", "Птица"]"#;
        let labels = parse_json_labels(content).unwrap();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], "Pingüino");
        assert_eq!(labels[1], "鸟类");
        assert_eq!(labels[2], "Птица");
    }

    #[test]
    fn test_parse_json_array_of_objects_missing_keys() {
        let content = r#"[{"name": "Species 1"}, {"other": "Species 2"}]"#;
        let result = parse_json_labels(content);
        // Objects without name/label/species keys are filtered out
        assert!(result.is_ok());
        let labels = result.unwrap();
        assert_eq!(labels.len(), 1); // Only the first object has a valid key
        assert_eq!(labels[0], "Species 1");
    }

    #[test]
    fn test_parse_json_deeply_nested() {
        let content = r#"{"data": {"labels": ["Species 1"]}}"#;
        let result = parse_json_labels(content);
        // Only supports one level of nesting
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_json_array_of_objects_species_key() {
        let content = r#"[{"species": "American Robin"}, {"species": "Northern Cardinal"}]"#;
        let labels = parse_json_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Northern Cardinal"]);
    }

    #[test]
    fn test_parse_text_labels_only_whitespace() {
        let content = "   \n\t\n  \n";
        let labels = parse_text_labels(content);
        // All empty/whitespace lines should be filtered out
        assert!(labels.is_empty());
    }

    #[test]
    fn test_parse_csv_labels_quoted_values() {
        let content = r#"label
"Species, with comma"
"Species with ""quotes"""
Species normal"#;
        let labels = parse_csv_labels(content).unwrap();
        // CSV parser should handle quoted values with commas and escaped quotes
        assert_eq!(
            labels,
            vec![
                "Species, with comma",
                "Species with \"quotes\"",
                "Species normal"
            ]
        );
    }

    // BirdNET v3.0 format tests

    #[test]
    fn test_parse_csv_labels_birdnet_v3_format() {
        let content = "idx;id;sci_name;com_name;class;order\n\
                       0;3;Abeillia abeillei;Emerald-chinned Hummingbird;Aves;Apodiformes\n\
                       1;5;Abroscopus albogularis;Rufous-faced Warbler;Aves;Passeriformes\n\
                       2;6;Abroscopus schisticeps;Black-faced Warbler;Aves;Passeriformes";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], "Abeillia abeillei");
        assert_eq!(labels[1], "Abroscopus albogularis");
        assert_eq!(labels[2], "Abroscopus schisticeps");
    }

    #[test]
    fn test_parse_csv_labels_semicolon_com_name_priority() {
        let content = "idx;id;other;com_name;class\n\
                       0;3;Something;Emerald-chinned Hummingbird;Aves\n\
                       1;5;Other;Rufous-faced Warbler;Aves";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], "Emerald-chinned Hummingbird");
        assert_eq!(labels[1], "Rufous-faced Warbler");
    }

    #[test]
    fn test_parse_csv_labels_numeric_first_column_no_header() {
        let content = "0;Abeillia abeillei;Extra\n\
                       1;Abroscopus albogularis;Extra\n\
                       2;Abroscopus schisticeps;Extra";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], "Abeillia abeillei");
    }

    #[test]
    fn test_detect_csv_delimiter_comma() {
        let content = "species,common_name\nSpecies1,Common1";
        let delimiter = detect_csv_delimiter(content);
        assert_eq!(delimiter, b',');
    }

    #[test]
    fn test_detect_csv_delimiter_semicolon() {
        let content = "idx;sci_name;com_name\n0;Species1;Common1";
        let delimiter = detect_csv_delimiter(content);
        assert_eq!(delimiter, b';');
    }

    #[test]
    fn test_parse_csv_labels_with_bom() {
        let content = "\u{FEFF}idx;sci_name\n0;Abeillia abeillei\n1;Test species";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], "Abeillia abeillei");
    }

    #[test]
    fn test_parse_csv_labels_backward_compat_single_column() {
        let content = "American Robin\nNorthern Cardinal\nBlue Jay";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(
            labels,
            vec!["American Robin", "Northern Cardinal", "Blue Jay"]
        );
    }

    #[test]
    fn test_parse_csv_labels_backward_compat_comma_multicolumn() {
        let content = "species,scientific\nAmerican Robin,Turdus migratorius\nCardinal,Cardinalis";
        let labels = parse_csv_labels(content).unwrap();
        assert_eq!(labels, vec!["American Robin", "Cardinal"]);
    }

    #[test]
    fn test_parse_csv_labels_priority_ordering() {
        // Verify that sci_name takes priority even when com_name appears first
        let content = "com_name;sci_name;other\n\
                       Common Name 1;Scientific Name 1;Extra\n\
                       Common Name 2;Scientific Name 2;Extra";
        let labels = parse_csv_labels(content).unwrap();
        // Should select sci_name (higher priority) not com_name
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], "Scientific Name 1");
        assert_eq!(labels[1], "Scientific Name 2");
    }
}
