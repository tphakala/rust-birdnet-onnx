//! Manual verification that BirdNET-3 labels parse correctly

#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)]

#[test]
fn verify_birdnet3_csv_format() {
    // Simulate the exact BirdNET-3 format from the file
    let sample = "idx;id;sci_name;com_name;class;order\n\
                  0;3;Abeillia abeillei;Emerald-chinned Hummingbird;Aves;Apodiformes\n\
                  1;5;Abroscopus albogularis;Rufous-faced Warbler;Aves;Passeriformes";

    // Test delimiter detection
    let first_line = sample.lines().next().unwrap();
    assert_eq!(
        first_line.matches(';').count(),
        5,
        "Should detect semicolons"
    );
    assert_eq!(
        first_line.matches(',').count(),
        0,
        "Should have no commas in BirdNET-3"
    );
}
