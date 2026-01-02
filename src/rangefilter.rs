//! Range filter for location and date-based species filtering

use crate::error::{Error, Result};

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
}
