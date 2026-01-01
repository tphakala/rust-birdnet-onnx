//! Runtime library detection and initialization.
//!
//! This module provides functionality to detect and load the ONNX Runtime
//! library from bundled locations relative to the executable.

use std::path::PathBuf;

use crate::Error;

/// Attempts to find the ONNX Runtime library in common locations.
///
/// # Search Order
///
/// 1. `ORT_DYLIB_PATH` environment variable (highest priority for overrides)
/// 2. `<exe_dir>/lib/libonnxruntime.so` (Linux bundled structure)
/// 3. `<exe_dir>/lib/libonnxruntime.so.*` (Linux versioned)
/// 4. `<cwd>/lib/libonnxruntime.so` (development/CI structure)
/// 5. `<cwd>/lib/libonnxruntime.so.*` (development/CI versioned)
/// 6. `<exe_dir>/onnxruntime.dll` (Windows bundled structure)
/// 7. `<exe_dir>/lib/libonnxruntime.dylib` (macOS bundled structure)
/// 8. Returns `None` to let ort use system library paths
#[must_use]
pub fn find_ort_library() -> Option<PathBuf> {
    // Check environment variable first (allows explicit override)
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let cwd = std::env::current_dir().ok();

    #[cfg(target_os = "linux")]
    {
        // Check exe_dir/lib/ first (bundled distribution)
        if let Some(path) = find_linux_lib(&exe_dir) {
            return Some(path);
        }

        // Check cwd/lib/ (development/CI)
        if let Some(ref cwd) = cwd
            && let Some(path) = find_linux_lib(cwd)
        {
            return Some(path);
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows: DLLs are in same directory as exe
        let bundled = exe_dir.join("onnxruntime.dll");
        if bundled.exists() {
            return Some(bundled);
        }

        // Also check cwd
        if let Some(ref cwd) = cwd {
            let dev = cwd.join("onnxruntime.dll");
            if dev.exists() {
                return Some(dev);
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: check lib/ subdirectory
        let bundled = exe_dir.join("lib").join("libonnxruntime.dylib");
        if bundled.exists() {
            return Some(bundled);
        }

        // Also check cwd/lib/
        if let Some(ref cwd) = cwd {
            let dev = cwd.join("lib").join("libonnxruntime.dylib");
            if dev.exists() {
                return Some(dev);
            }
        }
    }

    // Suppress unused variable warnings on non-target platforms
    let _ = cwd;

    None
}

/// Find ONNX Runtime library in a Linux lib directory.
#[cfg(target_os = "linux")]
fn find_linux_lib(base_dir: &std::path::Path) -> Option<PathBuf> {
    let lib_dir = base_dir.join("lib");

    // Check for exact name first
    let bundled = lib_dir.join("libonnxruntime.so");
    if bundled.exists() {
        return Some(bundled);
    }

    // Check for versioned .so files
    if let Ok(entries) = std::fs::read_dir(&lib_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("libonnxruntime.so") {
                return Some(entry.path());
            }
        }
    }

    None
}

/// Initialize ONNX Runtime with auto-detected library path.
///
/// This should be called once at application startup, before any model loading.
/// If a bundled library is found, it will be used. Otherwise, ort will fall back
/// to system library paths.
///
/// # Errors
///
/// Returns an error if the ONNX Runtime library is found but fails to initialize.
///
/// # Example
///
/// ```ignore
/// use birdnet_onnx::init_runtime;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Initialize runtime (auto-detects bundled libraries)
///     init_runtime()?;
///
///     // Now load models...
///     Ok(())
/// }
/// ```
pub fn init_runtime() -> Result<(), Error> {
    if let Some(lib_path) = find_ort_library() {
        ort::init_from(lib_path.display().to_string())
            .commit()
            .map_err(|e| Error::RuntimeInit(e.to_string()))?;
    }
    // If no bundled library found, ort will try system paths automatically
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_find_ort_library_returns_option() {
        // This test just verifies the function doesn't panic
        // The actual result depends on the runtime environment
        let _ = find_ort_library();
    }
}
