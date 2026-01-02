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
    #![allow(clippy::disallowed_methods)]
    #![allow(unsafe_code)]
    use super::*;
    use std::env;

    #[test]
    fn test_find_ort_library_returns_option() {
        // This test just verifies the function doesn't panic
        // The actual result depends on the runtime environment
        let _ = find_ort_library();
    }

    #[test]
    fn test_find_ort_library_with_env_var() {
        // Save original env var if set
        let original = env::var("ORT_DYLIB_PATH").ok();

        // Test with non-existent path (should return None or fallback)
        unsafe {
            env::set_var("ORT_DYLIB_PATH", "/nonexistent/path/libonnxruntime.so");
        }
        let result = find_ort_library();
        // Should either return None or find a system library
        let _ = result;

        // Restore original env var
        unsafe {
            if let Some(val) = original {
                env::set_var("ORT_DYLIB_PATH", val);
            } else {
                env::remove_var("ORT_DYLIB_PATH");
            }
        }
    }

    #[test]
    fn test_find_ort_library_env_var_precedence() {
        // Save original env var
        let original = env::var("ORT_DYLIB_PATH").ok();

        // If ORT_DYLIB_PATH is set to an existing file, it should be returned
        if let Some(ref path) = original {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                unsafe {
                    env::set_var("ORT_DYLIB_PATH", path);
                }
                let result = find_ort_library();
                assert!(result.is_some());
                assert_eq!(result.unwrap(), path_buf);
            }
        }

        // Restore
        unsafe {
            if let Some(val) = original {
                env::set_var("ORT_DYLIB_PATH", val);
            } else {
                env::remove_var("ORT_DYLIB_PATH");
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_linux_lib_nonexistent_dir() {
        let fake_dir = PathBuf::from("/nonexistent/directory/that/does/not/exist");
        let result = find_linux_lib(&fake_dir);
        assert!(result.is_none());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_find_linux_lib_empty_dir() {
        // Use /tmp as a directory that exists but likely doesn't have ONNX runtime
        let tmp_dir = env::temp_dir();
        let result = find_linux_lib(&tmp_dir);
        // Should return None if no library is found
        // (This could be Some if someone actually has libonnxruntime.so in /tmp)
        let _ = result;
    }

    #[test]
    fn test_init_runtime_doesnt_panic() {
        // Test that init_runtime doesn't panic
        // The actual result depends on whether a library is found
        let result = init_runtime();
        // Either succeeds or fails with an error, but shouldn't panic
        let _ = result;
    }

    #[test]
    fn test_init_runtime_multiple_calls() {
        // Calling init_runtime multiple times should not panic
        let _ = init_runtime();
        let _ = init_runtime();
        let _ = init_runtime();
    }

    #[test]
    fn test_pathbuf_from_env_nonexistent() {
        let fake_path = "/this/path/definitely/does/not/exist/libonnxruntime.so";
        let path_buf = PathBuf::from(fake_path);
        assert!(!path_buf.exists());
        // Verifies PathBuf creation doesn't validate existence
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_find_linux_lib_with_current_dir() {
        // Test finding library relative to current directory
        let cwd = env::current_dir().unwrap();
        let result = find_linux_lib(&cwd);

        // If lib/libonnxruntime.so exists in cwd, it should be found
        let expected = cwd.join("lib").join("libonnxruntime.so");
        if expected.exists() {
            assert_eq!(result, Some(expected));
        }
    }

    #[test]
    fn test_find_ort_library_consistent_results() {
        // Calling multiple times should give consistent results
        let result1 = find_ort_library();
        let result2 = find_ort_library();
        assert_eq!(result1, result2);
    }
}
