//! CLI tool for analyzing WAV files for bird species using `BirdNET`/`Perch` models.

#![allow(clippy::print_stdout)] // CLI tool needs stdout
#![allow(clippy::print_stderr)] // CLI tool needs stderr

use birdnet_onnx::{
    BatchInferenceContext, CancellationToken, Classifier, ExecutionProviderInfo, InferenceOptions,
    ModelType, Result, available_execution_providers,
};
#[cfg(feature = "load-dynamic")]
use birdnet_onnx::{find_ort_library, init_runtime};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tracing_subscriber::{EnvFilter, fmt};

/// Normalization factor for 16-bit signed audio samples.
/// This is 2^15 (32768), used to convert i16 samples to f32 range [-1.0, 1.0].
const I16_NORMALIZATION_FACTOR: f32 = 32768.0;

/// All supported execution providers in canonical order.
const ALL_EXECUTION_PROVIDERS: &[ExecutionProviderInfo] = &[
    ExecutionProviderInfo::Cpu,
    ExecutionProviderInfo::Cuda,
    ExecutionProviderInfo::TensorRt,
    ExecutionProviderInfo::DirectMl,
    ExecutionProviderInfo::CoreMl,
    ExecutionProviderInfo::Rocm,
    ExecutionProviderInfo::OpenVino,
    ExecutionProviderInfo::OneDnn,
    ExecutionProviderInfo::Qnn,
    ExecutionProviderInfo::Acl,
    ExecutionProviderInfo::ArmNn,
];

/// Default batch size for CPU inference (conservative for cache efficiency).
const DEFAULT_CPU_BATCH_SIZE: usize = 8;

/// Default batch size for GPU inference (safe for 4GB+ VRAM, ~38% usage).
const DEFAULT_GPU_BATCH_SIZE: usize = 32;

/// Analyze WAV files for bird species using `BirdNET`/`Perch` ONNX models.
#[derive(Parser, Debug)]
#[command(name = "birdnet-analyze")]
#[command(about = "Analyze WAV files for bird species")]
struct Args {
    /// Input WAV file (16-bit mono, matching model sample rate)
    #[arg(required_unless_present = "list_providers")]
    audio_file: Option<PathBuf>,

    /// Path to ONNX model file
    #[arg(short, long, required_unless_present = "list_providers")]
    model: Option<PathBuf>,

    /// Path to labels file
    #[arg(short, long, required_unless_present = "list_providers")]
    labels: Option<PathBuf>,

    /// Overlap between segments in seconds
    #[arg(short, long, default_value = "0.0")]
    overlap: f32,

    /// Number of top predictions per segment
    #[arg(short = 'k', long, default_value = "3")]
    top_k: usize,

    /// Minimum confidence threshold
    #[arg(long, default_value = "0.1")]
    min_confidence: f32,

    /// Override model type detection (v24, v30, perch)
    #[arg(long)]
    model_type: Option<String>,

    /// List available execution providers and exit
    #[arg(long)]
    list_providers: bool,

    /// Execution provider to use (cpu, cuda, tensorrt, directml, coreml, rocm, openvino, onednn, qnn, acl, armnn)
    #[arg(long, default_value = "cpu")]
    provider: String,

    /// Batch size for inference (defaults: 8 for CPU, 32 for GPU)
    #[arg(short, long)]
    batch_size: Option<usize>,

    /// Timeout per batch in seconds (0 = no timeout)
    #[arg(short, long, default_value = "1")]
    timeout: u64,

    /// Enable verbose logging for debugging
    #[arg(short, long)]
    verbose: bool,
}

/// Parse model type from CLI argument.
fn parse_model_type(arg: Option<&str>) -> Result<Option<ModelType>> {
    match arg {
        Some("v24") => Ok(Some(ModelType::BirdNetV24)),
        Some("v30") => Ok(Some(ModelType::BirdNetV30)),
        Some("perch") => Ok(Some(ModelType::PerchV2)),
        Some(other) => Err(birdnet_onnx::Error::ModelDetection {
            reason: format!("unknown model type '{other}', expected: v24, v30, perch"),
        }),
        None => Ok(None),
    }
}

/// Get display name for model type.
const fn model_display_name(model_type: ModelType) -> &'static str {
    match model_type {
        ModelType::BirdNetV24 => "BirdNET v2.4",
        ModelType::BirdNetV30 => "BirdNET v3.0",
        ModelType::PerchV2 => "Perch v2",
    }
}

/// Parse execution provider from CLI argument.
fn parse_provider(s: &str) -> Result<ExecutionProviderInfo> {
    ALL_EXECUTION_PROVIDERS
        .iter()
        .find(|p| p.as_str().eq_ignore_ascii_case(s))
        .copied()
        .ok_or_else(|| {
            let provider_names: Vec<&str> =
                ALL_EXECUTION_PROVIDERS.iter().map(|p| p.as_str()).collect();
            birdnet_onnx::Error::ModelDetection {
                reason: format!(
                    "unknown provider '{s}'. Valid providers: {}",
                    provider_names.join(", ")
                ),
            }
        })
}

/// Get human-readable description for execution provider.
const fn provider_description(provider: ExecutionProviderInfo) -> &'static str {
    match provider {
        ExecutionProviderInfo::Cpu => "Always available",
        ExecutionProviderInfo::Cuda => "NVIDIA GPU acceleration",
        ExecutionProviderInfo::TensorRt => "NVIDIA GPU with optimization",
        ExecutionProviderInfo::DirectMl => "DirectX 12 GPU acceleration (Windows)",
        ExecutionProviderInfo::CoreMl => "Apple Neural Engine acceleration (macOS/iOS)",
        ExecutionProviderInfo::Rocm => "AMD GPU acceleration",
        ExecutionProviderInfo::OpenVino => "Intel hardware acceleration",
        ExecutionProviderInfo::OneDnn => "Intel CPU optimization",
        ExecutionProviderInfo::Qnn => "Qualcomm NPU acceleration",
        ExecutionProviderInfo::Acl => "Arm CPU optimization",
        ExecutionProviderInfo::ArmNn => "Arm NPU acceleration",
    }
}

/// List all execution providers and exit.
#[allow(clippy::unnecessary_wraps)] // Returns Result for load-dynamic feature
fn list_providers_and_exit() -> Result<()> {
    // Dynamic loading: Initialize ONNX Runtime and show library path
    #[cfg(feature = "load-dynamic")]
    {
        if let Err(e) = init_runtime() {
            eprintln!("Error initializing ONNX Runtime: {e}");
            eprintln!();
            #[cfg(target_os = "windows")]
            {
                eprintln!("Windows DLL search order for onnxruntime.dll:");
                if let Ok(exe_path) = std::env::current_exe() {
                    if let Some(exe_dir) = exe_path.parent() {
                        eprintln!("  1. Exe directory: {}\\onnxruntime.dll", exe_dir.display());
                    }
                }
                if let Ok(cwd) = std::env::current_dir() {
                    eprintln!("  2. Current directory: {}\\onnxruntime.dll", cwd.display());
                }
                eprintln!("  3. System directory: C:\\Windows\\System32\\onnxruntime.dll");
                eprintln!("  4. Windows directory: C:\\Windows\\onnxruntime.dll");
                eprintln!("  5. Directories in PATH environment variable");
                eprintln!();
                eprintln!("To search your entire system:");
                eprintln!(
                    "  Get-ChildItem -Path C:\\ -Filter onnxruntime.dll -Recurse -ErrorAction SilentlyContinue"
                );
            }
            return Err(e);
        }

        // Show ONNX Runtime library path for debugging
        if let Some(lib_path) = find_ort_library() {
            println!("ONNX Runtime library: {}", lib_path.display());
        } else {
            println!("ONNX Runtime library: <using system library paths>");
            #[cfg(target_os = "windows")]
            {
                println!("  Note: On Windows, this searches the PATH for onnxruntime.dll");
                println!("  To find the current location, use: where onnxruntime.dll");
            }
            #[cfg(target_os = "linux")]
            {
                println!(
                    "  Note: On Linux, this searches paths in /etc/ld.so.conf and the LD_LIBRARY_PATH environment variable."
                );
                println!("  To find the library, you can try: ldconfig -p | grep onnxruntime");
            }
            #[cfg(target_os = "macos")]
            {
                println!(
                    "  Note: On macOS, this searches standard paths and the DYLD_LIBRARY_PATH environment variable."
                );
                println!(
                    "  To find the library, you can try: find /usr/local /opt -name \"libonnxruntime.dylib\" 2>/dev/null"
                );
            }
        }
        println!();
    }

    // Static linking: ONNX Runtime is embedded in the binary
    #[cfg(not(feature = "load-dynamic"))]
    {
        println!("ONNX Runtime library: <statically linked>");
        println!();
    }

    let available = available_execution_providers();

    println!("Available execution providers:");

    for &provider in ALL_EXECUTION_PROVIDERS {
        let is_available = available.contains(&provider);
        let symbol = if is_available { "✓" } else { "✗" };
        let name = provider.as_str();
        let description = provider_description(provider);

        if is_available {
            println!("  {symbol} {name} - {description}");
        } else {
            println!("  {symbol} {name} - {description} (not available)");
            println!(
                "    Reason: Provider not available (may require library installation or ONNX Runtime built with provider support)"
            );
        }
    }

    Ok(())
}

fn main() {
    // Parse args with custom error handling for better help messages
    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) => {
            // If parsing failed and no args were provided, show helpful message
            if std::env::args().len() == 1 {
                eprintln!("Analyze WAV files for bird species using BirdNET/Perch ONNX models.\n");
                eprintln!("Usage:");
                eprintln!("  # Analyze audio file:");
                eprintln!("  birdnet-analyze --model <MODEL> --labels <LABELS> <AUDIO_FILE>\n");
                eprintln!("  # List available execution providers:");
                eprintln!("  birdnet-analyze --list-providers\n");
                eprintln!("For more information, try '--help'.");
                std::process::exit(2);
            }
            // Otherwise show the default clap error
            e.exit();
        }
    };

    if let Err(e) = run_with_args(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[allow(clippy::needless_pass_by_value)] // Args is consumed throughout this function
fn run_with_args(args: Args) -> Result<()> {
    // Handle --list-providers flag
    if args.list_providers {
        list_providers_and_exit()?;
        std::process::exit(0);
    }

    // Configure verbose logging with tracing
    if args.verbose {
        // Initialize tracing subscriber with timestamps for ort internal logging
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("ort=debug"));

        fmt()
            .with_target(false)
            .with_level(true)
            .with_timer(fmt::time::SystemTime)
            .with_env_filter(filter)
            .init();

        eprintln!(
            "{} [DEBUG] Verbose logging enabled (RUST_LOG=ort=debug)",
            timestamp()
        );
    }

    // Initialize ONNX Runtime (only needed for dynamic loading)
    #[cfg(feature = "load-dynamic")]
    {
        if args.verbose {
            eprintln!("{} [DEBUG] Initializing ONNX Runtime...", timestamp());
        }
        let init_start = Instant::now();
        init_runtime()?;
        if args.verbose {
            eprintln!(
                "{} [DEBUG] ONNX Runtime initialized in {:?}",
                timestamp(),
                init_start.elapsed()
            );
        }
    }

    // Parse and validate execution provider
    let requested_provider = parse_provider(&args.provider)?;
    let available_providers = available_execution_providers();

    if !available_providers.contains(&requested_provider) {
        let available_names: Vec<String> = available_providers
            .iter()
            .map(|p| p.as_str().to_string())
            .collect();
        return Err(birdnet_onnx::Error::ModelDetection {
            reason: format!(
                "{} provider not available\nAvailable providers: {}",
                requested_provider.as_str(),
                available_names.join(", ")
            ),
        });
    }

    // Determine batch size based on provider
    let batch_size = args.batch_size.unwrap_or_else(|| {
        if requested_provider == ExecutionProviderInfo::Cpu {
            DEFAULT_CPU_BATCH_SIZE
        } else {
            DEFAULT_GPU_BATCH_SIZE
        }
    });

    // Parse model type override if provided
    let model_type_override = parse_model_type(args.model_type.as_deref())?;

    // Unwrap required arguments (safe because of required_unless_present)
    let audio_file =
        args.audio_file
            .as_ref()
            .ok_or_else(|| birdnet_onnx::Error::ModelDetection {
                reason: "audio file is required".to_string(),
            })?;
    let model_path = args
        .model
        .as_ref()
        .ok_or_else(|| birdnet_onnx::Error::ModelDetection {
            reason: "model path is required".to_string(),
        })?;
    let labels_path = args
        .labels
        .as_ref()
        .ok_or_else(|| birdnet_onnx::Error::ModelDetection {
            reason: "labels path is required".to_string(),
        })?;

    // Build classifier with selected execution provider
    if args.verbose {
        eprintln!(
            "{} [DEBUG] Building classifier with {} provider...",
            timestamp(),
            requested_provider.as_str()
        );
    }
    let build_start = Instant::now();

    let mut builder = Classifier::builder()
        .model_path(model_path.display().to_string())
        .labels_path(labels_path.display().to_string())
        .top_k(args.top_k)
        .min_confidence(args.min_confidence);

    if let Some(mt) = model_type_override {
        builder = builder.model_type(mt);
    }

    // Configure execution provider
    builder = match requested_provider {
        ExecutionProviderInfo::Cpu => builder, // CPU is default, no need to add
        ExecutionProviderInfo::Cuda => builder.with_cuda(),
        ExecutionProviderInfo::TensorRt => builder.with_tensorrt(),
        ExecutionProviderInfo::DirectMl => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::DirectMLExecutionProvider::default(),
        ),
        ExecutionProviderInfo::CoreMl => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::CoreMLExecutionProvider::default(),
        ),
        ExecutionProviderInfo::Rocm => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::ROCmExecutionProvider::default(),
        ),
        ExecutionProviderInfo::OpenVino => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::OpenVINOExecutionProvider::default(),
        ),
        ExecutionProviderInfo::OneDnn => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::OneDNNExecutionProvider::default(),
        ),
        ExecutionProviderInfo::Qnn => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::QNNExecutionProvider::default(),
        ),
        ExecutionProviderInfo::Acl => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::ACLExecutionProvider::default(),
        ),
        ExecutionProviderInfo::ArmNn => builder.execution_provider(
            birdnet_onnx::ort_execution_providers::ArmNNExecutionProvider::default(),
        ),
    };

    let classifier = builder.build()?;
    if args.verbose {
        eprintln!(
            "{} [DEBUG] Classifier built in {:?}",
            timestamp(),
            build_start.elapsed()
        );
    }
    let config = classifier.config();

    // Read WAV file
    if args.verbose {
        eprintln!(
            "{} [DEBUG] Reading WAV file: {}",
            timestamp(),
            audio_file.display()
        );
    }
    let wav_start = Instant::now();
    let (samples, sample_rate, duration_secs) = read_wav(audio_file)?;
    if args.verbose {
        eprintln!(
            "{} [DEBUG] WAV file read in {:?}",
            timestamp(),
            wav_start.elapsed()
        );
    }

    // Validate sample rate
    if sample_rate != config.sample_rate {
        return Err(birdnet_onnx::Error::AudioFormat {
            reason: format!(
                "model expects {} Hz audio, WAV is {} Hz",
                config.sample_rate, sample_rate
            ),
        });
    }

    // Validate overlap
    if args.overlap >= config.segment_duration {
        return Err(birdnet_onnx::Error::ModelDetection {
            reason: format!(
                "overlap ({:.1}s) must be less than segment duration ({:.1}s)",
                args.overlap, config.segment_duration
            ),
        });
    }

    // Create batch context for GPU providers (enables memory reuse via IoBinding)
    // Falls back to regular predict_batch for CPU or unsupported models (PerchV2)
    let mut batch_context: Option<BatchInferenceContext> = if requested_provider
        == ExecutionProviderInfo::Cpu
    {
        None
    } else {
        match classifier.create_batch_context(batch_size) {
            Ok(ctx) => {
                if args.verbose {
                    let buffer_bytes = ctx.input_buffer_bytes();
                    #[allow(clippy::cast_precision_loss)]
                    let buffer_mb = buffer_bytes as f64 / (1024.0 * 1024.0);
                    eprintln!(
                        "{} [DEBUG] Created IoBinding batch context (max_batch_size={}, input_buffer={:.1}MB pre-allocated)",
                        timestamp(),
                        batch_size,
                        buffer_mb
                    );
                }
                Some(ctx)
            }
            Err(e) => {
                if args.verbose {
                    eprintln!(
                        "{} [DEBUG] IoBinding not available: {e}, using standard batch inference",
                        timestamp()
                    );
                }
                None
            }
        }
    };

    // Print header
    let model_name = model_display_name(config.model_type);
    println!("Using execution provider: {}", requested_provider.as_str());
    if batch_context.is_some() {
        println!("Batch size: {batch_size} (IoBinding enabled)");
    } else {
        println!("Batch size: {batch_size}");
    }
    println!(
        "Analyzing: {} ({}, {} Hz)",
        audio_file.display(),
        format_duration(duration_secs),
        sample_rate
    );
    println!(
        "Model: {} ({:.1}s segments, {:.1}s overlap)",
        model_name, config.segment_duration, args.overlap
    );
    println!();

    // Chunk audio and run inference
    if args.verbose {
        eprintln!("{} [DEBUG] Chunking audio into segments...", timestamp());
    }
    let chunk_start = Instant::now();
    let segments = chunk_audio(&samples, config.sample_count, args.overlap, sample_rate);
    let segment_count = segments.len();
    if args.verbose {
        eprintln!(
            "{} [DEBUG] Created {} segments in {:?}",
            timestamp(),
            segment_count,
            chunk_start.elapsed()
        );
        eprintln!(
            "{} [DEBUG] Starting inference (batch_size={batch_size})...",
            timestamp()
        );
    }

    // Set up Ctrl+C handler for graceful cancellation
    let cancel_token = CancellationToken::new();
    let cancel_token_handler = cancel_token.clone();
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_handler = Arc::clone(&cancelled);

    ctrlc::set_handler(move || {
        if cancelled_handler.swap(true, Ordering::SeqCst) {
            // Second Ctrl+C - force exit
            eprintln!("\nForce exiting...");
            std::process::exit(1);
        }
        eprintln!("\nCancelling... (press Ctrl+C again to force exit)");
        cancel_token_handler.cancel();
    })
    .ok(); // Ignore error if handler already set

    let start_time = Instant::now();
    let mut batch_num = 0;

    // Process segments in batches for better GPU performance
    for batch_chunk in segments.chunks(batch_size) {
        // Check if cancelled before starting batch
        if cancelled.load(Ordering::SeqCst) {
            eprintln!("Processing cancelled by user");
            break;
        }
        batch_num += 1;

        // Prepare batch: collect references to segment data
        let batch_segments: Vec<&[f32]> = batch_chunk
            .iter()
            .map(|(_, segment)| segment.as_slice())
            .collect();

        // Run batch inference
        if args.verbose {
            let total_batches = segment_count.div_ceil(batch_size);
            if batch_context.is_some() {
                eprintln!(
                    "{} [DEBUG] Processing batch {}/{} ({} segments, reusing pre-allocated input buffer)...",
                    timestamp(),
                    batch_num,
                    total_batches,
                    batch_segments.len()
                );
            } else {
                eprintln!(
                    "{} [DEBUG] Processing batch {}/{} ({} segments)...",
                    timestamp(),
                    batch_num,
                    total_batches,
                    batch_segments.len()
                );
            }
        }
        let batch_start = Instant::now();
        let inference_options = if args.timeout > 0 {
            InferenceOptions::timeout(Duration::from_secs(args.timeout))
                .with_cancellation_token(cancel_token.clone())
        } else {
            InferenceOptions::new().with_cancellation_token(cancel_token.clone())
        };

        // Use IoBinding context if available, otherwise fall back to standard batch inference
        let results = if let Some(ref mut ctx) = batch_context {
            classifier.predict_batch_with_context(ctx, &batch_segments, &inference_options)?
        } else {
            classifier.predict_batch(&batch_segments, &inference_options)?
        };
        if args.verbose {
            eprintln!(
                "{} [DEBUG] Batch {} completed in {:?}",
                timestamp(),
                batch_num,
                batch_start.elapsed()
            );
        }

        // Process results with their corresponding time offsets
        for ((time_offset, _), result) in batch_chunk.iter().zip(results) {
            if result.predictions.is_empty() {
                continue;
            }

            // Format predictions
            let preds: Vec<String> = result
                .predictions
                .iter()
                .map(|p| format!("{} ({:.1}%)", p.species, p.confidence * 100.0))
                .collect();

            println!("{}  {}", format_time(*time_offset), preds.join(", "));
        }
    }

    let elapsed = start_time.elapsed();
    let elapsed_secs = elapsed.as_secs_f32();
    #[allow(clippy::cast_precision_loss)]
    let segments_per_sec = segment_count as f32 / elapsed_secs;
    let audio_secs_per_sec = duration_secs / elapsed_secs;
    let audio_duration = format_duration(duration_secs);

    println!();
    println!(
        "{segment_count} segments of {audio_duration} audio analyzed in {elapsed_secs:.1}s ({segments_per_sec:.1} segments/s, {audio_secs_per_sec:.1}x realtime)"
    );

    Ok(())
}

/// Read WAV file and return samples, sample rate, and duration.
fn read_wav(path: &PathBuf) -> Result<(Vec<f32>, u32, f32)> {
    let reader = hound::WavReader::open(path).map_err(|e| birdnet_onnx::Error::AudioRead {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;

    let spec = reader.spec();

    // Validate format
    if spec.channels != 1 {
        return Err(birdnet_onnx::Error::AudioFormat {
            reason: format!(
                "WAV must be mono (1 channel), got {} channels",
                spec.channels
            ),
        });
    }

    if spec.bits_per_sample != 16 {
        return Err(birdnet_onnx::Error::AudioFormat {
            reason: format!("WAV must be 16-bit, got {}-bit", spec.bits_per_sample),
        });
    }

    if spec.sample_format != hound::SampleFormat::Int {
        return Err(birdnet_onnx::Error::AudioFormat {
            reason: "WAV must be integer format, not float".to_string(),
        });
    }

    // Read samples and convert to f32 [-1.0, 1.0]
    let samples: std::result::Result<Vec<f32>, _> = reader
        .into_samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) / I16_NORMALIZATION_FACTOR))
        .collect();

    let samples = samples.map_err(|e| birdnet_onnx::Error::AudioRead {
        path: path.display().to_string(),
        reason: format!("failed to read samples: {e}"),
    })?;

    if samples.is_empty() {
        return Err(birdnet_onnx::Error::AudioFormat {
            reason: "WAV file has no samples".to_string(),
        });
    }

    #[allow(clippy::cast_precision_loss)]
    let duration = samples.len() as f32 / spec.sample_rate as f32;

    Ok((samples, spec.sample_rate, duration))
}

/// Chunk audio into segments with overlap.
fn chunk_audio(
    samples: &[f32],
    segment_samples: usize,
    overlap_secs: f32,
    sample_rate: u32,
) -> Vec<(f32, Vec<f32>)> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let overlap_samples = (overlap_secs * sample_rate as f32) as usize;

    let step = segment_samples.saturating_sub(overlap_samples);
    if step == 0 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + segment_samples).min(samples.len());
        let mut segment = samples[pos..end].to_vec();

        // Pad if needed
        segment.resize(segment_samples, 0.0);

        #[allow(clippy::cast_precision_loss)]
        let start_time = pos as f32 / sample_rate as f32;
        segments.push((start_time, segment));

        pos += step;
    }

    segments
}

/// Format time offset as MM:SS.d
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn format_time(secs: f32) -> String {
    let total_secs = secs as u32;
    let mins = total_secs / 60;
    let secs_part = secs - (mins * 60) as f32;
    format!("{mins:02}:{secs_part:04.1}")
}

/// Format duration as human-readable string (e.g., "45s", "3m 23s", "1h 15m 30s").
fn format_duration(secs: f32) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total_secs = secs as u32;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs_part = total_secs % 60;

    if hours > 0 {
        format!("{hours}h {mins}m {secs_part}s")
    } else if mins > 0 {
        format!("{mins}m {secs_part}s")
    } else {
        format!("{secs_part}s")
    }
}

/// Get current timestamp in ISO 8601 format with milliseconds.
fn timestamp() -> String {
    let now = std::time::SystemTime::now();
    let datetime: chrono::DateTime<chrono::Utc> = now.into();
    datetime.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}
