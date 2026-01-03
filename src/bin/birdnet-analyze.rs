//! CLI tool for analyzing WAV files for bird species using `BirdNET`/`Perch` models.

#![allow(clippy::print_stdout)] // CLI tool needs stdout
#![allow(clippy::print_stderr)] // CLI tool needs stderr

use birdnet_onnx::{Classifier, ModelType, Result, init_runtime};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

/// Normalization factor for 16-bit signed audio samples.
/// This is 2^15 (32768), used to convert i16 samples to f32 range [-1.0, 1.0].
const I16_NORMALIZATION_FACTOR: f32 = 32768.0;

/// Analyze WAV files for bird species using `BirdNET`/`Perch` ONNX models.
#[derive(Parser, Debug)]
#[command(name = "birdnet-analyze")]
#[command(about = "Analyze WAV files for bird species")]
struct Args {
    /// Input WAV file (16-bit mono, matching model sample rate)
    audio_file: PathBuf,

    /// Path to ONNX model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to labels file
    #[arg(short, long)]
    labels: PathBuf,

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

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Initialize ONNX Runtime (auto-detects bundled libraries)
    init_runtime()?;

    // Parse model type override if provided
    let model_type_override = parse_model_type(args.model_type.as_deref())?;

    // Build classifier
    let mut builder = Classifier::builder()
        .model_path(args.model.display().to_string())
        .labels_path(args.labels.display().to_string())
        .top_k(args.top_k)
        .min_confidence(args.min_confidence);

    if let Some(mt) = model_type_override {
        builder = builder.model_type(mt);
    }

    let classifier = builder.build()?;
    let config = classifier.config();

    // Read WAV file
    let (samples, sample_rate, duration_secs) = read_wav(&args.audio_file)?;

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

    // Print header
    let model_name = model_display_name(config.model_type);
    println!(
        "Analyzing: {} ({}, {} Hz)",
        args.audio_file.display(),
        format_duration(duration_secs),
        sample_rate
    );
    println!(
        "Model: {} ({:.1}s segments, {:.1}s overlap)",
        model_name, config.segment_duration, args.overlap
    );
    println!();

    // Chunk audio and run inference
    let start_time = Instant::now();
    let segments = chunk_audio(&samples, config.sample_count, args.overlap, sample_rate);
    let segment_count = segments.len();

    for (time_offset, segment) in segments {
        let result = classifier.predict(&segment)?;

        if result.predictions.is_empty() {
            continue;
        }

        // Format predictions
        let preds: Vec<String> = result
            .predictions
            .iter()
            .map(|p| format!("{} ({:.1}%)", p.species, p.confidence * 100.0))
            .collect();

        println!("{}  {}", format_time(time_offset), preds.join(", "));
    }

    let elapsed = start_time.elapsed();
    println!();
    println!(
        "{} segments analyzed in {:.1}s",
        segment_count,
        elapsed.as_secs_f32()
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

/// Format duration as `XmYs`.
fn format_duration(secs: f32) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let total_secs = secs as u32;
    let mins = total_secs / 60;
    let secs_part = total_secs % 60;
    if mins > 0 {
        format!("{mins}m {secs_part}s")
    } else {
        format!("{secs_part}s")
    }
}
