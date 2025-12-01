//! PQC Scheduler - Hybrid Classical-Quantum Cryptographic Operation Scheduler
//!
//! A cryptographic scheduling framework with formal guarantees for the post-quantum
//! transition, implementing dynamic algorithm selection, multi-objective resource
//! optimization, graceful degradation, and verified migration management.
//!
//! Copyright (c) 2025 Dyber, Inc. All Rights Reserved.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::signal;
use tracing::{info, warn, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use pqc_scheduler::{
    config::Config,
    metrics::MetricsExporter,
    scheduler::Scheduler,
    workload::WorkloadGenerator,
};

/// Hybrid Classical-Quantum Cryptographic Operation Scheduler
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config/default.yaml")]
    config: PathBuf,

    /// Path to workload definition file
    #[arg(short, long, default_value = "workloads/cloudflare_mmpp.json")]
    workload: PathBuf,

    /// Duration of the experiment in seconds
    #[arg(short, long, default_value = "3600")]
    duration: u64,

    /// Warmup period in seconds (excluded from metrics)
    #[arg(long, default_value = "300")]
    warmup: u64,

    /// Random seed for reproducibility
    #[arg(short, long, default_value = "42")]
    seed: u64,

    /// Path to output results file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Override threat level (0.0 - 1.0)
    #[arg(long)]
    threat_level: Option<f64>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable metrics exporter
    #[arg(long, default_value = "true")]
    metrics: bool,

    /// Metrics port
    #[arg(long, default_value = "9090")]
    metrics_port: u16,

    /// Dry run mode (validate configuration without execution)
    #[arg(long)]
    dry_run: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    init_logging(args.verbose)?;

    info!("PQC Scheduler v{}", env!("CARGO_PKG_VERSION"));
    info!("Copyright (c) 2025 Dyber, Inc.");
    info!("-----------------------------------");

    // Load configuration
    info!("Loading configuration from {:?}", args.config);
    let mut config = Config::from_file(&args.config)
        .with_context(|| format!("Failed to load config from {:?}", args.config))?;

    // Apply command-line overrides
    if let Some(threat_level) = args.threat_level {
        info!("Overriding threat level to {:.2}", threat_level);
        config.threat.model.initial_threat_level = threat_level;
    }

    // Validate configuration
    config.validate().context("Configuration validation failed")?;
    info!("Configuration validated successfully");

    if args.dry_run {
        info!("Dry run mode - exiting after validation");
        return Ok(());
    }

    // Load workload definition
    info!("Loading workload from {:?}", args.workload);
    let mut workload = WorkloadGenerator::from_file(&args.workload)
        .with_context(|| format!("Failed to load workload from {:?}", args.workload))?;
    info!("Workload: {} ({})", workload.name(), workload.description());

    // Initialize metrics exporter
    let metrics_handle = if args.metrics {
        info!("Starting metrics exporter on port {}", args.metrics_port);
        Some(MetricsExporter::start(args.metrics_port).await?)
    } else {
        None
    };

    // Initialize scheduler
    info!("Initializing scheduler...");
    let mut scheduler = Scheduler::new(config, args.seed)
        .await
        .context("Failed to initialize scheduler")?;
    
    info!("Scheduler initialized with {} algorithms enabled", 
          scheduler.algorithm_count());

    // Run warmup period
    if args.warmup > 0 {
        info!("Starting warmup period ({} seconds)...", args.warmup);
        scheduler.set_warmup_mode(true);
        run_workload(&mut scheduler, &mut workload, Duration::from_secs(args.warmup)).await?;
        scheduler.set_warmup_mode(false);
        scheduler.reset_metrics();
        info!("Warmup complete");
    }

    // Run main experiment
    info!("Starting main experiment ({} seconds)...", args.duration);
    let start_time = std::time::Instant::now();
    
    // Set up graceful shutdown handler
    let scheduler_handle = scheduler.handle();
    tokio::spawn(async move {
        match signal::ctrl_c().await {
            Ok(()) => {
                warn!("Received shutdown signal");
                scheduler_handle.shutdown();
            }
            Err(err) => {
                warn!("Error listening for shutdown signal: {}", err);
            }
        }
    });

    // Run the workload
    run_workload(&mut scheduler, &mut workload, Duration::from_secs(args.duration)).await?;
    
    let elapsed = start_time.elapsed();
    info!("Experiment completed in {:.2}s", elapsed.as_secs_f64());

    // Collect final metrics
    let results = scheduler.collect_results();
    info!("-----------------------------------");
    info!("Results Summary:");
    info!("  Throughput: {:.0} ± {:.0} ops/s", 
          results.throughput_mean, results.throughput_std);
    info!("  Latency (mean): {:.1} ± {:.1} μs", 
          results.latency_mean_us, results.latency_std_us);
    info!("  Latency (p99): {:.1} μs", results.latency_p99_us);
    info!("  SLA Compliance: {:.2}%", results.sla_compliance * 100.0);
    info!("  Security Score: {:.2}", results.security_score);
    info!("  Algorithm Switches: {}", results.algorithm_switches);
    info!("  Fallback Events: {}", results.fallback_events);

    // Write results to file
    if let Some(output_path) = args.output {
        info!("Writing results to {:?}", output_path);
        results.write_json(&output_path)
            .with_context(|| format!("Failed to write results to {:?}", output_path))?;
    }

    // Shutdown metrics exporter
    if let Some(handle) = metrics_handle {
        handle.shutdown().await;
    }

    // Shutdown scheduler
    scheduler.shutdown().await?;

    info!("PQC Scheduler shutdown complete");
    Ok(())
}

/// Initialize the logging subsystem
fn init_logging(verbose: bool) -> Result<()> {
    let log_level = if verbose { Level::DEBUG } else { Level::INFO };
    
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(log_level.to_string()));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true).with_thread_ids(true))
        .with(filter)
        .init();

    Ok(())
}

/// Run the workload for the specified duration
async fn run_workload(
    scheduler: &mut Scheduler,
    workload: &mut WorkloadGenerator,
    duration: Duration,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + duration;
    let mut operations_submitted = 0u64;

    while tokio::time::Instant::now() < deadline {
        // Generate next batch of operations
        let operations = workload.generate_batch(100)?;
        
        for op in operations {
            if !scheduler.is_running() {
                break;
            }
            
            scheduler.submit(op).await?;
            operations_submitted += 1;
        }

        // Yield to allow processing
        tokio::task::yield_now().await;
    }

    // Wait for pending operations to complete
    scheduler.drain_queues(Duration::from_secs(5)).await?;

    info!("Submitted {} operations during workload execution", operations_submitted);
    Ok(())
}
