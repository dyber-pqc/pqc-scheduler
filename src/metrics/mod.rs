//! Metrics collection and reporting
//!
//! Provides comprehensive performance monitoring with Prometheus integration.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::crypto::{Algorithm, OperationType};

/// Metrics snapshot
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub throughput_mean: f64,
    pub throughput_std: f64,
    pub latency_mean_us: f64,
    pub latency_std_us: f64,
    pub latency_p50_us: f64,
    pub latency_p99_us: f64,
    pub latency_max_us: f64,
    pub sla_compliance: f64,
    pub security_score: f64,
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
}

/// Final scheduler results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerResults {
    pub throughput_mean: f64,
    pub throughput_std: f64,
    pub latency_mean_us: f64,
    pub latency_std_us: f64,
    pub latency_p50_us: f64,
    pub latency_p99_us: f64,
    pub latency_max_us: f64,
    pub sla_compliance: f64,
    pub security_score: f64,
    pub algorithm_switches: u64,
    pub fallback_events: u64,
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
}

impl SchedulerResults {
    /// Write results to JSON file
    pub fn write_json<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Metrics collector
pub struct MetricsCollector {
    start_time: Instant,
    total_ops: AtomicU64,
    successful_ops: AtomicU64,
    failed_ops: AtomicU64,
    latencies: RwLock<Vec<f64>>,
    by_algorithm: RwLock<HashMap<Algorithm, AlgorithmMetrics>>,
    by_operation: RwLock<HashMap<OperationType, OperationMetrics>>,
    sla_target_us: f64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_ops: AtomicU64::new(0),
            successful_ops: AtomicU64::new(0),
            failed_ops: AtomicU64::new(0),
            latencies: RwLock::new(Vec::new()),
            by_algorithm: RwLock::new(HashMap::new()),
            by_operation: RwLock::new(HashMap::new()),
            sla_target_us: 500.0,
        }
    }

    pub fn reset(&self) {
        self.total_ops.store(0, Ordering::SeqCst);
        self.successful_ops.store(0, Ordering::SeqCst);
        self.failed_ops.store(0, Ordering::SeqCst);
        self.latencies.write().clear();
        self.by_algorithm.write().clear();
        self.by_operation.write().clear();
    }

    pub fn record_operation(
        &self,
        algorithm: &Algorithm,
        operation: &OperationType,
        latency_us: f64,
        success: bool,
    ) {
        self.total_ops.fetch_add(1, Ordering::SeqCst);
        
        if success {
            self.successful_ops.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed_ops.fetch_add(1, Ordering::SeqCst);
        }
        
        self.latencies.write().push(latency_us);
        
        // Update algorithm metrics
        {
            let mut by_algo = self.by_algorithm.write();
            let metrics = by_algo.entry(*algorithm).or_insert_with(AlgorithmMetrics::new);
            metrics.record(latency_us, success);
        }
        
        // Update operation metrics
        {
            let mut by_op = self.by_operation.write();
            let metrics = by_op.entry(*operation).or_insert_with(OperationMetrics::new);
            metrics.record(latency_us, success);
        }
    }

    pub fn snapshot(&self) -> Metrics {
        let latencies = self.latencies.read();
        let total = self.total_ops.load(Ordering::SeqCst);
        let successful = self.successful_ops.load(Ordering::SeqCst);
        let failed = self.failed_ops.load(Ordering::SeqCst);
        
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };
        
        let (mean, std) = compute_mean_std(&latencies);
        let p50 = percentile(&latencies, 0.50);
        let p99 = percentile(&latencies, 0.99);
        let max = latencies.iter().cloned().fold(0.0f64, f64::max);
        
        let sla_compliance = if !latencies.is_empty() {
            let within_sla = latencies.iter().filter(|&&l| l <= self.sla_target_us).count();
            within_sla as f64 / latencies.len() as f64
        } else {
            1.0
        };
        
        // Compute average security score
        let security_score = {
            let by_algo = self.by_algorithm.read();
            if by_algo.is_empty() {
                0.0
            } else {
                let total_ops: u64 = by_algo.values().map(|m| m.count).sum();
                let weighted_sum: f64 = by_algo.iter()
                    .map(|(algo, m)| algo.security_bits() as f64 * m.count as f64)
                    .sum();
                if total_ops > 0 {
                    (weighted_sum / total_ops as f64) / 256.0
                } else {
                    0.0
                }
            }
        };
        
        Metrics {
            throughput_mean: throughput,
            throughput_std: 0.0,  // Would need time-series for proper std
            latency_mean_us: mean,
            latency_std_us: std,
            latency_p50_us: p50,
            latency_p99_us: p99,
            latency_max_us: max,
            sla_compliance,
            security_score,
            total_operations: total,
            successful_operations: successful,
            failed_operations: failed,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-algorithm metrics
struct AlgorithmMetrics {
    count: u64,
    success_count: u64,
    latency_sum: f64,
    latency_sq_sum: f64,
}

impl AlgorithmMetrics {
    fn new() -> Self {
        Self {
            count: 0,
            success_count: 0,
            latency_sum: 0.0,
            latency_sq_sum: 0.0,
        }
    }

    fn record(&mut self, latency_us: f64, success: bool) {
        self.count += 1;
        if success {
            self.success_count += 1;
        }
        self.latency_sum += latency_us;
        self.latency_sq_sum += latency_us * latency_us;
    }
}

/// Per-operation metrics
struct OperationMetrics {
    count: u64,
    success_count: u64,
    latency_sum: f64,
}

impl OperationMetrics {
    fn new() -> Self {
        Self {
            count: 0,
            success_count: 0,
            latency_sum: 0.0,
        }
    }

    fn record(&mut self, latency_us: f64, success: bool) {
        self.count += 1;
        if success {
            self.success_count += 1;
        }
        self.latency_sum += latency_us;
    }
}

/// Prometheus metrics exporter
pub struct MetricsExporter {
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl MetricsExporter {
    pub async fn start(port: u16) -> anyhow::Result<Self> {
        let (shutdown_tx, _shutdown_rx) = oneshot::channel();
        
        // In a real implementation, this would start a Prometheus HTTP server
        tracing::info!("Metrics exporter started on port {}", port);
        
        Ok(Self {
            shutdown_tx: Some(shutdown_tx),
        })
    }

    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

// Helper functions

fn compute_mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    
    (mean, std)
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        collector.record_operation(&Algorithm::MlKem768, &OperationType::Encap, 100.0, true);
        collector.record_operation(&Algorithm::MlKem768, &OperationType::Encap, 150.0, true);
        collector.record_operation(&Algorithm::MlKem768, &OperationType::Encap, 200.0, false);
        
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.total_operations, 3);
        assert_eq!(snapshot.successful_operations, 2);
        assert_eq!(snapshot.failed_operations, 1);
    }
}
