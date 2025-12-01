//! Core scheduler implementation
//!
//! This module provides the main `Scheduler` struct that orchestrates all
//! cryptographic operations, integrating the MDP algorithm selector,
//! multi-objective optimizer, graceful degradation controller, and
//! migration path manager.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

use crate::config::Config;
use crate::crypto::{Algorithm, CryptoBackend, Operation, OperationType};
use crate::degradation::{DegradationController, FallbackChain};
use crate::mdp::{MdpPolicy, MdpSolver, State, Action};
use crate::metrics::{Metrics, MetricsCollector, SchedulerResults};
use crate::migration::{MigrationManager, MigrationPhase};
use crate::optimizer::{MultiObjectiveOptimizer, ObjectiveWeights};
use crate::{DataSensitivity, OperationResult, Priority, SchedulerError};

/// Maximum number of concurrent operations
const MAX_CONCURRENT_OPS: usize = 10000;

/// Queue capacity for operation submissions
const QUEUE_CAPACITY: usize = 8192;

/// Handle for controlling the scheduler from external code
#[derive(Clone)]
pub struct SchedulerHandle {
    shutdown_tx: Arc<RwLock<Option<oneshot::Sender<()>>>>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl SchedulerHandle {
    /// Signal the scheduler to shutdown
    pub fn shutdown(&self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        if let Some(tx) = self.shutdown_tx.write().take() {
            let _ = tx.send(());
        }
    }

    /// Check if scheduler is still running
    pub fn is_running(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Operation submission with response channel
struct OperationSubmission {
    operation: Operation,
    response_tx: oneshot::Sender<Result<OperationResult, SchedulerError>>,
}

/// Main scheduler struct
pub struct Scheduler {
    /// Configuration
    config: Config,
    
    /// Random seed for reproducibility
    seed: u64,
    
    /// MDP policy for algorithm selection
    mdp_policy: Arc<RwLock<MdpPolicy>>,
    
    /// MDP solver for policy computation
    mdp_solver: MdpSolver,
    
    /// Multi-objective optimizer
    optimizer: MultiObjectiveOptimizer,
    
    /// Graceful degradation controller
    degradation_controller: DegradationController,
    
    /// Migration manager
    migration_manager: MigrationManager,
    
    /// Cryptographic backend
    crypto_backend: Arc<CryptoBackend>,
    
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    
    /// Current state
    current_state: Arc<RwLock<State>>,
    
    /// Pending operations
    pending_ops: Arc<DashMap<Uuid, Instant>>,
    
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    
    /// Operation submission channel
    submission_tx: mpsc::Sender<OperationSubmission>,
    
    /// Submission receiver (for internal processing)
    submission_rx: Option<mpsc::Receiver<OperationSubmission>>,
    
    /// Shutdown channel
    shutdown_rx: Option<oneshot::Receiver<()>>,
    
    /// Scheduler handle
    handle: SchedulerHandle,
    
    /// Warmup mode flag
    warmup_mode: bool,
    
    /// Algorithm switch counter
    algorithm_switches: Arc<std::sync::atomic::AtomicU64>,
    
    /// Fallback event counter
    fallback_events: Arc<std::sync::atomic::AtomicU64>,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration
    pub async fn new(config: Config, seed: u64) -> Result<Self> {
        info!("Initializing scheduler with seed {}", seed);
        
        // Initialize cryptographic backend
        let crypto_backend = Arc::new(
            CryptoBackend::new(&config.algorithms, &config.acceleration)
                .context("Failed to initialize crypto backend")?
        );
        
        // Initialize MDP solver
        let mdp_solver = MdpSolver::new(
            config.scheduler.mdp.gamma,
            config.scheduler.mdp.epsilon,
            config.scheduler.mdp.max_iterations,
        );
        
        // Compute initial policy
        info!("Computing initial MDP policy...");
        let initial_state = State::new(
            config.threat.model.initial_threat_level,
            50000.0,  // Initial workload estimate
            vec![0.0; config.algorithms.enabled_count()],
            config.compliance.clone(),
            config.migration.current_phase.clone(),
        );
        
        let mdp_policy = mdp_solver.compute_policy(&initial_state, &config)?;
        info!("Initial policy computed in {} iterations", mdp_solver.last_iteration_count());
        
        // Initialize optimizer
        let optimizer = MultiObjectiveOptimizer::new(
            ObjectiveWeights {
                security: config.scheduler.weights.security,
                latency: config.scheduler.weights.latency,
                cost: config.scheduler.weights.cost,
                throughput: config.scheduler.weights.throughput,
            },
        );
        
        // Initialize degradation controller
        let degradation_controller = DegradationController::new(
            FallbackChain::from_config(&config.degradation.fallback_chain),
            config.degradation.warm_standby.clone(),
            config.degradation.thresholds.clone(),
        );
        
        // Initialize migration manager
        let migration_manager = MigrationManager::new(
            config.migration.current_phase.clone(),
            config.migration.rollback.clone(),
            config.migration.dual_key.clone(),
        );
        
        // Initialize metrics collector
        let metrics = Arc::new(MetricsCollector::new());
        
        // Create channels
        let (submission_tx, submission_rx) = mpsc::channel(QUEUE_CAPACITY);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        
        // Create handle
        let handle = SchedulerHandle {
            shutdown_tx: Arc::new(RwLock::new(Some(shutdown_tx))),
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };
        
        Ok(Self {
            config,
            seed,
            mdp_policy: Arc::new(RwLock::new(mdp_policy)),
            mdp_solver,
            optimizer,
            degradation_controller,
            migration_manager,
            crypto_backend,
            metrics,
            current_state: Arc::new(RwLock::new(initial_state)),
            pending_ops: Arc::new(DashMap::new()),
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_OPS)),
            submission_tx,
            submission_rx: Some(submission_rx),
            shutdown_rx: Some(shutdown_rx),
            handle,
            warmup_mode: false,
            algorithm_switches: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            fallback_events: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Get a handle for external control
    pub fn handle(&self) -> SchedulerHandle {
        self.handle.clone()
    }

    /// Check if the scheduler is running
    pub fn is_running(&self) -> bool {
        self.handle.is_running()
    }

    /// Get the number of enabled algorithms
    pub fn algorithm_count(&self) -> usize {
        self.config.algorithms.enabled_count()
    }

    /// Set warmup mode
    pub fn set_warmup_mode(&mut self, warmup: bool) {
        self.warmup_mode = warmup;
    }

    /// Reset metrics (after warmup)
    pub fn reset_metrics(&self) {
        self.metrics.reset();
        self.algorithm_switches.store(0, std::sync::atomic::Ordering::SeqCst);
        self.fallback_events.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Submit an operation for processing
    pub async fn submit(&self, operation: Operation) -> Result<OperationResult, SchedulerError> {
        // Acquire semaphore permit
        let _permit = self.semaphore.acquire().await
            .map_err(|_| SchedulerError::ResourceExhausted("Semaphore closed".into()))?;
        
        // Create response channel
        let (response_tx, response_rx) = oneshot::channel();
        
        // Submit operation
        let submission = OperationSubmission { operation, response_tx };
        
        self.submission_tx.send(submission).await
            .map_err(|_| SchedulerError::Internal("Submission channel closed".into()))?;
        
        // Wait for result
        response_rx.await
            .map_err(|_| SchedulerError::Internal("Response channel dropped".into()))?
    }

    /// Process a single operation (internal)
    async fn process_operation(&self, operation: Operation) -> Result<OperationResult, SchedulerError> {
        let op_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        // Record pending operation
        self.pending_ops.insert(op_id, start_time);
        
        // Get current state
        let current_state = self.current_state.read().clone();
        
        // Select algorithm using MDP policy
        let policy = self.mdp_policy.read();
        let action = policy.get_action(&current_state)
            .ok_or_else(|| SchedulerError::Internal("No action found for state".into()))?;
        
        // Determine algorithm to use
        let algorithm = self.select_algorithm(&operation, &action)?;
        
        // Check for algorithm availability
        let algorithm = match self.degradation_controller.check_availability(&algorithm) {
            Ok(alg) => alg,
            Err(fallback) => {
                self.fallback_events.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                debug!("Using fallback algorithm: {:?}", fallback);
                fallback
            }
        };
        
        // Execute cryptographic operation
        let result = self.crypto_backend.execute(&algorithm, &operation).await;
        
        // Remove from pending
        self.pending_ops.remove(&op_id);
        
        // Calculate latency
        let latency_us = start_time.elapsed().as_micros() as f64;
        
        // Record metrics (unless in warmup)
        if !self.warmup_mode {
            self.metrics.record_operation(
                &algorithm,
                &operation.operation_type(),
                latency_us,
                result.is_ok(),
            );
        }
        
        // Build result
        match result {
            Ok(_) => Ok(OperationResult {
                id: op_id,
                algorithm,
                operation_type: operation.operation_type(),
                latency_us,
                success: true,
                security_bits: algorithm.security_bits(),
                error: None,
            }),
            Err(e) => Ok(OperationResult {
                id: op_id,
                algorithm,
                operation_type: operation.operation_type(),
                latency_us,
                success: false,
                security_bits: 0,
                error: Some(e.to_string()),
            }),
        }
    }

    /// Select algorithm based on operation and MDP action
    fn select_algorithm(
        &self,
        operation: &Operation,
        action: &Action,
    ) -> Result<Algorithm, SchedulerError> {
        // If operation specifies an algorithm, use it (if allowed)
        if let Some(requested) = operation.requested_algorithm() {
            if self.is_algorithm_allowed(&requested) {
                return Ok(requested);
            }
        }
        
        // Use MDP-selected algorithm
        let algorithm = match operation.operation_type() {
            OperationType::KeyGen | OperationType::Encap | OperationType::Decap => {
                action.kem_algorithm.clone()
            }
            OperationType::Sign | OperationType::Verify => {
                action.signature_algorithm.clone()
            }
        };
        
        // Verify algorithm is allowed
        if !self.is_algorithm_allowed(&algorithm) {
            return Err(SchedulerError::AlgorithmUnavailable(
                format!("{:?} not enabled", algorithm)
            ));
        }
        
        Ok(algorithm)
    }

    /// Check if an algorithm is allowed by configuration
    fn is_algorithm_allowed(&self, algorithm: &Algorithm) -> bool {
        self.config.algorithms.is_enabled(algorithm)
    }

    /// Drain pending operations
    pub async fn drain_queues(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        
        while !self.pending_ops.is_empty() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        if !self.pending_ops.is_empty() {
            warn!("{} operations still pending after drain timeout", self.pending_ops.len());
        }
        
        Ok(())
    }

    /// Collect final results
    pub fn collect_results(&self) -> SchedulerResults {
        let metrics = self.metrics.snapshot();
        
        SchedulerResults {
            throughput_mean: metrics.throughput_mean,
            throughput_std: metrics.throughput_std,
            latency_mean_us: metrics.latency_mean_us,
            latency_std_us: metrics.latency_std_us,
            latency_p50_us: metrics.latency_p50_us,
            latency_p99_us: metrics.latency_p99_us,
            latency_max_us: metrics.latency_max_us,
            sla_compliance: metrics.sla_compliance,
            security_score: metrics.security_score,
            algorithm_switches: self.algorithm_switches.load(std::sync::atomic::Ordering::SeqCst),
            fallback_events: self.fallback_events.load(std::sync::atomic::Ordering::SeqCst),
            total_operations: metrics.total_operations,
            successful_operations: metrics.successful_operations,
            failed_operations: metrics.failed_operations,
        }
    }

    /// Shutdown the scheduler gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down scheduler...");
        
        // Signal shutdown
        self.handle.shutdown();
        
        // Wait for pending operations
        self.drain_queues(Duration::from_secs(5)).await?;
        
        // Shutdown crypto backend
        self.crypto_backend.shutdown().await?;
        
        info!("Scheduler shutdown complete");
        Ok(())
    }
}

/// Module re-exports
pub mod state {
    pub use crate::mdp::State;
}

pub mod action {
    pub use crate::mdp::Action;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_scheduler_creation() {
        let config = Config::default();
        let scheduler = Scheduler::new(config, 42).await;
        assert!(scheduler.is_ok());
    }
}
