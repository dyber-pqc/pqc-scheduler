//! # PQC Scheduler
//!
//! Hybrid Classical-Quantum Cryptographic Operation Scheduler with Dynamic
//! Algorithm Selection and Multi-Objective Resource Optimization.
//!
//! This crate provides a comprehensive framework for managing cryptographic
//! operations during the transition from classical to post-quantum cryptography.
//!
//! ## Features
//!
//! - **Dynamic Algorithm Selection**: MDP-based optimal algorithm selection
//! - **Multi-Objective Optimization**: Pareto-optimal resource allocation
//! - **Graceful Degradation**: Formal availability guarantees with sub-ms failover
//! - **Migration Management**: Risk-quantified transition strategies
//!
//! ## Example
//!
//! ```rust,no_run
//! use pqc_scheduler::{Scheduler, Config, Operation, Algorithm, Priority};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Load configuration
//!     let config = Config::from_file("config/default.yaml")?;
//!     
//!     // Create scheduler
//!     let mut scheduler = Scheduler::new(config, 42).await?;
//!     
//!     // Submit operations
//!     let result = scheduler.submit(Operation::KeyExchange {
//!         algorithm: Some(Algorithm::MlKem768),
//!         priority: Priority::High,
//!         data_sensitivity: DataSensitivity::Medium,
//!     }).await?;
//!     
//!     println!("Operation completed in {} μs", result.latency_us);
//!     
//!     scheduler.shutdown().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The scheduler consists of four main components:
//!
//! 1. **MDP Algorithm Selector** (`mdp` module): Uses Markov Decision Process
//!    modeling for optimal algorithm selection under uncertainty.
//!
//! 2. **Multi-Objective Optimizer** (`optimizer` module): Achieves Pareto-optimal
//!    tradeoffs between security, latency, and resource utilization.
//!
//! 3. **Graceful Degradation Controller** (`degradation` module): Manages fallback
//!    chains and ensures service continuity during failures.
//!
//! 4. **Migration Path Manager** (`migration` module): Handles phased transitions
//!    with automatic rollback capabilities.
//!
//! Copyright (c) 2025 Dyber, Inc. All Rights Reserved.

#![doc(html_root_url = "https://docs.rs/pqc-scheduler/0.1.0")]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod config;
pub mod crypto;
pub mod degradation;
pub mod mdp;
pub mod metrics;
pub mod migration;
pub mod optimizer;
pub mod scheduler;
pub mod workload;

// Re-export commonly used types
pub use config::Config;
pub use crypto::{Algorithm, AlgorithmClass, Operation, OperationType};
pub use degradation::FallbackChain;
pub use mdp::{MdpPolicy, State, Action};
pub use metrics::{Metrics, SchedulerResults};
pub use migration::{MigrationPhase, MigrationStrategy};
pub use optimizer::{ObjectiveWeights, ParetoFrontier};
pub use scheduler::{Scheduler, SchedulerHandle};
pub use workload::WorkloadGenerator;
pub use serde::{Deserialize, Serialize};

/// Priority levels for cryptographic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Priority {
    /// Real-time operations with guaranteed latency
    Realtime,
    /// High priority operations
    High,
    /// Normal priority (default)
    Normal,
    /// Background operations
    Background,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

/// Data sensitivity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DataSensitivity {
    /// Low sensitivity data
    Low,
    /// Medium sensitivity data
    Medium,
    /// High sensitivity data
    High,
    /// Critical sensitivity data
    Critical,
}

impl Default for DataSensitivity {
    fn default() -> Self {
        DataSensitivity::Medium
    }
}

/// Compliance framework requirements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ComplianceRequirements {
    #[serde(default)]
    pub fips_140_3: bool,
    #[serde(default)]
    pub pci_dss: bool,
    #[serde(default)]
    pub hipaa: bool,
    #[serde(default)]
    pub fedramp: bool,
}

/// Result of a cryptographic operation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OperationResult {
    /// Unique operation identifier
    pub id: uuid::Uuid,
    /// Algorithm used
    pub algorithm: Algorithm,
    /// Operation type
    pub operation_type: OperationType,
    /// Latency in microseconds
    pub latency_us: f64,
    /// Whether the operation succeeded
    pub success: bool,
    /// Security level achieved (bits)
    pub security_bits: u32,
    /// Error message if failed
    pub error: Option<String>,
}

/// Error types for the scheduler
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Algorithm not available
    #[error("Algorithm not available: {0}")]
    AlgorithmUnavailable(String),
    
    /// Resource exhaustion
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    /// Operation timeout
    #[error("Operation timed out after {0} ms")]
    Timeout(u64),
    
    /// Compliance violation
    #[error("Compliance violation: {0}")]
    ComplianceViolation(String),
    
    /// Cryptographic error
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        Algorithm, AlgorithmClass, Config, DataSensitivity, Operation, OperationType,
        OperationResult, Priority, Scheduler, SchedulerError, ComplianceRequirements,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn test_data_sensitivity_default() {
        assert_eq!(DataSensitivity::default(), DataSensitivity::Medium);
    }
}
