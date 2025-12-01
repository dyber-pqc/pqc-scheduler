//! Workload generation
//!
//! Provides synthetic workload generation based on configurable patterns
//! including MMPP (Markov-Modulated Poisson Process) for realistic traffic.

use std::path::Path;

use anyhow::{Context, Result};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::crypto::{Algorithm, Operation};
use crate::{DataSensitivity, Priority};

/// Workload generator
pub struct WorkloadGenerator {
    config: WorkloadConfig,
    rng: ChaCha8Rng,
    current_state: usize,
}

impl WorkloadGenerator {
    /// Load workload from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read workload file: {:?}", path.as_ref()))?;
        
        let config: WorkloadConfig = serde_json::from_str(&content)
            .context("Failed to parse workload JSON")?;
        
        Ok(Self {
            config,
            rng: ChaCha8Rng::seed_from_u64(42),
            current_state: 0,
        })
    }

    /// Get workload name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get workload description
    pub fn description(&self) -> &str {
        &self.config.description
    }

    /// Set random seed
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Generate a batch of operations
    pub fn generate_batch(&mut self, count: usize) -> Result<Vec<Operation>> {
        let mut operations = Vec::with_capacity(count);
        
        for _ in 0..count {
            let op = self.generate_single()?;
            operations.push(op);
        }
        
        Ok(operations)
    }

    /// Generate a single operation
    fn generate_single(&mut self) -> Result<Operation> {
        // Determine operation type (KEM vs signature)
        let is_kem = self.rng.gen::<f64>() < self.config.algorithm_mix.kem;
        
        // Determine priority
        let priority = self.sample_priority();
        
        // Determine data sensitivity
        let sensitivity = self.sample_sensitivity();
        
        if is_kem {
            // Sample KEM operation type
            let op_roll = self.rng.gen::<f64>();
            let kem_ops = &self.config.operation_mix.kem_operations;
            
            if op_roll < kem_ops.keygen {
                Ok(Operation::KeyExchange {
                    algorithm: None,
                    priority,
                    data_sensitivity: sensitivity,
                })
            } else if op_roll < kem_ops.keygen + kem_ops.encap {
                Ok(Operation::Encapsulation {
                    algorithm: None,
                    public_key: vec![0u8; 1184],  // ML-KEM-768 public key size
                    priority,
                })
            } else {
                Ok(Operation::Decapsulation {
                    algorithm: None,
                    ciphertext: vec![0u8; 1088],  // ML-KEM-768 ciphertext size
                    secret_key: vec![0u8; 2400],
                    priority,
                })
            }
        } else {
            // Sample signature operation type
            let op_roll = self.rng.gen::<f64>();
            let sig_ops = &self.config.operation_mix.signature_operations;
            
            if op_roll < sig_ops.keygen {
                Ok(Operation::KeyExchange {
                    algorithm: None,
                    priority,
                    data_sensitivity: sensitivity,
                })
            } else if op_roll < sig_ops.keygen + sig_ops.sign {
                Ok(Operation::Sign {
                    algorithm: None,
                    message: vec![0u8; 256],  // Sample message
                    secret_key: vec![0u8; 4000],
                    priority,
                })
            } else {
                Ok(Operation::Verify {
                    algorithm: None,
                    message: vec![0u8; 256],
                    signature: vec![0u8; 3293],  // ML-DSA-65 signature size
                    public_key: vec![0u8; 1952],
                    priority,
                })
            }
        }
    }

    fn sample_priority(&mut self) -> Priority {
        let roll = self.rng.gen::<f64>();
        let dist = &self.config.priority_distribution;
        
        if roll < dist.realtime {
            Priority::Realtime
        } else if roll < dist.realtime + dist.high {
            Priority::High
        } else if roll < dist.realtime + dist.high + dist.normal {
            Priority::Normal
        } else {
            Priority::Background
        }
    }

    fn sample_sensitivity(&mut self) -> DataSensitivity {
        let roll = self.rng.gen::<f64>();
        let dist = &self.config.data_sensitivity_distribution;
        
        if roll < dist.low {
            DataSensitivity::Low
        } else if roll < dist.low + dist.medium {
            DataSensitivity::Medium
        } else if roll < dist.low + dist.medium + dist.high {
            DataSensitivity::High
        } else {
            DataSensitivity::Critical
        }
    }
}

/// Workload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub version: String,
    pub arrival_model: String,
    #[serde(default)]
    pub mmpp_config: Option<MmppConfig>,
    #[serde(default)]
    pub poisson_config: Option<PoissonConfig>,
    pub algorithm_mix: AlgorithmMix,
    pub operation_mix: OperationMix,
    #[serde(default)]
    pub algorithm_distribution: Option<AlgorithmDistribution>,
    pub priority_distribution: PriorityDistribution,
    pub data_sensitivity_distribution: SensitivityDistribution,
    #[serde(default)]
    pub client_config: Option<ClientConfig>,
    #[serde(default)]
    pub temporal_patterns: Option<TemporalPatterns>,
    #[serde(default)]
    pub burst_config: Option<BurstConfig>,
    #[serde(default)]
    pub threat_evolution: Option<ThreatEvolution>,
    #[serde(default)]
    pub compliance_requirements: Option<crate::ComplianceRequirements>,
    #[serde(default)]
    pub metrics_collection: Option<MetricsCollection>,
}

/// MMPP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmppConfig {
    pub description: String,
    pub states: Vec<MmppState>,
    #[serde(default)]
    pub transition_matrix: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub initial_state: Option<String>,
}

/// MMPP state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmppState {
    pub name: String,
    pub rate: f64,
    pub duration_mean: f64,
    #[serde(default)]
    pub duration_std: Option<f64>,
}

/// Poisson configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonConfig {
    pub rate: f64,
    #[serde(default)]
    pub description: Option<String>,
}

/// Algorithm mix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMix {
    pub kem: f64,
    pub signature: f64,
}

/// Operation mix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMix {
    pub kem_operations: KemOperations,
    pub signature_operations: SignatureOperations,
}

/// KEM operation distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KemOperations {
    pub keygen: f64,
    pub encap: f64,
    pub decap: f64,
}

/// Signature operation distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureOperations {
    pub keygen: f64,
    pub sign: f64,
    pub verify: f64,
}

/// Algorithm distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDistribution {
    #[serde(default)]
    pub kem: std::collections::HashMap<String, f64>,
    #[serde(default)]
    pub signature: std::collections::HashMap<String, f64>,
}

/// Priority distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityDistribution {
    pub realtime: f64,
    pub high: f64,
    pub normal: f64,
    pub background: f64,
}

/// Sensitivity distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityDistribution {
    pub low: f64,
    pub medium: f64,
    pub high: f64,
    pub critical: f64,
}

/// Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    pub concurrent_connections: usize,
    pub connection_duration_mean_sec: f64,
    pub connection_duration_std_sec: f64,
    pub requests_per_connection_mean: usize,
    pub think_time_mean_ms: f64,
    pub think_time_std_ms: f64,
}

/// Temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    pub diurnal_enabled: bool,
    #[serde(default)]
    pub diurnal_peak_hour_utc: Option<u32>,
    #[serde(default)]
    pub diurnal_trough_hour_utc: Option<u32>,
    #[serde(default)]
    pub diurnal_amplitude: Option<f64>,
    pub weekly_enabled: bool,
}

/// Burst configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstConfig {
    pub enabled: bool,
    #[serde(default)]
    pub burst_probability: Option<f64>,
    #[serde(default)]
    pub burst_multiplier_min: Option<f64>,
    #[serde(default)]
    pub burst_multiplier_max: Option<f64>,
    #[serde(default)]
    pub burst_duration_mean_sec: Option<f64>,
    #[serde(default)]
    pub burst_duration_std_sec: Option<f64>,
}

/// Threat evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvolution {
    pub enabled: bool,
    pub initial_threat: f64,
    #[serde(default)]
    pub threat_events: Vec<ThreatEvent>,
}

/// Threat event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvent {
    pub time_offset_sec: u64,
    pub threat_level: f64,
    #[serde(default)]
    pub description: Option<String>,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollection {
    pub sample_interval_ms: u64,
    pub histogram_percentiles: Vec<f64>,
    pub aggregate_by: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_generation() {
        let config = WorkloadConfig {
            name: "test".into(),
            description: "Test workload".into(),
            version: "1.0.0".into(),
            arrival_model: "poisson".into(),
            mmpp_config: None,
            poisson_config: Some(PoissonConfig {
                rate: 1000.0,
                description: None,
            }),
            algorithm_mix: AlgorithmMix { kem: 0.6, signature: 0.4 },
            operation_mix: OperationMix {
                kem_operations: KemOperations { keygen: 0.1, encap: 0.5, decap: 0.4 },
                signature_operations: SignatureOperations { keygen: 0.1, sign: 0.3, verify: 0.6 },
            },
            algorithm_distribution: None,
            priority_distribution: PriorityDistribution {
                realtime: 0.0, high: 0.2, normal: 0.7, background: 0.1,
            },
            data_sensitivity_distribution: SensitivityDistribution {
                low: 0.3, medium: 0.5, high: 0.15, critical: 0.05,
            },
            client_config: None,
            temporal_patterns: None,
            burst_config: None,
            threat_evolution: None,
            compliance_requirements: None,
            metrics_collection: None,
        };
        
        let mut generator = WorkloadGenerator {
            config,
            rng: ChaCha8Rng::seed_from_u64(42),
            current_state: 0,
        };
        
        let batch = generator.generate_batch(100).unwrap();
        assert_eq!(batch.len(), 100);
    }
}
