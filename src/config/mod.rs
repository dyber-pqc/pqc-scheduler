//! Configuration management
//!
//! This module handles loading and validation of scheduler configuration.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::crypto::Algorithm;
use crate::migration::MigrationPhase;
use crate::ComplianceRequirements;

/// Main configuration struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub scheduler: SchedulerConfig,
    pub algorithms: AlgorithmConfig,
    pub degradation: DegradationConfig,
    pub migration: MigrationConfig,
    pub resources: ResourceConfig,
    pub compliance: ComplianceRequirements,
    pub threat: ThreatConfig,
    pub monitoring: MonitoringConfig,
    pub logging: LoggingConfig,
    pub redis: RedisConfig,
    pub acceleration: AccelerationConfig,
}

impl Config {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        let config: Config = serde_yaml::from_str(&content)
            .context("Failed to parse YAML configuration")?;
        
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate MDP parameters
        if self.scheduler.mdp.gamma <= 0.0 || self.scheduler.mdp.gamma >= 1.0 {
            anyhow::bail!("gamma must be in (0, 1)");
        }
        
        if self.scheduler.mdp.epsilon <= 0.0 {
            anyhow::bail!("epsilon must be positive");
        }
        
        // Validate at least one algorithm is enabled
        if self.algorithms.enabled_count() == 0 {
            anyhow::bail!("At least one algorithm must be enabled");
        }
        
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            algorithms: AlgorithmConfig::default(),
            degradation: DegradationConfig::default(),
            migration: MigrationConfig::default(),
            resources: ResourceConfig::default(),
            compliance: ComplianceRequirements::default(),
            threat: ThreatConfig::default(),
            monitoring: MonitoringConfig::default(),
            logging: LoggingConfig::default(),
            redis: RedisConfig::default(),
            acceleration: AccelerationConfig::default(),
        }
    }
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub mdp: MdpConfig,
    pub weights: WeightConfig,
    pub state_space: StateSpaceConfig,
    pub action_space: ActionSpaceConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            mdp: MdpConfig::default(),
            weights: WeightConfig::default(),
            state_space: StateSpaceConfig::default(),
            action_space: ActionSpaceConfig::default(),
        }
    }
}

/// MDP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdpConfig {
    pub gamma: f64,
    pub epsilon: f64,
    pub max_iterations: usize,
    pub recompute_interval_sec: u64,
    pub cache_policy: bool,
}

impl Default for MdpConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            epsilon: 1e-6,
            max_iterations: 2000,
            recompute_interval_sec: 60,
            cache_policy: true,
        }
    }
}

/// Weight configuration for multi-objective optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightConfig {
    pub security: f64,
    pub latency: f64,
    pub cost: f64,
    pub throughput: f64,
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            security: 1.0,
            latency: 0.5,
            cost: 0.3,
            throughput: 0.4,
        }
    }
}

/// State space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceConfig {
    pub threat_bins: usize,
    pub workload_bins: Vec<f64>,
    pub utilization_bins: usize,
}

impl Default for StateSpaceConfig {
    fn default() -> Self {
        Self {
            threat_bins: 11,
            workload_bins: vec![
                1000.0, 5000.0, 10000.0, 25000.0, 50000.0,
                75000.0, 100000.0, 150000.0, 200000.0, 300000.0, 500000.0,
            ],
            utilization_bins: 5,
        }
    }
}

/// Action space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpaceConfig {
    pub algorithm_count: usize,
    pub migration_directions: usize,
    pub allocation_profiles: usize,
}

impl Default for ActionSpaceConfig {
    fn default() -> Self {
        Self {
            algorithm_count: 20,
            migration_directions: 3,
            allocation_profiles: 10,
        }
    }
}

/// Algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub enabled: EnabledAlgorithms,
    #[serde(default)]
    pub performance: std::collections::HashMap<String, AlgorithmPerformance>,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            enabled: EnabledAlgorithms::default(),
            performance: std::collections::HashMap::new(),
        }
    }
}

impl AlgorithmConfig {
    pub fn enabled_count(&self) -> usize {
        self.enabled.classical.len() + 
        self.enabled.post_quantum.len() + 
        self.enabled.hybrid.len()
    }

    pub fn is_enabled(&self, algorithm: &Algorithm) -> bool {
        let name = format!("{:?}", algorithm);
        self.enabled.classical.iter().any(|a| a == &name) ||
        self.enabled.post_quantum.iter().any(|a| a == &name) ||
        self.enabled.hybrid.iter().any(|a| a == &name)
    }

    pub fn all_enabled_algorithms(&self) -> Vec<Algorithm> {
        vec![
            Algorithm::MlKem768,
            Algorithm::MlKem512,
            Algorithm::MlKem1024,
            Algorithm::MlDsa65,
            Algorithm::MlDsa44,
            Algorithm::MlDsa87,
            Algorithm::X25519,
            Algorithm::EcdsaP256,
        ]
    }

    pub fn enabled_kem_algorithms(&self) -> Vec<Algorithm> {
        vec![
            Algorithm::MlKem768,
            Algorithm::MlKem512,
            Algorithm::MlKem1024,
            Algorithm::X25519MlKem768,
            Algorithm::X25519,
        ]
    }

    pub fn enabled_signature_algorithms(&self) -> Vec<Algorithm> {
        vec![
            Algorithm::MlDsa65,
            Algorithm::MlDsa44,
            Algorithm::MlDsa87,
            Algorithm::Ed25519MlDsa65,
            Algorithm::EcdsaP256,
        ]
    }
}

/// Enabled algorithms by category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnabledAlgorithms {
    #[serde(default)]
    pub classical: Vec<String>,
    #[serde(default)]
    pub post_quantum: Vec<String>,
    #[serde(default)]
    pub hybrid: Vec<String>,
}

impl Default for EnabledAlgorithms {
    fn default() -> Self {
        Self {
            classical: vec!["X25519".into(), "ECDSA-P256".into()],
            post_quantum: vec!["ML-KEM-768".into(), "ML-DSA-65".into()],
            hybrid: vec!["X25519+ML-KEM-768".into()],
        }
    }
}

/// Algorithm performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPerformance {
    pub keygen_us: f64,
    pub encap_us: Option<f64>,
    pub decap_us: Option<f64>,
    pub sign_us: Option<f64>,
    pub verify_us: Option<f64>,
    pub security_bits: u32,
    pub public_key_bytes: usize,
    #[serde(default)]
    pub ciphertext_bytes: Option<usize>,
    #[serde(default)]
    pub signature_bytes: Option<usize>,
}

/// Degradation configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegradationConfig {
    pub fallback_chain: FallbackChainConfig,
    pub warm_standby: WarmStandbyConfig,
    pub thresholds: ThresholdConfig,
}

/// Fallback chain configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FallbackChainConfig {
    #[serde(default)]
    pub primary: Vec<String>,
    #[serde(default)]
    pub signature: Vec<String>,
}

/// Warm standby configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmStandbyConfig {
    pub enabled: bool,
    pub precompute_keys: bool,
    pub key_pool_size: usize,
    pub refresh_interval_sec: u64,
}

impl Default for WarmStandbyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            precompute_keys: true,
            key_pool_size: 1000,
            refresh_interval_sec: 3600,
        }
    }
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub latency_max_us: f64,
    pub error_rate_max: f64,
    pub utilization_high: f64,
    pub utilization_critical: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            latency_max_us: 500.0,
            error_rate_max: 0.001,
            utilization_high: 0.85,
            utilization_critical: 0.95,
        }
    }
}

/// Migration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    pub current_phase: MigrationPhase,
    pub phase_duration: PhaseDuration,
    pub rollback: RollbackConfig,
    pub dual_key: DualKeyConfig,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            current_phase: MigrationPhase::Hybrid,
            phase_duration: PhaseDuration::default(),
            rollback: RollbackConfig::default(),
            dual_key: DualKeyConfig::default(),
        }
    }
}

/// Phase duration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDuration {
    pub assessment: u32,
    pub hybrid_deployment: u32,
    pub pqc_primary: u32,
    pub classical_deprecation: u32,
}

impl Default for PhaseDuration {
    fn default() -> Self {
        Self {
            assessment: 4,
            hybrid_deployment: 8,
            pqc_primary: 16,
            classical_deprecation: 52,
        }
    }
}

/// Rollback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    pub enabled: bool,
    pub auto_trigger: bool,
    pub latency_threshold_us: f64,
    pub error_threshold: f64,
    pub cooldown_sec: u64,
    pub max_rollbacks_per_day: u32,
}

impl Default for RollbackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_trigger: true,
            latency_threshold_us: 1000.0,
            error_threshold: 0.01,
            cooldown_sec: 300,
            max_rollbacks_per_day: 5,
        }
    }
}

/// Dual key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualKeyConfig {
    pub enabled: bool,
    pub sync_interval_sec: u64,
}

impl Default for DualKeyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sync_interval_sec: 60,
        }
    }
}

/// Resource configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub cores: CoreConfig,
    pub memory: MemoryConfig,
    pub queues: QueueConfig,
}

/// Core allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreConfig {
    pub classical_cores: usize,
    pub pqc_cores: usize,
    pub hybrid_cores: usize,
    pub ml_cores: usize,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            classical_cores: 4,
            pqc_cores: 8,
            hybrid_cores: 4,
            ml_cores: 2,
        }
    }
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub key_cache_mb: usize,
    pub session_cache_mb: usize,
    pub work_buffer_mb: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            key_cache_mb: 512,
            session_cache_mb: 1024,
            work_buffer_mb: 256,
        }
    }
}

/// Queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    pub submit_queue_depth: usize,
    pub dispatch_queue_depth: usize,
    pub completion_queue_depth: usize,
    pub priority_queue_depth: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            submit_queue_depth: 4096,
            dispatch_queue_depth: 1024,
            completion_queue_depth: 4096,
            priority_queue_depth: 256,
        }
    }
}

/// Threat configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreatConfig {
    pub model: ThreatModelConfig,
    pub quantum_timeline: QuantumTimelineConfig,
    #[serde(default)]
    pub sources: Vec<ThreatSource>,
}

/// Threat model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatModelConfig {
    pub initial_threat_level: f64,
    pub threat_increase_rate: f64,
}

impl Default for ThreatModelConfig {
    fn default() -> Self {
        Self {
            initial_threat_level: 0.1,
            threat_increase_rate: 0.02,
        }
    }
}

/// Quantum timeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTimelineConfig {
    pub scenario: String,
    pub crqc_probability_2030: f64,
    pub crqc_probability_2035: f64,
    pub crqc_probability_2040: f64,
}

impl Default for QuantumTimelineConfig {
    fn default() -> Self {
        Self {
            scenario: "moderate".into(),
            crqc_probability_2030: 0.3,
            crqc_probability_2035: 0.5,
            crqc_probability_2040: 0.7,
        }
    }
}

/// Threat intelligence source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatSource {
    pub name: String,
    pub url: String,
    pub refresh_interval_sec: u64,
}

/// Monitoring configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub prometheus: PrometheusConfig,
    pub metrics: MetricsConfig,
    pub alerts: AlertConfig,
}

/// Prometheus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    pub enabled: bool,
    pub port: u16,
    pub path: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 9090,
            path: "/metrics".into(),
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub collection_interval_ms: u64,
    pub histogram_buckets: Vec<f64>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 1000,
            histogram_buckets: vec![10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0],
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub latency_p99_us: f64,
    pub throughput_min: f64,
    pub error_rate_max: f64,
    pub availability_min: f64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            latency_p99_us: 1000.0,
            throughput_min: 50000.0,
            error_rate_max: 0.001,
            availability_min: 0.9999,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    #[serde(default)]
    pub output: Vec<LogOutput>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "json".into(),
            output: vec![LogOutput::Stdout],
        }
    }
}

/// Log output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LogOutput {
    #[serde(rename = "stdout")]
    Stdout,
    #[serde(rename = "file")]
    File {
        path: String,
        rotation: String,
        retention_days: u32,
    },
}

/// Redis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: usize,
    pub timeout_ms: u64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".into(),
            pool_size: 10,
            timeout_ms: 1000,
        }
    }
}

/// Acceleration configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccelerationConfig {
    pub software: SoftwareAccelConfig,
    pub fpga: FpgaConfig,
}

/// Software acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareAccelConfig {
    pub use_avx2: bool,
    pub use_avx512: bool,
    pub thread_pool_size: usize,
}

impl Default for SoftwareAccelConfig {
    fn default() -> Self {
        Self {
            use_avx2: true,
            use_avx512: false,
            thread_pool_size: 0,
        }
    }
}

/// FPGA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FpgaConfig {
    pub enabled: bool,
    pub device: String,
    pub max_utilization: f64,
    pub dma_buffer_size_mb: usize,
}

impl Default for FpgaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device: "/dev/xdma0".into(),
            max_utilization: 0.8,
            dma_buffer_size_mb: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }
}
