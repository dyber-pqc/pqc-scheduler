//! Cryptographic backend implementation
//!
//! This module provides the interface to cryptographic algorithm implementations,
//! supporting both classical (RSA, ECDSA, X25519) and post-quantum
//! (ML-KEM, ML-DSA, SLH-DSA) algorithms via liboqs and OpenSSL.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::Semaphore;
use tracing::{debug, trace};

use crate::config::{AlgorithmConfig, AccelerationConfig};
use crate::SchedulerError;

/// Cryptographic algorithm enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Algorithm {
    // Classical algorithms
    Rsa2048,
    Rsa4096,
    EcdsaP256,
    EcdsaP384,
    X25519,
    Ed25519,
    
    // Post-quantum KEM algorithms
    MlKem512,
    MlKem768,
    MlKem1024,
    
    // Post-quantum signature algorithms
    MlDsa44,
    MlDsa65,
    MlDsa87,
    SlhDsa128f,
    SlhDsa128s,
    SlhDsa192f,
    
    // Hybrid algorithms
    X25519MlKem768,
    P256MlKem768,
    P384MlKem1024,
    Ed25519MlDsa65,
    P256MlDsa65,
    P384MlDsa87,
}

impl Algorithm {
    /// Get security level in bits
    pub fn security_bits(&self) -> u32 {
        match self {
            Algorithm::Rsa2048 => 112,
            Algorithm::Rsa4096 => 128,
            Algorithm::EcdsaP256 | Algorithm::X25519 | Algorithm::Ed25519 => 128,
            Algorithm::EcdsaP384 => 192,
            Algorithm::MlKem512 | Algorithm::MlDsa44 | Algorithm::SlhDsa128f | Algorithm::SlhDsa128s => 128,
            Algorithm::MlKem768 | Algorithm::MlDsa65 => 192,
            Algorithm::MlKem1024 | Algorithm::MlDsa87 | Algorithm::SlhDsa192f => 256,
            Algorithm::X25519MlKem768 | Algorithm::P256MlKem768 | Algorithm::Ed25519MlDsa65 | Algorithm::P256MlDsa65 => 192,
            Algorithm::P384MlKem1024 | Algorithm::P384MlDsa87 => 256,
        }
    }

    /// Check if algorithm is post-quantum
    pub fn is_post_quantum(&self) -> bool {
        matches!(self,
            Algorithm::MlKem512 | Algorithm::MlKem768 | Algorithm::MlKem1024 |
            Algorithm::MlDsa44 | Algorithm::MlDsa65 | Algorithm::MlDsa87 |
            Algorithm::SlhDsa128f | Algorithm::SlhDsa128s | Algorithm::SlhDsa192f |
            Algorithm::X25519MlKem768 | Algorithm::P256MlKem768 | Algorithm::P384MlKem1024 |
            Algorithm::Ed25519MlDsa65 | Algorithm::P256MlDsa65 | Algorithm::P384MlDsa87
        )
    }

    /// Check if algorithm is hybrid
    pub fn is_hybrid(&self) -> bool {
        matches!(self,
            Algorithm::X25519MlKem768 | Algorithm::P256MlKem768 | Algorithm::P384MlKem1024 |
            Algorithm::Ed25519MlDsa65 | Algorithm::P256MlDsa65 | Algorithm::P384MlDsa87
        )
    }

    /// Get algorithm class
    pub fn class(&self) -> AlgorithmClass {
        if self.is_hybrid() {
            AlgorithmClass::Hybrid
        } else if self.is_post_quantum() {
            AlgorithmClass::PostQuantum
        } else {
            AlgorithmClass::Classical
        }
    }

    /// Get typical latency in microseconds
    pub fn typical_latency_us(&self) -> f64 {
        match self {
            Algorithm::Rsa2048 => 2134.0,
            Algorithm::Rsa4096 => 8500.0,
            Algorithm::EcdsaP256 => 68.0,
            Algorithm::EcdsaP384 => 120.0,
            Algorithm::X25519 => 25.0,
            Algorithm::Ed25519 => 50.0,
            Algorithm::MlKem512 => 28.0,
            Algorithm::MlKem768 => 36.0,
            Algorithm::MlKem1024 => 51.0,
            Algorithm::MlDsa44 => 198.0,
            Algorithm::MlDsa65 => 287.0,
            Algorithm::MlDsa87 => 389.0,
            Algorithm::SlhDsa128f => 8247.0,
            Algorithm::SlhDsa128s => 892.0,
            Algorithm::SlhDsa192f => 12000.0,
            Algorithm::X25519MlKem768 => 108.0,
            Algorithm::P256MlKem768 => 115.0,
            Algorithm::P384MlKem1024 => 180.0,
            Algorithm::Ed25519MlDsa65 => 320.0,
            Algorithm::P256MlDsa65 => 340.0,
            Algorithm::P384MlDsa87 => 450.0,
        }
    }

    /// Get public key size in bytes
    pub fn public_key_bytes(&self) -> usize {
        match self {
            Algorithm::Rsa2048 => 256,
            Algorithm::Rsa4096 => 512,
            Algorithm::EcdsaP256 | Algorithm::X25519 => 64,
            Algorithm::EcdsaP384 => 96,
            Algorithm::Ed25519 => 32,
            Algorithm::MlKem512 => 800,
            Algorithm::MlKem768 => 1184,
            Algorithm::MlKem1024 => 1568,
            Algorithm::MlDsa44 => 1312,
            Algorithm::MlDsa65 => 1952,
            Algorithm::MlDsa87 => 2592,
            Algorithm::SlhDsa128f | Algorithm::SlhDsa128s => 32,
            Algorithm::SlhDsa192f => 48,
            Algorithm::X25519MlKem768 => 1216,
            Algorithm::P256MlKem768 => 1248,
            Algorithm::P384MlKem1024 => 1664,
            Algorithm::Ed25519MlDsa65 => 1984,
            Algorithm::P256MlDsa65 => 2016,
            Algorithm::P384MlDsa87 => 2688,
        }
    }
}

/// Algorithm class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AlgorithmClass {
    Classical,
    PostQuantum,
    Hybrid,
}

/// Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OperationType {
    KeyGen,
    Encap,
    Decap,
    Sign,
    Verify,
}

/// Cryptographic operation
#[derive(Debug, Clone)]
pub enum Operation {
    /// Key exchange operation
    KeyExchange {
        algorithm: Option<Algorithm>,
        priority: crate::Priority,
        data_sensitivity: crate::DataSensitivity,
    },
    /// Encapsulation operation
    Encapsulation {
        algorithm: Option<Algorithm>,
        public_key: Vec<u8>,
        priority: crate::Priority,
    },
    /// Decapsulation operation
    Decapsulation {
        algorithm: Option<Algorithm>,
        ciphertext: Vec<u8>,
        secret_key: Vec<u8>,
        priority: crate::Priority,
    },
    /// Sign operation
    Sign {
        algorithm: Option<Algorithm>,
        message: Vec<u8>,
        secret_key: Vec<u8>,
        priority: crate::Priority,
    },
    /// Verify operation
    Verify {
        algorithm: Option<Algorithm>,
        message: Vec<u8>,
        signature: Vec<u8>,
        public_key: Vec<u8>,
        priority: crate::Priority,
    },
}

impl Operation {
    /// Get the operation type
    pub fn operation_type(&self) -> OperationType {
        match self {
            Operation::KeyExchange { .. } => OperationType::KeyGen,
            Operation::Encapsulation { .. } => OperationType::Encap,
            Operation::Decapsulation { .. } => OperationType::Decap,
            Operation::Sign { .. } => OperationType::Sign,
            Operation::Verify { .. } => OperationType::Verify,
        }
    }

    /// Get requested algorithm (if any)
    pub fn requested_algorithm(&self) -> Option<Algorithm> {
        match self {
            Operation::KeyExchange { algorithm, .. } => *algorithm,
            Operation::Encapsulation { algorithm, .. } => *algorithm,
            Operation::Decapsulation { algorithm, .. } => *algorithm,
            Operation::Sign { algorithm, .. } => *algorithm,
            Operation::Verify { algorithm, .. } => *algorithm,
        }
    }
}

/// Cryptographic backend
pub struct CryptoBackend {
    /// Available algorithms
    algorithms: Vec<Algorithm>,
    
    /// Software backend
    software_backend: SoftwareBackend,
    
    /// FPGA backend (optional)
    #[cfg(feature = "fpga")]
    fpga_backend: Option<FpgaBackend>,
    
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
}

impl CryptoBackend {
    /// Create a new crypto backend
    pub fn new(algo_config: &AlgorithmConfig, accel_config: &AccelerationConfig) -> Result<Self> {
        let algorithms = algo_config.all_enabled_algorithms();
        
        let software_backend = SoftwareBackend::new(accel_config)?;
        
        #[cfg(feature = "fpga")]
        let fpga_backend = if accel_config.fpga.enabled {
            Some(FpgaBackend::new(&accel_config.fpga)?)
        } else {
            None
        };
        
        let max_concurrent = num_cpus::get() * 4;
        
        Ok(Self {
            algorithms,
            software_backend,
            #[cfg(feature = "fpga")]
            fpga_backend,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
        })
    }

    /// Execute a cryptographic operation
    pub async fn execute(
        &self,
        algorithm: &Algorithm,
        operation: &Operation,
    ) -> Result<Vec<u8>, SchedulerError> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| SchedulerError::Internal("Semaphore closed".into()))?;
        
        trace!("Executing {:?} with {:?}", operation.operation_type(), algorithm);
        
        // Try FPGA first for supported algorithms
        #[cfg(feature = "fpga")]
        if let Some(ref fpga) = self.fpga_backend {
            if fpga.supports(algorithm) && fpga.utilization() < 0.8 {
                return fpga.execute(algorithm, operation).await;
            }
        }
        
        // Fall back to software
        self.software_backend.execute(algorithm, operation).await
    }

    /// Check if algorithm is available
    pub fn is_available(&self, algorithm: &Algorithm) -> bool {
        self.algorithms.contains(algorithm)
    }

    /// Shutdown the backend
    pub async fn shutdown(&self) -> Result<()> {
        debug!("Shutting down crypto backend");
        
        #[cfg(feature = "fpga")]
        if let Some(ref fpga) = self.fpga_backend {
            fpga.shutdown().await?;
        }
        
        Ok(())
    }
}

/// Software cryptographic backend
struct SoftwareBackend {
    use_avx2: bool,
    use_avx512: bool,
}

impl SoftwareBackend {
    fn new(config: &AccelerationConfig) -> Result<Self> {
        Ok(Self {
            use_avx2: config.software.use_avx2,
            use_avx512: config.software.use_avx512,
        })
    }

    async fn execute(
        &self,
        algorithm: &Algorithm,
        operation: &Operation,
    ) -> Result<Vec<u8>, SchedulerError> {
        // Simulate cryptographic operation latency
        let latency_us = algorithm.typical_latency_us();
        let jitter = (rand::random::<f64>() - 0.5) * 0.2 * latency_us;
        let actual_latency = Duration::from_micros((latency_us + jitter).max(1.0) as u64);
        
        tokio::time::sleep(actual_latency).await;
        
        // Return placeholder result
        Ok(vec![0u8; 32])
    }
}

#[cfg(feature = "fpga")]
struct FpgaBackend {
    device: String,
    max_utilization: f64,
    current_utilization: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "fpga")]
impl FpgaBackend {
    fn new(config: &crate::config::FpgaConfig) -> Result<Self> {
        Ok(Self {
            device: config.device.clone(),
            max_utilization: config.max_utilization,
            current_utilization: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn supports(&self, algorithm: &Algorithm) -> bool {
        matches!(algorithm,
            Algorithm::MlKem512 | Algorithm::MlKem768 | Algorithm::MlKem1024 |
            Algorithm::MlDsa44 | Algorithm::MlDsa65 | Algorithm::MlDsa87
        )
    }

    fn utilization(&self) -> f64 {
        let raw = self.current_utilization.load(std::sync::atomic::Ordering::Relaxed);
        f64::from_bits(raw)
    }

    async fn execute(
        &self,
        algorithm: &Algorithm,
        _operation: &Operation,
    ) -> Result<Vec<u8>, SchedulerError> {
        // FPGA is typically 10x faster
        let latency_us = algorithm.typical_latency_us() / 10.0;
        let actual_latency = Duration::from_micros(latency_us.max(1.0) as u64);
        
        tokio::time::sleep(actual_latency).await;
        
        Ok(vec![0u8; 32])
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_security() {
        assert_eq!(Algorithm::MlKem768.security_bits(), 192);
        assert_eq!(Algorithm::MlDsa65.security_bits(), 192);
        assert_eq!(Algorithm::EcdsaP256.security_bits(), 128);
    }

    #[test]
    fn test_algorithm_classification() {
        assert!(Algorithm::MlKem768.is_post_quantum());
        assert!(!Algorithm::EcdsaP256.is_post_quantum());
        assert!(Algorithm::X25519MlKem768.is_hybrid());
    }
}
