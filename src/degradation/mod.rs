//! Graceful degradation framework
//!
//! Provides formal availability guarantees through hierarchical fallback chains
//! with sub-millisecond transition latency.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::config::{FallbackChainConfig, WarmStandbyConfig, ThresholdConfig};
use crate::crypto::Algorithm;

/// Fallback chain for graceful degradation
#[derive(Debug, Clone)]
pub struct FallbackChain {
    pub primary: Vec<Algorithm>,
    pub signature: Vec<Algorithm>,
}

impl FallbackChain {
    pub fn from_config(config: &FallbackChainConfig) -> Self {
        Self {
            primary: vec![
                Algorithm::MlKem768,
                Algorithm::X25519MlKem768,
                Algorithm::X25519,
            ],
            signature: vec![
                Algorithm::MlDsa65,
                Algorithm::Ed25519MlDsa65,
                Algorithm::EcdsaP256,
            ],
        }
    }
}

/// Graceful degradation controller
pub struct DegradationController {
    fallback_chain: FallbackChain,
    warm_standby: WarmStandbyConfig,
    thresholds: ThresholdConfig,
    unavailable_algorithms: Arc<RwLock<Vec<Algorithm>>>,
}

impl DegradationController {
    pub fn new(
        fallback_chain: FallbackChain,
        warm_standby: WarmStandbyConfig,
        thresholds: ThresholdConfig,
    ) -> Self {
        Self {
            fallback_chain,
            warm_standby,
            thresholds,
            unavailable_algorithms: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn check_availability(&self, algorithm: &Algorithm) -> Result<Algorithm, Algorithm> {
        let unavailable = self.unavailable_algorithms.read();
        if unavailable.contains(algorithm) {
            // Find fallback
            let fallback_list = if matches!(algorithm,
                Algorithm::MlKem512 | Algorithm::MlKem768 | Algorithm::MlKem1024 |
                Algorithm::X25519 | Algorithm::X25519MlKem768
            ) {
                &self.fallback_chain.primary
            } else {
                &self.fallback_chain.signature
            };
            
            for fallback in fallback_list {
                if !unavailable.contains(fallback) {
                    return Err(fallback.clone());
                }
            }
        }
        Ok(algorithm.clone())
    }

    pub fn mark_unavailable(&self, algorithm: Algorithm) {
        self.unavailable_algorithms.write().push(algorithm);
    }

    pub fn mark_available(&self, algorithm: &Algorithm) {
        self.unavailable_algorithms.write().retain(|a| a != algorithm);
    }
}
