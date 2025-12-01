//! Integration tests for PQC Scheduler

use std::time::Duration;

// Test helper module
mod common {
    use std::path::PathBuf;
    
    pub fn get_test_config_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/default.yaml")
    }
    
    pub fn get_test_workload_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("workloads/constant_rate.json")
    }
}

#[cfg(test)]
mod config_tests {
    use super::common;
    
    #[test]
    fn test_load_default_config() {
        let config_path = common::get_test_config_path();
        
        // Read config file
        let content = std::fs::read_to_string(&config_path)
            .expect("Failed to read config file");
        
        // Parse YAML
        let config: serde_yaml::Value = serde_yaml::from_str(&content)
            .expect("Failed to parse YAML");
        
        // Verify essential fields exist
        assert!(config.get("scheduler").is_some(), "Missing scheduler config");
        assert!(config.get("algorithms").is_some(), "Missing algorithms config");
        assert!(config.get("degradation").is_some(), "Missing degradation config");
        assert!(config.get("migration").is_some(), "Missing migration config");
    }
    
    #[test]
    fn test_mdp_config_values() {
        let config_path = common::get_test_config_path();
        let content = std::fs::read_to_string(&config_path).unwrap();
        let config: serde_yaml::Value = serde_yaml::from_str(&content).unwrap();
        
        let mdp = config.get("scheduler")
            .and_then(|s| s.get("mdp"))
            .expect("Missing MDP config");
        
        let gamma = mdp.get("gamma")
            .and_then(|v| v.as_f64())
            .expect("Missing gamma");
        
        assert!(gamma > 0.0 && gamma < 1.0, "gamma must be in (0, 1)");
        
        let epsilon = mdp.get("epsilon")
            .and_then(|v| v.as_f64())
            .expect("Missing epsilon");
        
        assert!(epsilon > 0.0, "epsilon must be positive");
    }
}

#[cfg(test)]
mod workload_tests {
    use super::common;
    
    #[test]
    fn test_load_workload() {
        let workload_path = common::get_test_workload_path();
        
        let content = std::fs::read_to_string(&workload_path)
            .expect("Failed to read workload file");
        
        let workload: serde_json::Value = serde_json::from_str(&content)
            .expect("Failed to parse workload JSON");
        
        assert!(workload.get("name").is_some(), "Missing name");
        assert!(workload.get("arrival_model").is_some(), "Missing arrival_model");
        assert!(workload.get("algorithm_mix").is_some(), "Missing algorithm_mix");
    }
    
    #[test]
    fn test_algorithm_mix_valid() {
        let workload_path = common::get_test_workload_path();
        let content = std::fs::read_to_string(&workload_path).unwrap();
        let workload: serde_json::Value = serde_json::from_str(&content).unwrap();
        
        let algo_mix = workload.get("algorithm_mix").expect("Missing algorithm_mix");
        
        let kem = algo_mix.get("kem")
            .and_then(|v| v.as_f64())
            .expect("Missing kem ratio");
        
        let signature = algo_mix.get("signature")
            .and_then(|v| v.as_f64())
            .expect("Missing signature ratio");
        
        let sum = kem + signature;
        assert!((sum - 1.0).abs() < 0.001, "Algorithm mix should sum to 1.0");
    }
}

#[cfg(test)]
mod algorithm_tests {
    
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Algorithm {
        MlKem512,
        MlKem768,
        MlKem1024,
        MlDsa44,
        MlDsa65,
        MlDsa87,
        X25519,
        EcdsaP256,
    }
    
    impl Algorithm {
        fn security_bits(&self) -> u32 {
            match self {
                Algorithm::MlKem512 | Algorithm::MlDsa44 => 128,
                Algorithm::MlKem768 | Algorithm::MlDsa65 => 192,
                Algorithm::MlKem1024 | Algorithm::MlDsa87 => 256,
                Algorithm::X25519 | Algorithm::EcdsaP256 => 128,
            }
        }
        
        fn is_post_quantum(&self) -> bool {
            matches!(self, 
                Algorithm::MlKem512 | Algorithm::MlKem768 | Algorithm::MlKem1024 |
                Algorithm::MlDsa44 | Algorithm::MlDsa65 | Algorithm::MlDsa87
            )
        }
    }
    
    #[test]
    fn test_algorithm_security_levels() {
        assert_eq!(Algorithm::MlKem512.security_bits(), 128);
        assert_eq!(Algorithm::MlKem768.security_bits(), 192);
        assert_eq!(Algorithm::MlKem1024.security_bits(), 256);
        assert_eq!(Algorithm::MlDsa44.security_bits(), 128);
        assert_eq!(Algorithm::MlDsa65.security_bits(), 192);
        assert_eq!(Algorithm::MlDsa87.security_bits(), 256);
    }
    
    #[test]
    fn test_post_quantum_classification() {
        assert!(Algorithm::MlKem768.is_post_quantum());
        assert!(Algorithm::MlDsa65.is_post_quantum());
        assert!(!Algorithm::X25519.is_post_quantum());
        assert!(!Algorithm::EcdsaP256.is_post_quantum());
    }
    
    #[test]
    fn test_security_ordering() {
        // Verify security levels are properly ordered
        assert!(Algorithm::MlKem1024.security_bits() > Algorithm::MlKem768.security_bits());
        assert!(Algorithm::MlKem768.security_bits() > Algorithm::MlKem512.security_bits());
        assert!(Algorithm::MlDsa87.security_bits() > Algorithm::MlDsa65.security_bits());
        assert!(Algorithm::MlDsa65.security_bits() > Algorithm::MlDsa44.security_bits());
    }
}

#[cfg(test)]
mod mdp_tests {
    use std::collections::HashMap;
    
    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    struct DiscreteState {
        threat_bin: usize,
        workload_bin: usize,
    }
    
    fn discretize_threat(threat: f64) -> usize {
        ((threat * 10.0).floor() as usize).min(10)
    }
    
    fn discretize_workload(workload: f64, bins: &[f64]) -> usize {
        for (i, &threshold) in bins.iter().enumerate() {
            if workload < threshold {
                return i;
            }
        }
        bins.len() - 1
    }
    
    #[test]
    fn test_threat_discretization() {
        assert_eq!(discretize_threat(0.0), 0);
        assert_eq!(discretize_threat(0.1), 1);
        assert_eq!(discretize_threat(0.5), 5);
        assert_eq!(discretize_threat(0.99), 9);
        assert_eq!(discretize_threat(1.0), 10);
    }
    
    #[test]
    fn test_workload_discretization() {
        let bins = vec![1000.0, 5000.0, 10000.0, 50000.0, 100000.0];
        
        assert_eq!(discretize_workload(500.0, &bins), 0);
        assert_eq!(discretize_workload(3000.0, &bins), 1);
        assert_eq!(discretize_workload(7000.0, &bins), 2);
        assert_eq!(discretize_workload(30000.0, &bins), 3);
        assert_eq!(discretize_workload(75000.0, &bins), 4);
        assert_eq!(discretize_workload(200000.0, &bins), 4);
    }
    
    #[test]
    fn test_bellman_update() {
        // Simple Bellman update test
        let gamma = 0.99;
        let reward = 1.0;
        let future_value = 10.0;
        
        let q_value = reward + gamma * future_value;
        
        assert!((q_value - 10.9).abs() < 0.001);
    }
    
    #[test]
    fn test_value_iteration_convergence() {
        // Simple 2-state MDP
        let gamma = 0.9;
        let epsilon = 0.001;
        
        let mut v = vec![0.0, 0.0];
        let rewards = vec![1.0, 2.0];
        let transitions = vec![
            vec![0.5, 0.5],  // State 0 transitions
            vec![0.3, 0.7],  // State 1 transitions
        ];
        
        for _ in 0..1000 {
            let old_v = v.clone();
            
            for s in 0..2 {
                let future: f64 = (0..2)
                    .map(|s2| transitions[s][s2] * old_v[s2])
                    .sum();
                v[s] = rewards[s] + gamma * future;
            }
            
            let max_delta: f64 = v.iter()
                .zip(old_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            
            if max_delta < epsilon {
                break;
            }
        }
        
        // Values should have converged to positive values
        assert!(v[0] > 0.0);
        assert!(v[1] > 0.0);
        assert!(v[1] > v[0]);  // State 1 has higher reward
    }
}

#[cfg(test)]
mod degradation_tests {
    
    #[derive(Debug, Clone, PartialEq)]
    enum Algorithm {
        MlKem768,
        X25519MlKem768,
        X25519,
    }
    
    struct FallbackChain {
        chain: Vec<Algorithm>,
    }
    
    impl FallbackChain {
        fn new(chain: Vec<Algorithm>) -> Self {
            Self { chain }
        }
        
        fn get_fallback(&self, unavailable: &[Algorithm]) -> Option<&Algorithm> {
            self.chain.iter().find(|a| !unavailable.contains(a))
        }
    }
    
    #[test]
    fn test_fallback_chain_first_available() {
        let chain = FallbackChain::new(vec![
            Algorithm::MlKem768,
            Algorithm::X25519MlKem768,
            Algorithm::X25519,
        ]);
        
        let unavailable: Vec<Algorithm> = vec![];
        assert_eq!(chain.get_fallback(&unavailable), Some(&Algorithm::MlKem768));
    }
    
    #[test]
    fn test_fallback_chain_skip_unavailable() {
        let chain = FallbackChain::new(vec![
            Algorithm::MlKem768,
            Algorithm::X25519MlKem768,
            Algorithm::X25519,
        ]);
        
        let unavailable = vec![Algorithm::MlKem768];
        assert_eq!(chain.get_fallback(&unavailable), Some(&Algorithm::X25519MlKem768));
    }
    
    #[test]
    fn test_fallback_chain_all_unavailable() {
        let chain = FallbackChain::new(vec![
            Algorithm::MlKem768,
            Algorithm::X25519MlKem768,
            Algorithm::X25519,
        ]);
        
        let unavailable = vec![
            Algorithm::MlKem768,
            Algorithm::X25519MlKem768,
            Algorithm::X25519,
        ];
        assert_eq!(chain.get_fallback(&unavailable), None);
    }
    
    #[test]
    fn test_availability_calculation() {
        // Test system availability with redundancy
        let individual_availability = 0.999;  // Three nines
        let redundancy = 3;
        
        // A_sys = 1 - ∏(1 - A_i)
        let unavailability: f64 = (0..redundancy)
            .map(|_| 1.0 - individual_availability)
            .product();
        let system_availability = 1.0 - unavailability;
        
        // With 3 algorithms at 99.9%, system should achieve 99.9999999%
        assert!(system_availability > 0.999999);
    }
}

#[cfg(test)]
mod migration_tests {
    
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Phase {
        Classical,
        Hybrid,
        PostQuantum,
    }
    
    impl Phase {
        fn advance(&self) -> Option<Phase> {
            match self {
                Phase::Classical => Some(Phase::Hybrid),
                Phase::Hybrid => Some(Phase::PostQuantum),
                Phase::PostQuantum => None,
            }
        }
        
        fn rollback(&self) -> Option<Phase> {
            match self {
                Phase::Classical => None,
                Phase::Hybrid => Some(Phase::Classical),
                Phase::PostQuantum => Some(Phase::Hybrid),
            }
        }
    }
    
    #[test]
    fn test_phase_advancement() {
        let phase = Phase::Classical;
        assert_eq!(phase.advance(), Some(Phase::Hybrid));
        
        let phase = Phase::Hybrid;
        assert_eq!(phase.advance(), Some(Phase::PostQuantum));
        
        let phase = Phase::PostQuantum;
        assert_eq!(phase.advance(), None);
    }
    
    #[test]
    fn test_phase_rollback() {
        let phase = Phase::PostQuantum;
        assert_eq!(phase.rollback(), Some(Phase::Hybrid));
        
        let phase = Phase::Hybrid;
        assert_eq!(phase.rollback(), Some(Phase::Classical));
        
        let phase = Phase::Classical;
        assert_eq!(phase.rollback(), None);
    }
    
    #[test]
    fn test_migration_risk_calculation() {
        // Test risk calculation formula
        let exposure = 0.5;  // 50% classical exposure
        let sensitivity = 1.0;  // Normalized sensitivity
        let pq_probability = 0.1;  // 10% quantum threat probability
        let failure_probability = 0.05;
        let failure_cost = 100.0;
        
        let risk = exposure * sensitivity * pq_probability 
            + failure_probability * failure_cost;
        
        assert!((risk - 5.05).abs() < 0.001);
    }
}

#[cfg(test)]
mod optimizer_tests {
    
    #[derive(Debug, Clone)]
    struct ParetoPoint {
        security: f64,
        latency: f64,
        cost: f64,
    }
    
    impl ParetoPoint {
        fn dominates(&self, other: &ParetoPoint) -> bool {
            self.security >= other.security
                && self.latency <= other.latency
                && self.cost <= other.cost
                && (self.security > other.security
                    || self.latency < other.latency
                    || self.cost < other.cost)
        }
    }
    
    fn is_pareto_optimal(point: &ParetoPoint, all_points: &[ParetoPoint]) -> bool {
        !all_points.iter().any(|p| p.dominates(point))
    }
    
    #[test]
    fn test_pareto_dominance() {
        let a = ParetoPoint { security: 192.0, latency: 100.0, cost: 0.3 };
        let b = ParetoPoint { security: 128.0, latency: 150.0, cost: 0.4 };
        
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }
    
    #[test]
    fn test_pareto_non_dominance() {
        // These points are non-dominated (Pareto optimal)
        let a = ParetoPoint { security: 256.0, latency: 500.0, cost: 0.5 };
        let b = ParetoPoint { security: 192.0, latency: 200.0, cost: 0.3 };
        
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }
    
    #[test]
    fn test_pareto_frontier_identification() {
        let points = vec![
            ParetoPoint { security: 256.0, latency: 500.0, cost: 0.5 },  // Pareto optimal
            ParetoPoint { security: 192.0, latency: 200.0, cost: 0.3 },  // Pareto optimal
            ParetoPoint { security: 128.0, latency: 100.0, cost: 0.2 },  // Pareto optimal
            ParetoPoint { security: 128.0, latency: 300.0, cost: 0.4 },  // Dominated
        ];
        
        let pareto_count = points.iter()
            .filter(|p| is_pareto_optimal(p, &points))
            .count();
        
        assert_eq!(pareto_count, 3);
    }
}
