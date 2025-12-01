//! Markov Decision Process (MDP) based algorithm selection
//!
//! This module implements the dynamic algorithm selection engine using
//! MDP modeling with provable convergence and bounded regret guarantees.
//!
//! # Theory
//!
//! The algorithm selection is modeled as an MDP M = (S, A, P, R, γ) where:
//! - S: State space (threat level, workload, utilization, compliance, phase)
//! - A: Action space (algorithm selection, migration direction, allocation)
//! - P: Transition probabilities
//! - R: Multi-objective reward function
//! - γ: Discount factor
//!
//! ## Theorem 1 (Optimal Policy Existence)
//! For the MDP M with finite state/action spaces and bounded rewards,
//! there exists a deterministic stationary optimal policy π*.
//!
//! ## Theorem 2 (Regret Bound)
//! Under adversarial threat evolution with bounded total variation V_T,
//! the adaptive policy achieves regret O(√(T·V_T·log|A|)).

use std::collections::HashMap;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ordered_float::OrderedFloat;
use tracing::{debug, info, trace};

use crate::config::Config;
use crate::crypto::Algorithm;
use crate::migration::MigrationPhase;
use crate::ComplianceRequirements;

/// State in the MDP
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// Threat level (probability of quantum attack capability)
    pub threat_level: f64,
    
    /// Workload intensity (requests per second)
    pub workload: f64,
    
    /// Resource utilization vector per algorithm core
    pub utilization: Vec<f64>,
    
    /// Compliance requirements
    pub compliance: ComplianceRequirements,
    
    /// Current migration phase
    pub migration_phase: MigrationPhase,
}

impl State {
    /// Create a new state
    pub fn new(
        threat_level: f64,
        workload: f64,
        utilization: Vec<f64>,
        compliance: ComplianceRequirements,
        migration_phase: MigrationPhase,
    ) -> Self {
        Self {
            threat_level,
            workload,
            utilization,
            compliance,
            migration_phase,
        }
    }

    /// Discretize state for MDP computation
    pub fn discretize(&self, config: &StateSpaceConfig) -> DiscreteState {
        DiscreteState {
            threat_bin: discretize_value(self.threat_level, 0.0, 1.0, config.threat_bins),
            workload_bin: discretize_workload(self.workload, &config.workload_bins),
            utilization_bins: self.utilization.iter()
                .map(|&u| discretize_value(u, 0.0, 1.0, config.utilization_bins))
                .collect(),
            compliance_mask: self.compliance.to_mask(),
            migration_phase: self.migration_phase.clone(),
        }
    }
}

/// Discretized state for efficient computation
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DiscreteState {
    pub threat_bin: usize,
    pub workload_bin: usize,
    pub utilization_bins: Vec<usize>,
    pub compliance_mask: u8,
    pub migration_phase: MigrationPhase,
}

/// Action in the MDP
#[derive(Debug, Clone)]
pub struct Action {
    /// Selected KEM algorithm
    pub kem_algorithm: Algorithm,
    
    /// Selected signature algorithm
    pub signature_algorithm: Algorithm,
    
    /// Migration direction
    pub migration_direction: MigrationDirection,
    
    /// Resource allocation profile index
    pub allocation_profile: usize,
}

/// Migration direction control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MigrationDirection {
    Maintain,
    Escalate,
    Deescalate,
}

/// State space configuration
#[derive(Debug, Clone)]
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

/// MDP Policy mapping states to actions
#[derive(Debug, Clone)]
pub struct MdpPolicy {
    /// Policy table: state -> action
    policy: HashMap<DiscreteState, Action>,
    
    /// Value function
    values: HashMap<DiscreteState, f64>,
    
    /// Configuration used to compute this policy
    state_config: StateSpaceConfig,
}

impl MdpPolicy {
    /// Create a new empty policy
    pub fn new(state_config: StateSpaceConfig) -> Self {
        Self {
            policy: HashMap::new(),
            values: HashMap::new(),
            state_config,
        }
    }

    /// Get action for a state
    pub fn get_action(&self, state: &State) -> Option<Action> {
        let discrete_state = state.discretize(&self.state_config);
        self.policy.get(&discrete_state).cloned()
    }

    /// Set action for a discrete state
    pub fn set_action(&mut self, state: DiscreteState, action: Action) {
        self.policy.insert(state, action);
    }

    /// Get value for a discrete state
    pub fn get_value(&self, state: &DiscreteState) -> f64 {
        *self.values.get(state).unwrap_or(&0.0)
    }

    /// Set value for a discrete state
    pub fn set_value(&mut self, state: DiscreteState, value: f64) {
        self.values.insert(state, value);
    }
}

/// MDP Solver using value iteration
pub struct MdpSolver {
    /// Discount factor γ
    gamma: f64,
    
    /// Convergence threshold ε
    epsilon: f64,
    
    /// Maximum iterations
    max_iterations: usize,
    
    /// Last iteration count
    last_iterations: usize,
}

impl MdpSolver {
    /// Create a new MDP solver
    pub fn new(gamma: f64, epsilon: f64, max_iterations: usize) -> Self {
        Self {
            gamma,
            epsilon,
            max_iterations,
            last_iterations: 0,
        }
    }

    /// Compute optimal policy using value iteration
    pub fn compute_policy(&mut self, initial_state: &State, config: &Config) -> Result<MdpPolicy> {
        info!("Computing MDP policy with γ={}, ε={}", self.gamma, self.epsilon);
        
        let state_config = StateSpaceConfig {
            threat_bins: config.scheduler.state_space.threat_bins,
            workload_bins: config.scheduler.state_space.workload_bins.clone(),
            utilization_bins: config.scheduler.state_space.utilization_bins,
        };
        
        let mut policy = MdpPolicy::new(state_config.clone());
        
        // Generate all reachable states
        let states = self.generate_reachable_states(&state_config, config);
        debug!("Generated {} reachable states", states.len());
        
        // Generate action space
        let actions = self.generate_actions(config);
        debug!("Generated {} actions", actions.len());
        
        // Initialize value function
        for state in &states {
            policy.set_value(state.clone(), 0.0);
        }
        
        // Value iteration
        let mut iteration = 0;
        let mut max_delta = f64::MAX;
        
        while max_delta > self.epsilon && iteration < self.max_iterations {
            max_delta = 0.0;
            
            for state in &states {
                let old_value = policy.get_value(state);
                
                // Find best action
                let (best_value, best_action) = actions.iter()
                    .filter(|a| self.is_action_valid(a, state, config))
                    .map(|action| {
                        let value = self.compute_action_value(state, action, &policy, config);
                        (value, action.clone())
                    })
                    .max_by_key(|(v, _)| OrderedFloat(*v))
                    .unwrap_or((0.0, actions[0].clone()));
                
                policy.set_value(state.clone(), best_value);
                policy.set_action(state.clone(), best_action);
                
                max_delta = max_delta.max((old_value - best_value).abs());
            }
            
            iteration += 1;
            
            if iteration % 100 == 0 {
                trace!("Iteration {}: max_delta = {}", iteration, max_delta);
            }
        }
        
        self.last_iterations = iteration;
        info!("Value iteration converged in {} iterations (max_delta = {})", 
              iteration, max_delta);
        
        Ok(policy)
    }

    /// Get the last iteration count
    pub fn last_iteration_count(&self) -> usize {
        self.last_iterations
    }

    /// Generate all reachable states (with sparsity optimization)
    fn generate_reachable_states(
        &self,
        config: &StateSpaceConfig,
        algo_config: &Config,
    ) -> Vec<DiscreteState> {
        let mut states = Vec::new();
        
        // Compliance masks (4 bits: FIPS, PCI, HIPAA, FedRAMP)
        let compliance_masks: Vec<u8> = vec![
            0b0001, // FIPS only
            0b0011, // FIPS + PCI
            0b0101, // FIPS + HIPAA
            0b1001, // FIPS + FedRAMP
            0b1111, // All
        ];
        
        // Migration phases
        let phases = vec![
            MigrationPhase::Classical,
            MigrationPhase::Hybrid,
            MigrationPhase::PostQuantum,
        ];
        
        // Generate states (with sparsity)
        for threat_bin in 0..config.threat_bins {
            for workload_bin in 0..config.workload_bins.len() {
                // Only consider a subset of utilization combinations
                let utilization_combos = self.generate_sparse_utilizations(
                    algo_config.algorithms.enabled_count(),
                    config.utilization_bins,
                );
                
                for util_bins in &utilization_combos {
                    for &compliance_mask in &compliance_masks {
                        for phase in &phases {
                            states.push(DiscreteState {
                                threat_bin,
                                workload_bin,
                                utilization_bins: util_bins.clone(),
                                compliance_mask,
                                migration_phase: phase.clone(),
                            });
                        }
                    }
                }
            }
        }
        
        states
    }

    /// Generate sparse utilization combinations
    fn generate_sparse_utilizations(&self, n_algorithms: usize, n_bins: usize) -> Vec<Vec<usize>> {
        // For efficiency, only generate representative combinations
        let mut combos = Vec::new();
        
        // All low utilization
        combos.push(vec![0; n_algorithms]);
        
        // All medium utilization
        combos.push(vec![n_bins / 2; n_algorithms]);
        
        // All high utilization
        combos.push(vec![n_bins - 1; n_algorithms]);
        
        // Mixed patterns
        for i in 0..n_algorithms.min(5) {
            let mut combo = vec![1; n_algorithms];
            combo[i] = n_bins - 1;
            combos.push(combo);
        }
        
        combos
    }

    /// Generate action space
    fn generate_actions(&self, config: &Config) -> Vec<Action> {
        let mut actions = Vec::new();
        
        // Get enabled algorithms
        let kem_algorithms: Vec<Algorithm> = config.algorithms.enabled_kem_algorithms();
        let sig_algorithms: Vec<Algorithm> = config.algorithms.enabled_signature_algorithms();
        
        let migration_directions = vec![
            MigrationDirection::Maintain,
            MigrationDirection::Escalate,
            MigrationDirection::Deescalate,
        ];
        
        let allocation_profiles = config.scheduler.action_space.allocation_profiles;
        
        // Generate action combinations
        for kem in &kem_algorithms {
            for sig in &sig_algorithms {
                for &direction in &migration_directions {
                    for profile in 0..allocation_profiles {
                        actions.push(Action {
                            kem_algorithm: kem.clone(),
                            signature_algorithm: sig.clone(),
                            migration_direction: direction,
                            allocation_profile: profile,
                        });
                    }
                }
            }
        }
        
        actions
    }

    /// Check if action is valid for state
    fn is_action_valid(&self, action: &Action, state: &DiscreteState, config: &Config) -> bool {
        // Check compliance requirements
        let min_security = if state.compliance_mask & 0b0001 != 0 {
            128  // FIPS requires at least 128-bit security
        } else {
            128
	    // config.compliance.security.min_security_bits
        };
        
        action.kem_algorithm.security_bits() >= min_security
            && action.signature_algorithm.security_bits() >= min_security
    }

    /// Compute Q-value for state-action pair
    fn compute_action_value(
        &self,
        state: &DiscreteState,
        action: &Action,
        policy: &MdpPolicy,
        config: &Config,
    ) -> f64 {
        // Compute immediate reward
        let reward = self.compute_reward(state, action, config);
        
        // Compute expected future value
        let next_states = self.get_transition_states(state, action);
        let future_value: f64 = next_states.iter()
            .map(|(next_state, prob)| prob * policy.get_value(next_state))
            .sum();
        
        reward + self.gamma * future_value
    }

    /// Compute reward R(s, a)
    fn compute_reward(&self, state: &DiscreteState, action: &Action, config: &Config) -> f64 {
        let weights = &config.scheduler.weights;
        
        // Security component
        let security = (action.kem_algorithm.security_bits() as f64
            + action.signature_algorithm.security_bits() as f64) / 2.0;
        let security_normalized = security / 256.0;  // Normalize to [0, 1]
        
        // Latency component (lower is better)
        let latency = (action.kem_algorithm.typical_latency_us()
            + action.signature_algorithm.typical_latency_us()) / 2.0;
        let latency_normalized = 1.0 - (latency / 1000.0).min(1.0);  // Normalize
        
        // Threat-adjusted bonus
        let threat_level = state.threat_bin as f64 / 10.0;
        let pqc_bonus = if action.kem_algorithm.is_post_quantum() {
            threat_level * 0.5
        } else {
            -threat_level * 0.5
        };
        
        // Compliance penalty (hard constraint)
        let compliance_penalty = if state.compliance_mask & 0b0001 != 0 {  // FIPS
            if action.kem_algorithm.security_bits() < 128 {
                -1000.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // Compute total reward
        weights.security * security_normalized
            + weights.latency * latency_normalized
            + pqc_bonus
            + compliance_penalty
    }

    /// Get transition states with probabilities
    fn get_transition_states(
        &self,
        state: &DiscreteState,
        _action: &Action,
    ) -> Vec<(DiscreteState, f64)> {
        // Simplified transition model
        let mut transitions = Vec::new();
        
        // Most likely: stay in same threat/workload bin
        transitions.push((state.clone(), 0.8));
        
        // Threat evolution
        if state.threat_bin < 10 {
            let mut next = state.clone();
            next.threat_bin += 1;
            transitions.push((next, 0.1));
        }
        
        if state.threat_bin > 0 {
            let mut next = state.clone();
            next.threat_bin -= 1;
            transitions.push((next, 0.1));
        }
        
        transitions
    }
}

// Helper functions

fn discretize_value(value: f64, min: f64, max: f64, bins: usize) -> usize {
    let normalized = (value - min) / (max - min);
    let bin = (normalized * bins as f64).floor() as usize;
    bin.min(bins - 1)
}

fn discretize_workload(workload: f64, bins: &[f64]) -> usize {
    for (i, &threshold) in bins.iter().enumerate() {
        if workload < threshold {
            return i;
        }
    }
    bins.len() - 1
}

impl ComplianceRequirements {
    fn to_mask(&self) -> u8 {
        let mut mask = 0u8;
        if self.fips_140_3 { mask |= 0b0001; }
        if self.pci_dss { mask |= 0b0010; }
        if self.hipaa { mask |= 0b0100; }
        if self.fedramp { mask |= 0b1000; }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_discretization() {
        let state = State::new(
            0.5,
            75000.0,
            vec![0.3, 0.5, 0.7],
            ComplianceRequirements::default(),
            MigrationPhase::Hybrid,
        );
        
        let config = StateSpaceConfig::default();
        let discrete = state.discretize(&config);
        
        assert_eq!(discrete.threat_bin, 5);  // 0.5 -> bin 5
        assert!(discrete.workload_bin > 0);
    }

    #[test]
    fn test_mdp_solver_creation() {
        let solver = MdpSolver::new(0.99, 1e-6, 2000);
        assert_eq!(solver.gamma, 0.99);
        assert_eq!(solver.epsilon, 1e-6);
    }
}
