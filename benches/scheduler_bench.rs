//! Scheduler benchmarks using Criterion
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

// Mock implementations for benchmarking
mod mock {
    use std::collections::HashMap;
    
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Algorithm {
        MlKem768,
        MlDsa65,
        X25519,
        EcdsaP256,
    }
    
    impl Algorithm {
        pub fn security_bits(&self) -> u32 {
            match self {
                Algorithm::MlKem768 | Algorithm::MlDsa65 => 192,
                Algorithm::X25519 | Algorithm::EcdsaP256 => 128,
            }
        }
        
        pub fn typical_latency_us(&self) -> f64 {
            match self {
                Algorithm::MlKem768 => 36.0,
                Algorithm::MlDsa65 => 287.0,
                Algorithm::X25519 => 25.0,
                Algorithm::EcdsaP256 => 68.0,
            }
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct State {
        pub threat_level: f64,
        pub workload: f64,
        pub utilization: Vec<f64>,
    }
    
    impl State {
        pub fn new(threat: f64, workload: f64, util: Vec<f64>) -> Self {
            Self {
                threat_level: threat,
                workload,
                utilization: util,
            }
        }
        
        pub fn discretize(&self) -> DiscreteState {
            DiscreteState {
                threat_bin: (self.threat_level * 10.0) as usize,
                workload_bin: (self.workload / 50000.0).min(10.0) as usize,
            }
        }
    }
    
    #[derive(Debug, Clone, Hash, PartialEq, Eq)]
    pub struct DiscreteState {
        pub threat_bin: usize,
        pub workload_bin: usize,
    }
    
    #[derive(Debug, Clone)]
    pub struct Action {
        pub kem_algorithm: Algorithm,
        pub signature_algorithm: Algorithm,
    }
    
    pub struct MdpPolicy {
        policy: HashMap<DiscreteState, Action>,
    }
    
    impl MdpPolicy {
        pub fn new() -> Self {
            Self {
                policy: HashMap::new(),
            }
        }
        
        pub fn set_action(&mut self, state: DiscreteState, action: Action) {
            self.policy.insert(state, action);
        }
        
        pub fn get_action(&self, state: &State) -> Option<Action> {
            let discrete = state.discretize();
            self.policy.get(&discrete).cloned()
        }
    }
    
    pub fn compute_reward(state: &DiscreteState, action: &Action) -> f64 {
        let security = (action.kem_algorithm.security_bits() as f64 
            + action.signature_algorithm.security_bits() as f64) / 2.0 / 256.0;
        let latency = 1.0 - (action.kem_algorithm.typical_latency_us() 
            + action.signature_algorithm.typical_latency_us()) / 2.0 / 1000.0;
        let threat_bonus = if state.threat_bin > 5 { 0.2 } else { 0.0 };
        
        security + 0.5 * latency + threat_bonus
    }
    
    pub fn value_iteration(states: &[DiscreteState], actions: &[Action], gamma: f64, epsilon: f64) -> MdpPolicy {
        let mut policy = MdpPolicy::new();
        let mut values: HashMap<DiscreteState, f64> = HashMap::new();
        
        // Initialize values
        for state in states {
            values.insert(state.clone(), 0.0);
        }
        
        let mut iteration = 0;
        let max_iterations = 1000;
        
        loop {
            let mut max_delta = 0.0f64;
            
            for state in states {
                let old_value = *values.get(state).unwrap_or(&0.0);
                
                let (best_value, best_action) = actions.iter()
                    .map(|action| {
                        let reward = compute_reward(state, action);
                        let future = gamma * old_value;
                        (reward + future, action.clone())
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();
                
                values.insert(state.clone(), best_value);
                policy.set_action(state.clone(), best_action);
                
                max_delta = max_delta.max((old_value - best_value).abs());
            }
            
            iteration += 1;
            
            if max_delta < epsilon || iteration >= max_iterations {
                break;
            }
        }
        
        policy
    }
}

use mock::*;

fn bench_state_discretization(c: &mut Criterion) {
    let state = State::new(0.5, 100000.0, vec![0.3, 0.5, 0.7, 0.2]);
    
    c.bench_function("state_discretization", |b| {
        b.iter(|| {
            black_box(state.discretize())
        })
    });
}

fn bench_policy_lookup(c: &mut Criterion) {
    // Create a policy with some entries
    let mut policy = MdpPolicy::new();
    for threat in 0..11 {
        for workload in 0..11 {
            let state = DiscreteState {
                threat_bin: threat,
                workload_bin: workload,
            };
            let action = Action {
                kem_algorithm: if threat > 5 { Algorithm::MlKem768 } else { Algorithm::X25519 },
                signature_algorithm: if threat > 5 { Algorithm::MlDsa65 } else { Algorithm::EcdsaP256 },
            };
            policy.set_action(state, action);
        }
    }
    
    let state = State::new(0.5, 100000.0, vec![0.3, 0.5, 0.7, 0.2]);
    
    c.bench_function("policy_lookup", |b| {
        b.iter(|| {
            black_box(policy.get_action(&state))
        })
    });
}

fn bench_reward_computation(c: &mut Criterion) {
    let state = DiscreteState {
        threat_bin: 5,
        workload_bin: 5,
    };
    let action = Action {
        kem_algorithm: Algorithm::MlKem768,
        signature_algorithm: Algorithm::MlDsa65,
    };
    
    c.bench_function("reward_computation", |b| {
        b.iter(|| {
            black_box(compute_reward(&state, &action))
        })
    });
}

fn bench_value_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_iteration");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    
    for state_count in [100, 500, 1000].iter() {
        let states: Vec<DiscreteState> = (0..11)
            .flat_map(|t| (0..*state_count/11).map(move |w| DiscreteState {
                threat_bin: t,
                workload_bin: w % 11,
            }))
            .take(*state_count)
            .collect();
        
        let actions = vec![
            Action { kem_algorithm: Algorithm::MlKem768, signature_algorithm: Algorithm::MlDsa65 },
            Action { kem_algorithm: Algorithm::X25519, signature_algorithm: Algorithm::EcdsaP256 },
            Action { kem_algorithm: Algorithm::MlKem768, signature_algorithm: Algorithm::EcdsaP256 },
            Action { kem_algorithm: Algorithm::X25519, signature_algorithm: Algorithm::MlDsa65 },
        ];
        
        group.throughput(Throughput::Elements(*state_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(state_count),
            state_count,
            |b, _| {
                b.iter(|| {
                    black_box(value_iteration(&states, &actions, 0.99, 1e-6))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_algorithm_selection(c: &mut Criterion) {
    let algorithms = [
        Algorithm::MlKem768,
        Algorithm::MlDsa65,
        Algorithm::X25519,
        Algorithm::EcdsaP256,
    ];
    
    c.bench_function("algorithm_security_lookup", |b| {
        b.iter(|| {
            for algo in &algorithms {
                black_box(algo.security_bits());
            }
        })
    });
    
    c.bench_function("algorithm_latency_lookup", |b| {
        b.iter(|| {
            for algo in &algorithms {
                black_box(algo.typical_latency_us());
            }
        })
    });
}

criterion_group!(
    benches,
    bench_state_discretization,
    bench_policy_lookup,
    bench_reward_computation,
    bench_value_iteration,
    bench_algorithm_selection,
);

criterion_main!(benches);
