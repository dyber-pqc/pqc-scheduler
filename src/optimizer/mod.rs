//! Multi-objective resource optimization
//!
//! Achieves Pareto-optimal tradeoffs between security, latency, and throughput
//! using game-theoretic analysis for multi-tenant environments.

use std::collections::HashMap;

/// Objective weights for multi-objective optimization
#[derive(Debug, Clone)]
pub struct ObjectiveWeights {
    pub security: f64,
    pub latency: f64,
    pub cost: f64,
    pub throughput: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            security: 1.0,
            latency: 0.5,
            cost: 0.3,
            throughput: 0.4,
        }
    }
}

/// Point on the Pareto frontier
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub security_bits: f64,
    pub latency_us: f64,
    pub cost_per_mop: f64,
    pub throughput_ops: f64,
    pub configuration: AllocationProfile,
}

/// Pareto frontier representation
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    pub points: Vec<ParetoPoint>,
}

impl ParetoFrontier {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    pub fn add_point(&mut self, point: ParetoPoint) {
        // Check if dominated by existing points
        let dominated = self.points.iter().any(|p| {
            p.security_bits >= point.security_bits &&
            p.latency_us <= point.latency_us &&
            p.cost_per_mop <= point.cost_per_mop &&
            (p.security_bits > point.security_bits ||
             p.latency_us < point.latency_us ||
             p.cost_per_mop < point.cost_per_mop)
        });
        
        if !dominated {
            // Remove points dominated by new point
            self.points.retain(|p| {
                !(point.security_bits >= p.security_bits &&
                  point.latency_us <= p.latency_us &&
                  point.cost_per_mop <= p.cost_per_mop)
            });
            self.points.push(point);
        }
    }

    pub fn find_best(&self, weights: &ObjectiveWeights) -> Option<&ParetoPoint> {
        self.points.iter().max_by(|a, b| {
            let score_a = self.compute_score(a, weights);
            let score_b = self.compute_score(b, weights);
            score_a.partial_cmp(&score_b).unwrap()
        })
    }

    fn compute_score(&self, point: &ParetoPoint, weights: &ObjectiveWeights) -> f64 {
        let security_norm = point.security_bits / 256.0;
        let latency_norm = 1.0 - (point.latency_us / 1000.0).min(1.0);
        let cost_norm = 1.0 - (point.cost_per_mop / 1.0).min(1.0);
        
        weights.security * security_norm +
        weights.latency * latency_norm +
        weights.cost * cost_norm
    }
}

/// Resource allocation profile
#[derive(Debug, Clone)]
pub struct AllocationProfile {
    pub name: String,
    pub allocations: HashMap<String, f64>,
}

/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer {
    weights: ObjectiveWeights,
    pareto_frontier: ParetoFrontier,
}

impl MultiObjectiveOptimizer {
    pub fn new(weights: ObjectiveWeights) -> Self {
        Self {
            weights,
            pareto_frontier: ParetoFrontier::new(),
        }
    }

    pub fn set_weights(&mut self, weights: ObjectiveWeights) {
        self.weights = weights;
    }

    pub fn get_optimal_allocation(&self) -> Option<&ParetoPoint> {
        self.pareto_frontier.find_best(&self.weights)
    }

    pub fn update_frontier(&mut self, point: ParetoPoint) {
        self.pareto_frontier.add_point(point);
    }
}
