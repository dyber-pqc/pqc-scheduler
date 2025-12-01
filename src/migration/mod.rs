//! Migration path management
//!
//! Provides risk-quantified transition strategies with automatic rollback capabilities.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::{RollbackConfig, DualKeyConfig};

/// Migration phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MigrationPhase {
    #[serde(rename = "classical")]
    Classical,
    #[serde(rename = "hybrid")]
    Hybrid,
    #[serde(rename = "pqc")]
    PostQuantum,
}

impl Default for MigrationPhase {
    fn default() -> Self {
        MigrationPhase::Hybrid
    }
}

/// Migration strategy
#[derive(Debug, Clone)]
pub struct MigrationStrategy {
    pub name: String,
    pub phases: Vec<MigrationPhase>,
    pub duration_weeks: u32,
    pub risk_score: f64,
}

/// Migration manager
pub struct MigrationManager {
    current_phase: MigrationPhase,
    rollback_config: RollbackConfig,
    dual_key_config: DualKeyConfig,
    last_rollback: Option<Instant>,
    rollbacks_today: u32,
}

impl MigrationManager {
    pub fn new(
        current_phase: MigrationPhase,
        rollback_config: RollbackConfig,
        dual_key_config: DualKeyConfig,
    ) -> Self {
        Self {
            current_phase,
            rollback_config,
            dual_key_config,
            last_rollback: None,
            rollbacks_today: 0,
        }
    }

    pub fn current_phase(&self) -> MigrationPhase {
        self.current_phase
    }

    pub fn can_rollback(&self) -> bool {
        if !self.rollback_config.enabled {
            return false;
        }
        
        if self.rollbacks_today >= self.rollback_config.max_rollbacks_per_day {
            return false;
        }
        
        if let Some(last) = self.last_rollback {
            if last.elapsed() < Duration::from_secs(self.rollback_config.cooldown_sec) {
                return false;
            }
        }
        
        true
    }

    pub fn rollback(&mut self) -> Option<MigrationPhase> {
        if !self.can_rollback() {
            return None;
        }
        
        let new_phase = match self.current_phase {
            MigrationPhase::PostQuantum => MigrationPhase::Hybrid,
            MigrationPhase::Hybrid => MigrationPhase::Classical,
            MigrationPhase::Classical => return None,
        };
        
        self.current_phase = new_phase;
        self.last_rollback = Some(Instant::now());
        self.rollbacks_today += 1;
        
        Some(new_phase)
    }

    pub fn advance(&mut self) -> Option<MigrationPhase> {
        let new_phase = match self.current_phase {
            MigrationPhase::Classical => MigrationPhase::Hybrid,
            MigrationPhase::Hybrid => MigrationPhase::PostQuantum,
            MigrationPhase::PostQuantum => return None,
        };
        
        self.current_phase = new_phase;
        Some(new_phase)
    }
}
