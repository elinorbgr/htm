//! Cortical Learning Algorithm
//!
//! The contents of this module are primarily an implementation of the
//! algorithm described in
//! [Numenta's paper from 2011](http://numenta.org/resources/HTM_CorticalLearningAlgorithms.pdf).
//! The main brick is the CLA Region, which includes both a pattern
//! and a transition analysis.

mod pattern;
mod transition;

use topology::Topology;
use Pooling;

pub use self::pattern::{PatternMemory, PatternMemoryConfig};
pub use self::transition::{TransitionMemory, TransitionMemoryConfig};

/// A CLA Region
///
/// It includes a Pattern Memory and a Transition Memory, and will apply both sequentially to all
/// input you provide it.
pub struct Region<T: Topology> {
    pattern: PatternMemory<T>,
    transition: TransitionMemory
}

impl<T: Topology> Region<T> {
    /// Creates a new Region with given parameters for its internal Pattern and Trasition Memories.
    pub fn new(input_size: usize, columns_count: usize, depth: usize, topology: T, pattern_config: PatternMemoryConfig, transition_config: TransitionMemoryConfig) -> Region<T> {
        Region {
            pattern: PatternMemory::new(input_size, columns_count, topology, pattern_config),
            transition: TransitionMemory::new(columns_count, depth, transition_config)
        }
    }
}

impl<T: Topology> Pooling for Region<T> {
    fn pool(&mut self, inputs: &[usize]) -> Vec<usize> {
        let active_columns = self.pattern.pool(inputs);
        self.transition.pool(&active_columns)
    }

    fn pool_train(&mut self, inputs: &[usize]) -> Vec<usize> {
        let active_columns = self.pattern.pool_train(inputs);
        self.transition.pool_train(&active_columns)
    }

    fn anomaly(&self) -> f64 {
        // TODO: integrate an anomaly value from the PatternMemory
        self.transition.anomaly()
    }
}