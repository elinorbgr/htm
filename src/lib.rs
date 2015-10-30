//! Hierarchical Temporal Memory tools
//!
//! This library contains tools to use HTM-based algorithms,
//! for machine learning and anomaly detection.
//!
//! The main algorithm implemented here is currently the Cortical
//! Learning Algorithm, from [Numenta](http://numenta.com/), and available in
//! the cla module.

#![warn(missing_docs)]

#![cfg_attr(feature = "nightly", feature(test))]
#[cfg(feature = "nightly")]
extern crate test;

extern crate itertools;
extern crate rand;

pub mod cla;
pub mod topology;

/// Pooling interface
///
/// Generic interface for poolers, working on lists of active input/outputs.
pub trait Pooling {
    /// Proceed to a non-training pool
    ///
    /// As input, a slice of indexes of active inputs. Not ordering constraints,
    /// but there should not be duplication (or results will likely be erroneous).
    ///
    /// The output is a vec containing the indexes of active outputs. No ordering is
    /// guaranteed, but there is no duplication. It is thus appropriate to be fed into
    /// an other Pooling layer.
    fn pool(&mut self, inputs: &[usize]) -> Vec<usize>;

    /// Proceed to a training pool
    ///
    /// Same inputs and outputs as `pool(...)`, but will also train the layer on this
    /// input.
    fn pool_train(&mut self, inputs: &[usize]) -> Vec<usize>;

    /// Anomaly of the last processed input.
    ///
    /// Gets the anomaly of the last input that was processed, as a floating number between
    /// `0.0` (100% expected) to `1.0` (100% unexpected).
    fn anomaly(&self) -> f64;
}
