#![cfg_attr(feature = "nightly", feature(test))]
#[cfg(feature = "nightly")]
extern crate test;

extern crate itertools;
extern crate rand;

pub mod region;
pub mod topology;

pub trait Pooling {
    fn pool(&mut self, inputs: &[usize]) -> Vec<usize>;
    fn pool_train(&mut self, input: &[usize]) -> Vec<usize>;
    fn anomaly(&self) -> f64;
}
