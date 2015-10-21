extern crate itertools;
extern crate rand;

pub mod region;
pub mod topology;

pub trait Pooling {
    fn pool(&mut self, inputs: &[bool]) -> Vec<bool>;
    fn pool_train(&mut self, input: &[bool]) -> Vec<bool>;
}
