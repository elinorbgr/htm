extern crate rand;

mod region;
pub mod topology;

pub use region::Region;

pub trait CorticalLearning {
    fn spatial_pool(&self, inputs: &[bool]) -> Vec<bool>;
    fn spatial_pool_train(&mut self, input: &[bool]) -> Vec<bool>;
    fn temporal_pool(&self, columns: &[bool]) -> Vec<bool>;
    fn temporal_pool_train(&mut self, columns: &[bool]) -> Vec<bool>;
    fn pool(&self, inputs: &[bool]) -> Vec<bool> {
        let cols = self.spatial_pool(inputs);
        self.temporal_pool(&cols)
    }
    fn pool_train(&mut self, inputs: &[bool]) -> Vec<bool> {
        let cols = self.spatial_pool_train(inputs);
        self.temporal_pool_train(&cols)
    }
}
