
pub trait Topology {
    fn neighbors(&self, i: usize, radius: f64) -> Vec<usize>;
    fn radius<I: IntoIterator<Item=usize>>(&self, nodes: I) -> f64;
}

pub struct OneDimension {
    length: usize,
}

impl OneDimension {
    pub fn new(length: usize) -> OneDimension {
        OneDimension { length: length }
    }
}

impl Topology for OneDimension {
    fn neighbors(&self, i: usize, radius: f64) -> Vec<usize> {
        let radius = radius.floor() as usize;
        (
         (if i > radius { i - radius } else { 0 })
         ..
         (if i > (self.length - radius) { self.length } else { i + radius })
        ).collect()
    }

    fn radius<I: IntoIterator<Item=usize>>(&self, nodes: I) -> f64 {
        let (min, max) = nodes.into_iter().fold((self.length, 0), |(min, max), i| {
            (if i < min { i } else { min }, if i > max { i } else { max })
        });
        (max - min) as f64 / 2.0
    }
}