
pub trait Topology {
    fn neighbors(&self, i: usize, radius: f64) -> Vec<usize>;
    fn radius<I: IntoIterator<Item=usize>>(&self, center: usize, nodes: I) -> f64;
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

    fn radius<I: IntoIterator<Item=usize>>(&self, center: usize, nodes: I) -> f64 {
        nodes.into_iter()
             .map(|i| if i > center { i - center } else { center -i })
             .fold(0, |m, i| if i > m { i } else { m })
             as f64
    }
}