//! Types and traits for handling the layers topology.

use std::ops::Range;

/// Common interface for topology of neuron organizations.
///
/// This trait is used to make the mapping between the inputs indexes to
/// their location in a topology of layer.
pub trait Topology {
    /// Type of the iterator over neighbors returned by `.neighbors(...)`.
    type NeighborsIter: Iterator<Item=usize>;

    /// Neighbors of node `i` within a given radius.
    ///
    /// Returns an iterator over the neighbors indexes.
    fn neighbors(&self, i: usize, radius: f64) -> Self::NeighborsIter;

    /// Radius of the sphere containing given nodes.
    ///
    /// Returns the minimmum radius of the sphere of given center containing all
    /// nodes yielded by the provided iterator.
    fn radius<I: IntoIterator<Item=usize>>(&self, center: usize, nodes: I) -> f64;
}

/// Single dimension topology
///
/// All inputs are arganized as a single line, the simplest topology.
pub struct OneDimension {
    length: usize,
}

impl OneDimension {
    /// Create a new `OneDimension` topology.
    ///
    /// It represents a line of `length` inputs, with indexes
    /// ranging from `0` to `length-1` included.
    pub fn new(length: usize) -> OneDimension {
        OneDimension { length: length }
    }
}

impl Topology for OneDimension {
    type NeighborsIter = Range<usize>;
    fn neighbors(&self, i: usize, radius: f64) -> Range<usize> {
        let radius = radius.floor() as usize;
        (
         (if i > radius { i - radius } else { 0 })
         ..
         (if i > (self.length - radius) { self.length } else { i + radius })
        )
    }

    fn radius<I: IntoIterator<Item=usize>>(&self, center: usize, nodes: I) -> f64 {
        nodes.into_iter()
             .map(|i| if i > center { i - center } else { center -i })
             .fold(0, |m, i| if i > m { i } else { m })
             as f64
    }
}