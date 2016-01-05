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

    fn surface(radius: f64) -> f64;
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
         (if (i + radius) > self.length { self.length } else { i + radius })
        )
    }

    fn radius<I: IntoIterator<Item=usize>>(&self, center: usize, nodes: I) -> f64 {
        nodes.into_iter()
             .map(|i| if i > center { i - center } else { center -i })
             .fold(0, |m, i| if i > m { i } else { m })
             as f64
    }

    fn surface(r: f64) -> f64 { 2.0*r }
}


/// A double dimension topology
///
/// Inputs are organized as a rectangle
pub struct TwoDimensions {
    width: usize,
    height: usize
}

/// Iterator over the neighbors of a 2D point.
pub struct TwoDNIter {
    width: f64,
    height: f64,
    point: (f64, f64),
    center: (f64, f64),
    radius2: f64,
    max_x: f64,
    y_offset: f64,
}

fn dist(a: (usize, usize), b: (usize, usize)) -> f64 {
    let dx = if a.0 > b.0 { a.0 - b.0 } else { b.0 - a.0 };
    let dy = if a.1 > b.1 { a.1 - b.1 } else { b.1 - a.1 };
    ((dx*dx + dy*dy) as f64).sqrt()
}

impl Iterator for TwoDNIter {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.point.0 > self.max_x || self.point.0 >= self.width {
            let new_y = {
                // go to next row
                self.point.1 + 1.0
            };
            let dx = self.radius2 - (new_y - self.center.1).powi(2);
            if dx < 0.0 || new_y >= self.height {
                // finished
                return None
            }
            let dx = dx.sqrt().floor();
            if self.center.0 < dx {
                self.point = (0.0, new_y);
            } else {
                self.point = (self.center.0 - dx, new_y)
            }
            self.y_offset += self.width;
            self.max_x = self.center.0 + dx;
        }
        let oldx = self.point.0;
        self.point.0 += 1.0;
        Some((oldx + self.y_offset) as usize)
    }
}

impl TwoDimensions {
    /// Create a new `TwoDimension` topology.
    ///
    /// It represents a line of `width*height` inputs, with indexes
    /// ranging from `0` to `width*height-1` included.
    pub fn new(width: usize, height: usize) -> TwoDimensions {
        TwoDimensions {
            width: width,
            height: height
        }
    }

    fn coord_of(&self, index: usize) -> (usize, usize) {
        (index % self.width, index / self.width)
    }
}

impl Topology for TwoDimensions {
    type NeighborsIter = TwoDNIter;

    fn neighbors(&self, i: usize, radius: f64) -> TwoDNIter {
        let mut iter = TwoDNIter {
            width: self.width as f64,
            height: self.height as f64,
            point: (0.0,0.0),
            center: { let (x, y) = self.coord_of(i); (x as f64, y as f64) },
            radius2: radius.powi(2),
            max_x: 0.0,
            y_offset: 0.0
        };
        // init the iter
        let new_y = {
            let ny = iter.center.1 - radius;
            if ny <= 0.0 { 0.0 } else { ny.ceil() }
        };
        let dy = iter.radius2 - (new_y - iter.center.1).powi(2);
        assert!(dy >= 0.0);
        let dy = dy.sqrt().floor();
        if iter.center.0 < dy {
            iter.point = (0.0, new_y);
        } else {
            iter.point = (iter.center.0 - dy, new_y)
        }
        iter.max_x = iter.center.0 + dy;
        iter.y_offset = iter.point.1 * iter.width;
        // return it
        iter
    }

    fn radius<I: IntoIterator<Item=usize>>(&self, center: usize, nodes: I) -> f64 {
        let c = self.coord_of(center);
        nodes.into_iter()
             .map(|i| {
                let p = self.coord_of(i);
                dist(c, p)
             })
             .fold(0f64, |m, i| if i > m { i } else { m })
    }

    fn surface(r: f64) -> f64 { ::std::f64::consts::PI * r * r }
}


#[cfg(all(test, feature = "nightly"))]
mod benches {
    use test::Bencher;
    use super::{TwoDimensions, Topology};
    #[bench]
    fn neigh_iter_2d(b: &mut Bencher) {
        b.iter(|| {
            let t = TwoDimensions::new(64, 32);
            t.neighbors(256, 42.5).count()
        });
    }
}