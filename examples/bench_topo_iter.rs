#![feature(test)]

extern crate htm;
extern crate test;

use htm::topology::{Topology, TwoDimensions};

fn main() {
    for _ in 0..100 {
        let t = TwoDimensions::new(64,32);
        for i in t.neighbors(45, 3.5) {
            test::black_box(i);
        }
    }
}