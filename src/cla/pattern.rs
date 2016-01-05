//! The Pattern Memory layer

use std::cell::Cell;
use std::rc::Rc;

use Pooling;
use topology::Topology;

struct Synapse {
    source: usize,
    destination: usize,
    permanence: Cell<f64>
}

struct Column {
    inputs: Vec<Rc<Synapse>>,
    boost: f64,
    active_duty_cycle: f64,
    overlap_duty_cycle: f64,
}

impl Column {
    fn new() -> Column {
        Column {
            inputs: Vec::new(),
            boost: 1.0,
            active_duty_cycle: 0.0,
            overlap_duty_cycle: 0.0
        }
    }
}

/// Parameters of a PatternMemory layer.
#[derive(Copy,Clone)]
pub struct PatternMemoryConfig {
    /// The permanence threasold to count a synapse as connected
    ///
    /// Between `0.0` and `1.0`, an example value would be `0.2`.
    pub connected_perm: f64,
    /// The amplitude of increase when reinforcing a synapse
    ///
    /// An example value would be `0.003`
    pub permanence_inc: f64,
    /// The amplitude of decrease when weakening a synapse
    ///
    /// An example value would be `0.0005`
    pub permanence_dec: f64,
    /// The weight of a new value in the exponential sliding average of activity
    ///
    /// An example value would be `0.01` to make it count for 1/100 of the total activity
    pub sliding_average_factor: f64,
    /// The minimum overlap of a segment with the input pattern for it to count as activated
    ///
    /// An example value would be around the same order of magnitude of the expected activity of the input
    pub min_overlap: usize,
    /// The target number of active cells in a neighborhood
    ///
    /// An example value would be around 2% of the number of outputs
    pub desired_local_activity: usize,
    /// The initial deviation of permanence values around connected_perm
    ///
    /// A typical value would be `0.1`
    pub initial_dev: f64,
    /// The number of potential synapses in a segment
    ///
    /// An example value would be around a 3 or 4 times `min_overlap`
    pub proximal_segment_size: usize,
    /// The number of potential synapses in a segment
    ///
    /// An example value would be around a 3 or 4 times `min_overlap`
    pub proximal_segment_density: f64,
}

/// A layer recognising spatial patterns.
///
/// Formerly known as Spatial Pooling, this layer is state-less, and can be used
/// for spatial sampling of patterns in the input.
pub struct PatternMemory<T: Topology> {
    _input_size: usize,
    columns: Vec<Column>,
    inputs: Vec<Vec<Rc<Synapse>>>,
    topology: T,
    config: PatternMemoryConfig,
    inhibition_radius: f64

}

/*
 * Preparation
 */

impl<T: Topology> PatternMemory<T> {
    /// Create a new Pattern Memory
    ///
    /// Using given number of inputs, of outputs, the given topology and given parameters.
    ///
    /// The topology must be configured for the same number of indexes as the outputs count.
    pub fn new(input_size: usize, output_size: usize, topology: T, config: PatternMemoryConfig) -> PatternMemory<T> {
        use rand::distributions::{Normal, Sample};
        let mut normal = Normal::new(config.connected_perm, config.initial_dev);
        let mut rng = ::rand::weak_rng();

        let mut columns: Vec<_> = (0..output_size).map(|_| Column::new()).collect();
        let mut inputs = vec![Vec::new(); input_size];

        let radius = config.proximal_segment_size as f64 / 2.0;

        for c in 0..output_size {
            for i in ::rand::sample(
                    &mut rng,
                    topology.neighbors(c, radius),
                    (T::surface(radius) * config.proximal_segment_density) as usize
                ) {
                let rc = Rc::new(Synapse { source: i, destination: c, permanence: Cell::new(normal.sample(&mut rng)) });
                columns[c].inputs.push(rc.clone());
                inputs[i].push(rc.clone());
            }
        }

        PatternMemory {
            columns: columns,
            _input_size: input_size,
            inputs: inputs,
            topology: topology,
            config: config,
            inhibition_radius: radius
        }
    }
}

/*
 * Cortical Learning impl
 */

impl<T: Topology> Pooling for PatternMemory<T> {
    fn pool(&mut self, inputs: &[usize]) -> Vec<usize> {
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        self.cortical_spatial_phase_2(&overlaps)
    }

    fn pool_train(&mut self, inputs: &[usize]) -> Vec<usize> {
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        let actives = self.cortical_spatial_phase_2(&overlaps);
        // phase 3: Learning
        self.cortical_spatial_phase_3(inputs, &overlaps, &actives);
        actives
    }

    fn anomaly(&self) -> f64 { 0.0 }
}

/*
 * Spatial Pooling
 */

impl<T: Topology> PatternMemory<T> {

    fn cortical_spatial_phase_1(&self, inputs: &[usize]) -> Vec<f64> {
        let mut overlaps = vec![0f64; self.columns.len()];
        for &i in inputs {
            for s in &self.inputs[i] {
                if s.permanence.get() >= self.config.connected_perm {
                    overlaps[s.destination] += 1.;
                }
            }
        }
        let min_overlap = self.config.min_overlap as f64;
        for ((i, o), b) in overlaps.iter_mut().enumerate().zip(self.columns.iter().map(|c| c.boost)) {
            if *o >= min_overlap {
                *o *= b;
            } else {
                *o = 0.0;
            }
        }
        overlaps
    }

    fn cortical_spatial_phase_2(&self, overlaps: &[f64]) -> Vec<usize> {
        /*
        overlaps.iter().enumerate().filter_map( |(i, &o)| {
            let rank = self.topology.neighbors(i, self.inhibition_radius)
                                        .map(|j| (overlaps[j] >= o) as usize).fold(0, ::std::ops::Add::add);
            if rank < self.config.desired_local_activity && o >= 1.0 {
                Some(i)
            } else {
                None
            }
        }).collect()
        */
        use std::cmp::{PartialOrd, Ordering};
        let mut cols: Vec<(usize,f64)> = overlaps.iter().cloned().enumerate().collect();
        cols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Less));
        cols.into_iter().flat_map(|(i, o)| if o >= 1.0 { Some(i) } else { None }).take(self.config.desired_local_activity).collect()
    }

    fn cortical_spatial_phase_3(&mut self, inputs: &[usize], overlaps: &[f64], actives: &[usize]) {
        for &i in actives.iter() {
            let col = &self.columns[i];
            for s in &col.inputs {
                let mut p = s.permanence.get();
                if inputs.contains(&s.source) {
                    p += self.config.permanence_inc;
                    if p > 1.0 { p = 1.0 }
                } else {
                    p -= self.config.permanence_dec;
                    if p < 0.0 { p = 0.0 }
                }
                s.permanence.set(p);
            }
        }

        let min_duty_cycles = (0..self.columns.len()).map(|i| {
            self.topology.neighbors(i, self.inhibition_radius)
                         .map(|j| self.columns[j].active_duty_cycle)
                         .fold(0.0, |a, b| if a > b { a } else { b })
             * 0.01
        }).collect::<Vec<_>>();

        let alpha = self.config.sliding_average_factor;
        for (((i, c), min_duty_cycle), overlap) in self.columns.iter_mut().enumerate().zip(min_duty_cycles.into_iter()).zip(overlaps.iter()) {
            c.active_duty_cycle *= (1.0 - alpha) * c.active_duty_cycle;
            if actives.contains(&i) { c.active_duty_cycle += alpha }
            c.boost = if c.active_duty_cycle >= min_duty_cycle { 1.0 } else { min_duty_cycle / c.active_duty_cycle };

            c.overlap_duty_cycle *= (1.0 - alpha) * c.overlap_duty_cycle;
            if *overlap >= self.config.connected_perm { c.overlap_duty_cycle += alpha }
            if c.overlap_duty_cycle < min_duty_cycle {
                let factor = 1.0 + 0.1 * self.config.connected_perm;
                for s in &mut c.inputs {
                    let mut p = s.permanence.get();
                    p *= factor;
                    if p > 1.0 { p = 1.0 }
                    s.permanence.set(p);
                }
            }
        }

        let cperm = self.config.connected_perm;

        self.inhibition_radius = self.columns.iter().enumerate().map(|(i, c)| {
            self.topology.radius(i, c.inputs.iter().flat_map(
                |s| if s.permanence.get() >= cperm { Some(s.source) } else { None }
            ))
        }).fold(0., ::std::ops::Add::add) / (self.columns.len() as f64);
    }
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use test::Bencher;
    use super::{PatternMemoryConfig, PatternMemory};
    use topology::OneDimension;
    use Pooling;

    static INPUT_SIZE: usize = 1024;
    static COL_COUNT: usize = 2048;

    #[bench]
    fn bench_pool(b: &mut Bencher) {
        let mut rng = ::rand::weak_rng();
        let input = (0..8).map(|_|
            ::rand::sample(&mut rng, 0..INPUT_SIZE, INPUT_SIZE/50)
        ).collect::<Vec<_>>();

        let mut pooler = PatternMemory::new(
            INPUT_SIZE,
            COL_COUNT,
            OneDimension::new(COL_COUNT),
            PatternMemoryConfig {
                connected_perm: 0.2,
                permanence_inc: 0.003,
                permanence_dec: 0.0005,
                sliding_average_factor: 0.01,
                min_overlap: 1,
                desired_local_activity: 40,
                initial_dev: 0.1,
                proximal_segment_size: 16,
            }
        );

        let mut i = 0;

        b.bench_n(3, |b| b.iter(|| { pooler.pool(&input[i]); i = (i+1) % 8; }));
    }

    #[bench]
    fn bench_train(b: &mut Bencher) {
        let mut rng = ::rand::weak_rng();
        let input = (0..8).map(|_|
            ::rand::sample(&mut rng, 0..INPUT_SIZE, INPUT_SIZE/50)
        ).collect::<Vec<_>>();

        let mut pooler = PatternMemory::new(
            INPUT_SIZE,
            COL_COUNT,
            OneDimension::new(COL_COUNT),
            PatternMemoryConfig {
                connected_perm: 0.2,
                permanence_inc: 0.003,
                permanence_dec: 0.0005,
                sliding_average_factor: 0.01,
                min_overlap: 1,
                desired_local_activity: 40,
                initial_dev: 0.1,
                proximal_segment_size: 16,
            }
        );

        let mut i = 0;

        b.bench_n(3, |b| b.iter(|| { pooler.pool_train(&input[i]); i = (i+1) % 8; }));
    }
}
