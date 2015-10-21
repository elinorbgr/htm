use std::ops::Range;

use itertools::Itertools;

use rand::Rng;

use Pooling;
use topology::Topology;

struct Synapse {
    source: usize,
    permanence: f64
}

struct Column {
    inputs: Vec<Synapse>,
    boost: f64,
    active_duty_cycle: f64,
    overlap_duty_cycle: f64,
}

impl Column {
    fn new<R: Rng>(proximal_segment_size: usize, inputs: Range<usize>, mean: f64, dev: f64, rng: &mut R) -> Column {
        use rand::distributions::{Normal, Range, Sample};
        let mut range = Range::new(inputs.start, inputs.end);
        let mut normal = Normal::new(mean, dev);
        let s = (0..proximal_segment_size).map(|_| {
            Synapse {
                source: range.sample(rng),
                permanence: normal.sample(rng)
            }
        }).collect();

        Column {
            inputs: s,
            boost: 1.0,
            active_duty_cycle: 0.0,
            overlap_duty_cycle: 0.0
        }
    }
}

pub struct SpatialPoolerConfig {
    input_size: usize,
    connected_perm: f64,
    permanence_inc: f64,
    permanence_dec: f64,
    sliding_average_factor: f64,
    min_overlap: usize,
    desired_local_activity: usize,
    initial_dev: f64,
    initial_proximal_segment_size: usize,
}

pub struct SpatialPooler<T: Topology> {
    columns: Vec<Column>,
    topology: T,
    config: SpatialPoolerConfig,
    inhibition_radius: f64
}

/*
 * Preparation
 */

impl<T: Topology> SpatialPooler<T> {
    pub fn new(columns: usize, topology: T, config: SpatialPoolerConfig) -> SpatialPooler<T> {
        let mut rng = ::rand::thread_rng();
        SpatialPooler {
            columns: (0..columns).map(|_|
                Column::new(config.initial_proximal_segment_size,
                            0..config.input_size,
                            config.connected_perm,
                            config.initial_dev,
                            &mut rng)
                ).collect(),
            topology: topology,
            config: config,
            inhibition_radius: 1.0
        }
    }
}

/*
 * Cortical Learning impl
 */
impl<T: Topology> Pooling for SpatialPooler<T> {
    fn pool(&mut self, inputs: &[bool]) -> Vec<bool> {
        assert!(inputs.len() == self.config.input_size,
            "Number of inputs provided did not match number of inputs of the SpatialPooler.");
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        self.cortical_spatial_phase_2(&overlaps)
    }

    fn pool_train(&mut self, inputs: &[bool]) -> Vec<bool> {
        assert!(inputs.len() == self.config.input_size,
            "Number of inputs provided did not match number of inputs of the SpatialPooler.");
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        let actives = self.cortical_spatial_phase_2(&overlaps);
        // phase 3: Learning
        self.cortical_spatial_phase_3(inputs, &overlaps, &actives);
        actives
    }
}

/*
 * Spatial Pooling
 */

impl<T: Topology> SpatialPooler<T> {
    fn cortical_spatial_phase_1(&self, inputs: &[bool]) -> Vec<f64> {
        self.columns.iter().map( |c| {
            let o = c.inputs.iter().map(|s|
                if s.permanence >= self.config.connected_perm { inputs[s.source] } else { false } as usize
            ).fold(0, ::std::ops::Add::add);
            if o >= self.config.min_overlap { o as f64 * c.boost } else { 0.0 }
        }).collect()
    }

    fn cortical_spatial_phase_2(&self, overlaps: &[f64]) -> Vec<bool> {
        overlaps.iter().enumerate().map( |(i, o)| {
            let mut acts = self.topology.neighbors(i, self.inhibition_radius)
                                        .into_iter()
                                        .map(|j| overlaps[j]).collect::<Vec<_>>();
            acts.sort_by(|a,b| ::std::cmp::PartialOrd::partial_cmp(b,a).unwrap_or(::std::cmp::Ordering::Less));
            acts.get(self.config.desired_local_activity - 1)
                .map(|a| { *o > 0.0 && *o > *a }).unwrap_or(false)
        }).collect()
    }

    fn cortical_spatial_phase_3(&mut self, inputs: &[bool], overlaps: &[f64], actives: &[bool]) {
        for c in self.columns.iter_mut() {
            for s in &mut c.inputs {
                if inputs[s.source] {
                    s.permanence += self.config.permanence_inc;
                    if s.permanence > 1.0 { s.permanence = 1.0 }
                } else {
                    s.permanence -= self.config.permanence_dec;
                    if s.permanence < 0.0 { s.permanence = 0.0 }
                }
            }
        }

        let min_duty_cycles = (0..self.columns.len()).map(|i| {
            self.topology.neighbors(i, self.inhibition_radius)
                         .into_iter()
                         .map(|j| self.columns[j].active_duty_cycle)
                         .fold(0.0, |a, b| if a > b { a } else { b })
             * 0.01
        }).collect::<Vec<_>>();

        let alpha = self.config.sliding_average_factor;
        for (((c, min_duty_cycle), overlap), active) in self.columns.iter_mut().zip(min_duty_cycles.into_iter()).zip(overlaps.iter()).zip(actives.iter()) {
            c.active_duty_cycle *= (1.0 - alpha) * c.active_duty_cycle;
            if *active { c.active_duty_cycle += alpha }
            c.boost = if c.active_duty_cycle >= min_duty_cycle { 1.0 } else { min_duty_cycle / c.active_duty_cycle };

            c.overlap_duty_cycle *= (1.0 - alpha) * c.overlap_duty_cycle;
            if *overlap >= self.config.connected_perm { c.overlap_duty_cycle += alpha }
            if c.overlap_duty_cycle < min_duty_cycle {
                let factor = 1.0 + 0.1 * self.config.connected_perm;
                for s in &mut c.inputs {
                    s.permanence *= factor;
                    if s.permanence > 1.0 { s.permanence = 1.0 }
                }
            }
        }

        self.inhibition_radius = self.columns.iter().enumerate().map(|(i, c)| {
            self.topology.radius(i, c.inputs.iter().map(|s| s.source))
        }).fold(0., ::std::ops::Add::add) / (self.columns.len() as f64);
    }
}