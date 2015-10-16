use std::ops::Range;

use rand::Rng;

use CorticalLearning;
use topology::Topology;

struct Synapse {
    active: bool,
    source: usize,
    permanence: f64
}

struct Segment {
    synapses: Vec<Synapse>
}

impl Segment {
    pub fn new<R: Rng>(size: usize, sources: Range<usize>, mean: f64, dev: f64, rng: &mut R) -> Segment {
        use rand::distributions::{Normal, Range, Sample};
        let mut range = Range::new(sources.start, sources.end);
        let mut normal = Normal::new(mean, dev);
        let s = (0..size).map(|_| {
            Synapse {
                active: false,
                source: range.sample(rng),
                permanence: normal.sample(rng)
            }
        }).collect();
        Segment { synapses: s }
    }
}

struct Cell {
    active: bool,
    predictive: bool,
    segments: Vec<Segment>
}

struct Column {
    cells: Vec<Cell>,
    inputs: Segment,
    boost: f64,
    active_duty_cycle: f64,
    overlap_duty_cycle: f64,
}

struct RegionConfig {
    input_size: usize,
    connected_perm: f64,
    permanence_inc: f64,
    permanence_dec: f64,
    sliding_average_factor: f64,
    min_overlap: usize,
    desired_local_activity: usize,
}

pub struct Region<T: Topology> {
    columns: Vec<Column>,
    topology: T,
    config: RegionConfig,
    inhibition_radius: f64
}

/*
 * Cortical Learning impl
 */
impl<T: Topology> CorticalLearning for Region<T> {
    fn spatial_pool(&self, inputs: &[bool]) -> Vec<bool> {
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        self.cortical_spatial_phase_2(&overlaps)
    }

    fn spatial_pool_train(&mut self, inputs: &[bool]) -> Vec<bool> {
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        let actives = self.cortical_spatial_phase_2(&overlaps);
        // phase 3: Learning
        self.cortical_spatial_phase_3(inputs, &overlaps, &actives);
        actives
    }

    fn temporal_pool(&self, _columns: &[bool]) -> Vec<bool> {
        unimplemented!()
    }

    fn temporal_pool_train(&mut self, _columns: &[bool]) -> Vec<bool> {
        unimplemented!()
    }
}

impl<T: Topology> Region<T> {
    fn cortical_spatial_phase_1(&self, inputs: &[bool]) -> Vec<f64> {
        self.columns.iter().map( |c| {
            let o = c.inputs.synapses.iter().map(|s|
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
            for s in &mut c.inputs.synapses {
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
                for s in &mut c.inputs.synapses {
                    s.permanence *= factor;
                    if s.permanence > 1.0 { s.permanence = 1.0 }
                }
            }
        }

        // TODO: update inhibition_radius
        self.inhibition_radius = self.columns.iter().map(|c| {
            self.topology.radius(c.inputs.synapses.iter().map(|s| s.source))
        }).fold(0., ::std::ops::Add::add) / (self.columns.len() as f64);
    }
}