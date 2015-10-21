use std::ops::Range;

use itertools::Itertools;

use rand::Rng;

use CorticalLearning;
use topology::Topology;

struct Synapse {
    source: usize,
    permanence: f64
}

struct Segment {
    synapses: Vec<Synapse>,
    sequence: bool,
}

impl Segment {
    pub fn new<R: Rng>(size: usize, sources: Range<usize>, mean: f64, dev: f64, rng: &mut R) -> Segment {
        use rand::distributions::{Normal, Range, Sample};
        let mut range = Range::new(sources.start, sources.end);
        let mut normal = Normal::new(mean, dev);
        let s = (0..size).map(|_| {
            Synapse {                source: range.sample(rng),
                permanence: normal.sample(rng)
            }
        }).collect();
        Segment { synapses: s, sequence: false }
    }

    fn activity(&self, values: &[bool], perm_thresold: f64) -> usize {
        self.synapses.iter()
                     .map(|s| (s.permanence >= perm_thresold && values[s.source]) as usize)
                     .fold(0usize, ::std::ops::Add::add)
    }

    fn raw_activity(&self, values: &[bool]) -> usize {
        self.synapses.iter()
                     .map(|s| values[s.source] as usize)
                     .fold(0usize, ::std::ops::Add::add)
    }

    fn active(&self, values: &[bool], perm_thresold: f64, act_thresold: usize) -> bool {
        self.activity(values, perm_thresold) >= act_thresold
    }
}

struct SegmentUpdate {
    index: Option<usize>,
    actives: Vec<usize>,
    news: Vec<usize>,
    sequence: bool
}

struct Cell {
    segments: Vec<Segment>,
    active: bool,
    predictive: bool,
    learning: bool,
    update_list: Vec<SegmentUpdate>
}

impl Cell {
    fn new<R: Rng>(segment_count: usize, segment_size: usize, sources: Range<usize>, mean: f64, dev: f64, rng: &mut R) -> Cell {
        Cell {
            segments: (0..segment_count).map(|_| Segment::new(segment_size, sources.clone(), mean, dev, rng)).collect(),
            active: false,
            predictive: false,
            learning: false,
            update_list: Vec::new()
        }
    }

    fn add_update<R: Rng>(&mut self, index: Option<usize>, values: &[bool], learnings: &[bool], perm_thresold: f64, new_synapses: usize, sequence: bool, rng: &mut R){
        let actives = if let Some(sindex) = index {
            self.segments[sindex].synapses.iter().enumerate()
                                   .filter_map(|(i, s)| if s.permanence >= perm_thresold && values[s.source] { Some(i) } else { None })
                                   .collect::<Vec<_>>()
        } else { Vec::new() };
        let news = if new_synapses > actives.len() {
            let potential = learnings.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }).collect::<Vec<_>>();
            ::rand::sample(rng, potential.iter().cloned(), new_synapses - actives.len())
        } else { Vec::new() };

        self.update_list.push(SegmentUpdate {
            index: index,
            actives: actives,
            news: news,
            sequence: sequence
        });
    }

    fn process_updates(&mut self, positive: bool, inc: f64, dec: f64, init: f64) {
        for seg in &mut self.segments { seg.sequence = false; }
        for su in &self.update_list {
            if let Some(i) = su.index {
                let segment = &mut self.segments[i];
                for (i, syn) in segment.synapses.iter_mut().enumerate() {
                    if su.actives.contains(&i) {
                        if positive {
                            syn.permanence += inc;
                            if syn.permanence > 1.0 { syn.permanence = 1.0 }
                        } else {
                            syn.permanence -= dec;
                            if syn.permanence < 0.0 { syn.permanence = 0.0 }
                        }
                    } else {
                        if positive {
                            syn.permanence -= dec;
                            if syn.permanence < 0.0 { syn.permanence = 0.0 }
                        }
                    }
                }
                for &t in &su.news {
                    segment.synapses.push(Synapse { source: t, permanence: init });
                }
                segment.sequence = su.sequence;
            } else {
                self.segments.push(
                    Segment {
                        synapses: su.news.iter().map(|&u| Synapse { source: u, permanence: init }).collect(),
                        sequence: su.sequence
                    }
                );
            }
        }
    }
}

struct Column {
    cells: Vec<Cell>,
    inputs: Segment,
    boost: f64,
    active_duty_cycle: f64,
    overlap_duty_cycle: f64,
}

impl Column {
    fn new<R: Rng>(depth: usize, segment_count: usize, proximal_segment_size: usize, distal_segment_size: usize, inputs: Range<usize>, sources: Range<usize>, mean: f64, dev: f64, rng: &mut R) -> Column {
        Column {
            cells: (0..depth).map(|_| Cell::new(segment_count, distal_segment_size, sources.clone(), mean, dev, rng)).collect(),
            inputs: Segment::new(proximal_segment_size, inputs, mean, dev, rng),
            boost: 1.0,
            active_duty_cycle: 0.0,
            overlap_duty_cycle: 0.0
        }
    }
}

pub struct RegionConfig {
    input_size: usize,
    initial_perm: f64,
    connected_perm: f64,
    permanence_inc: f64,
    permanence_dec: f64,
    sliding_average_factor: f64,
    min_overlap: usize,
    desired_local_activity: usize,
    activation_thresold: usize,
    learning_thresold: usize,
    new_synapses: usize,
    initial_dev: f64,
    initial_segment_count: usize,
    initial_proximal_segment_size: usize,
    initial_distal_segment_size: usize
}

pub struct Region<T: Topology> {
    columns: Vec<Column>,
    topology: T,
    depth: usize,
    config: RegionConfig,
    inhibition_radius: f64
}

/*
 * Preparation
 */

impl<T: Topology> Region<T> {
    pub fn new(columns: usize, depth: usize, topology: T, config: RegionConfig) -> Region<T> {
        let mut rng = ::rand::thread_rng();
        Region {
            columns: (0..columns).map(|_|
                Column::new(depth,
                            config.initial_segment_count,
                            config.initial_proximal_segment_size,
                            config.initial_distal_segment_size,
                            0..config.input_size,
                            0..(depth*columns),
                            config.connected_perm,
                            config.initial_dev,
                            &mut rng)
                ).collect(),
            topology: topology,
            depth: depth,
            config: config,
            inhibition_radius: 1.0
        }
    }
}

/*
 * Cortical Learning impl
 */
impl<T: Topology> CorticalLearning for Region<T> {
    fn spatial_pool(&self, inputs: &[bool]) -> Vec<bool> {
        assert!(inputs.len() == self.config.input_size,
            "Number of inputs provided did not match number of inputs of the region.");
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        self.cortical_spatial_phase_2(&overlaps)
    }

    fn spatial_pool_train(&mut self, inputs: &[bool]) -> Vec<bool> {
        assert!(inputs.len() == self.config.input_size,
            "Number of inputs provided did not match number of inputs of the region.");
        // phase 1: Overlaps
        let overlaps = self.cortical_spatial_phase_1(inputs);
        // phase 2: Inhibition
        let actives = self.cortical_spatial_phase_2(&overlaps);
        // phase 3: Learning
        self.cortical_spatial_phase_3(inputs, &overlaps, &actives);
        actives
    }

    fn temporal_pool(&mut self, active_cols: &[bool]) -> Vec<bool> {
        let active_cells = self.dump_active_cells_and_reset();
        self.cortical_temporal_phase_1(active_cols, &active_cells);
        self.cortical_temporal_phase_2(&active_cells);
        self.columns.iter().map(|col| {
            col.cells.iter().any(|cell| cell.active || cell.predictive)
        }).collect()
    }

    fn temporal_pool_train(&mut self, active_cols: &[bool]) -> Vec<bool> {
        let active_cells = self.dump_active_cells_and_reset();
        let predictive_cells = self.dump_predictive_cells_and_reset();
        let learning_cells = self.dump_learning_cells_and_reset();
        let mut rng = ::rand::thread_rng();
        self.cortical_temporal_phase_learning_1(active_cols, &active_cells, &learning_cells, &mut rng);
        self.cortical_temporal_phase_learning_2(&active_cells, &learning_cells, &mut rng);
        self.cortical_temporal_phase_learning_3(&predictive_cells);
        self.columns.iter().map(|col| {
            col.cells.iter().any(|cell| cell.active || cell.predictive)
        }).collect()
    }
}

/*
 * Spatial Pooling
 */

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

        self.inhibition_radius = self.columns.iter().enumerate().map(|(i, c)| {
            self.topology.radius(i, c.inputs.synapses.iter().map(|s| s.source))
        }).fold(0., ::std::ops::Add::add) / (self.columns.len() as f64);
    }
}

/*
 * Temporal Pooling
 */

impl<T: Topology> Region<T> {
    fn dump_active_cells_and_reset(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| { let a = c.active; c.active = false; a }).collect()
    }

    fn dump_predictive_cells_and_reset(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| { let l = c.predictive; c.predictive = false; l }).collect()
    }

    fn dump_learning_cells_and_reset(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| { let l = c.learning; c.learning = false; l }).collect()
    }

    fn cortical_temporal_phase_1(&mut self, active_cols: &[bool], active_cells: &[bool]) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        for col in self.columns.iter_mut().zip(active_cols.iter()).filter_map(|(c, a)| if *a { Some(c) } else { None }) {
            let mut predicted = false;
            for cell in col.cells.iter_mut() {
                if !cell.predictive { continue; }
                if !cell.segments.iter()
                                 .any(|s|
                                    s.sequence &&
                                    s.active(active_cells, connected_perm, activation_thresold)
                                 )
                {
                    continue;
                }
                predicted = true;
                cell.active = true;
            }
            if predicted { continue; }
            for cell in col.cells.iter_mut() {
                cell.active = true;
            }
        }
    }

    fn cortical_temporal_phase_learning_1<R: Rng>(&mut self, active_cols: &[bool], active_cells: &[bool], learning_cells: &[bool], rng: &mut R) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        let learning_thresold = self.config.learning_thresold;
        let new_synapses = self.config.new_synapses;
        for col in self.columns.iter_mut().zip(active_cols.iter()).filter_map(|(c, a)| if *a { Some(c) } else { None }) {
            let mut predicted = false;
            let mut chosen = false;
            for cell in col.cells.iter_mut() {
                if !cell.predictive { continue; }
                if let Some((i, _)) = {
                    let mut v = cell.segments.iter().enumerate()
                                 .map(|(i, s)| (i, s.activity(active_cells, connected_perm), s.sequence))
                                 .filter(|&(_, s, seq)| s >= activation_thresold && seq)
                                 .map(|(i, s, _)| (i,s))
                                 .collect::<Vec<_>>();
                    v.sort_by(|&(_, ref a), &(_, ref b)| b.cmp(a));
                    v.get(0).cloned()
                } {
                    predicted = true;
                    cell.active = true;
                    if cell.segments[i].active(learning_cells, connected_perm, activation_thresold) {
                        chosen = true;
                        cell.learning = true;
                    }
                }
            }
            if !predicted {
                for cell in col.cells.iter_mut() {
                    cell.active = true;
                }
            }
            if !chosen {
                let (opt_sindex, cindex) = col.cells.iter().enumerate().filter_map(|(ic, c)| {
                    let opt = c.segments.iter().map(|s| s.raw_activity(active_cells))
                                     .enumerate()
                                     .filter(|&(_, a)| a >= learning_thresold)
                                     .fold1(|(i1, a1), (i2, a2)| if a1 > a2 { (i1, a1) } else { (i2, a2) });
                    opt.map(|(i, a)| (i, a, ic))
                }).fold1(|(i1, a1, c1), (i2, a2, c2)| if a1 > a2 { (i1, a1, c1) } else { (i2, a2, c2) })
                  .map(|(i, _, c)| (Some(i), c))
                  .unwrap_or_else(|| {
                    col.cells.iter().map(|c| c.segments.len()).enumerate()
                        .fold1(|(l1, c1), (l2, c2)| if l1 < l2 { (l1, c1) } else { (l2, c2) } )
                        .map(|(_, c)| (None, c))
                        .unwrap()
                });
                col.cells[cindex].learning = true;
                col.cells[cindex].add_update(opt_sindex, active_cells, learning_cells, connected_perm, new_synapses, true, rng);
            }
        }
    }

    fn cortical_temporal_phase_2(&mut self, active_cells: &[bool]) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        for col in self.columns.iter_mut() {
        for cell in col.cells.iter_mut() {
        for s in cell.segments.iter() {
            if s.active(active_cells, connected_perm, activation_thresold) {
                cell.predictive = true;
            }
        }}}
    }

    fn cortical_temporal_phase_learning_2<R: Rng>(&mut self, active_cells: &[bool], training_cells: &[bool], rng: &mut R) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        let new_synapses = self.config.new_synapses;
        let learning_thresold = self.config.learning_thresold;
        for col in self.columns.iter_mut() {
        for cell in col.cells.iter_mut() {
        let active_segments = cell.segments.iter().enumerate()
                                   .filter_map(|(si, s)| if s.active(active_cells, connected_perm, activation_thresold) { Some(si) } else { None })
                                   .collect::<Vec<_>>();
        for si in active_segments {
                cell.predictive = true;
                cell.add_update(Some(si), active_cells, &[], connected_perm, 0, false, rng);

                let opt = cell.segments.iter().map(|s| s.raw_activity(active_cells))
                                     .enumerate()
                                     .filter(|&(_, a)| a >= learning_thresold)
                                     .fold1(|(i1, a1), (i2, a2)| if a1 > a2 { (i1, a1) } else { (i2, a2) })
                                     .map(|(i, _)| i);
                cell.add_update(opt, active_cells, training_cells, connected_perm, new_synapses, false, rng);
        }}}
    }

    fn cortical_temporal_phase_learning_3(&mut self, predictive_cells: &[bool]) {
        let inc = self.config.permanence_inc;
        let dec = self.config.permanence_dec;
        let init = self.config.initial_perm;
        for (coli, col) in self.columns.iter_mut().enumerate() {
        for (celli, cell) in col.cells.iter_mut().enumerate() {
            if cell.learning {
                cell.process_updates(true, inc, dec, init)
            } else if (!cell.predictive) && predictive_cells[coli * self.depth + celli]{
                cell.process_updates(false, inc, dec, init)
            }
            cell.update_list.clear();
        }}
    }
}