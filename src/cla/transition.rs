//! The Transition Memory layer

use std::ops::Range;

use itertools::Itertools;

use rand::Rng;

use Pooling;

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
            ::rand::sample(
                rng,
                learnings.iter().enumerate().filter_map(|(i, &b)| if b { Some(i) } else { None }),
                new_synapses - actives.len()
            )
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
}

impl Column {
    fn new<R: Rng>(depth: usize, segment_count: usize, distal_segment_size: usize, sources: Range<usize>, mean: f64, dev: f64, rng: &mut R) -> Column {
        Column {
            cells: (0..depth).map(|_| Cell::new(segment_count, distal_segment_size, sources.clone(), mean, dev, rng)).collect(),
        }
    }
}

/// Parameters of a TransitionMemory layer.
pub struct TransitionMemoryConfig {
    /// The initial permanence value of newly created synapses
    ///
    /// An example value would be `0.21`
    pub initial_perm: f64,
    /// The thresold in permanence for a synapse to be considered as connected
    ///
    /// Between `0.0` and `1.0`, an example value would be `0.2`
    pub connected_perm: f64,
    /// The amplitude of increase when reinforcing a synapse
    ///
    /// An example value would be `0.1`
    pub permanence_inc: f64,
    /// The amplitude of decrease when weakening a synapse
    ///
    /// An example value would be `0.1`
    pub permanence_dec: f64,
    /// The thresold of activation for a segment to be activated, in number of active synapses
    pub activation_thresold: usize,
    /// The thresold of raw activity on a segment for it to be picked for learning
    pub learning_thresold: usize,
    /// The maximum number of synapses to create when growing a segment
    pub new_synapses: usize,
    /// The initial deviation of permanence values around connected_perm
    ///
    /// A typical value would be `0.1`
    pub initial_dev: f64,
    /// The initial number of segments for each neuron
    pub initial_segment_count: usize,
    /// The initial number of potential synapses in a segment
    pub initial_distal_segment_size: usize
}

/// A layer recognising temporal patterns.
///
/// Formerly known as Temporal Pooling, this layer is state-full, and will 
/// link each input to the previously seen ones.
///
/// The output is of the same size of the input, and is the input mixed with a
/// prediction of the next expected input.
pub struct TransitionMemory {
    columns: Vec<Column>,
    depth: usize,
    config: TransitionMemoryConfig,
    last_anomaly: f64,
}

/*
 * Preparation
 */

impl TransitionMemory {
    /// Create a new Transition Memory
    ///
    /// Using given number of inputs/outputs, the given depth and given parameters.
    pub fn new(columns: usize, depth: usize, config: TransitionMemoryConfig) -> TransitionMemory {
        let mut rng = ::rand::weak_rng();
        TransitionMemory {
            columns: (0..columns).map(|_|
                Column::new(depth,
                            config.initial_segment_count,
                            config.initial_distal_segment_size,
                            0..(depth*columns),
                            config.connected_perm,
                            config.initial_dev,
                            &mut rng)
                ).collect(),
            depth: depth,
            config: config,
            last_anomaly: 1.0
        }
    }
}

/*
 * Cortical Learning impl
 */

impl Pooling for TransitionMemory {
    fn pool(&mut self, active_cols: &[usize]) -> Vec<usize> {
        let active_cells = self.dump_active_cells_and_reset();
        let predictive_cells = self.dump_predictive_cells_and_reset();
        self.cortical_temporal_phase_1(active_cols, &active_cells, &predictive_cells);
        self.cortical_temporal_phase_2();
        self.update_anomaly(&predictive_cells);
        self.columns.iter().enumerate().filter_map(|(i, col)| {
            if col.cells.iter().any(|cell| cell.active || cell.predictive) {
                Some(i)
            } else {
                None
            }
        }).collect()
    }

    fn pool_train(&mut self, active_cols: &[usize]) -> Vec<usize> {
        let active_cells = self.dump_active_cells_and_reset();
        let predictive_cells = self.dump_predictive_cells_and_reset();
        let learning_cells = self.dump_learning_cells_and_reset();
        let mut rng = ::rand::weak_rng();
        self.cortical_temporal_phase_learning_1(active_cols, &active_cells, &learning_cells, &predictive_cells, &mut rng);
        self.cortical_temporal_phase_learning_2(&learning_cells, &mut rng);
        self.cortical_temporal_phase_learning_3(&predictive_cells);
        self.update_anomaly(&predictive_cells);
        self.columns.iter().enumerate().filter_map(|(i, col)| {
            if col.cells.iter().any(|cell| cell.active || cell.predictive) {
                Some(i)
            } else {
                None
            }
        }).collect()
    }

    fn anomaly(&self) -> f64 {
        self.last_anomaly
    }
}

/*
 * Temporal Pooling
 */

impl TransitionMemory {
    fn dump_active_cells_and_reset(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| { let a = c.active; c.active = false; a }).collect()
    }

    fn dump_active_cells(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| c.active).collect()
    }

    fn dump_predictive_cells_and_reset(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| { let l = c.predictive; c.predictive = false; l }).collect()
    }

    fn dump_learning_cells_and_reset(&mut self) -> Vec<bool> {
        self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).map(|c| { let l = c.learning; c.learning = false; l }).collect()
    }

    fn cortical_temporal_phase_1(&mut self, active_cols: &[usize], active_cells: &[bool], predictive_cells: &[bool]) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        for &coli in active_cols {
            let col = &mut self.columns[coli];
            let mut predicted = false;
            for (celli, cell) in col.cells.iter_mut().enumerate() {
                if !predictive_cells[coli*self.depth + celli] { continue; }
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

    fn cortical_temporal_phase_learning_1<R: Rng>(&mut self, active_cols: &[usize], active_cells: &[bool], learning_cells: &[bool], predictive_cells: &[bool], rng: &mut R) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        let learning_thresold = self.config.learning_thresold;
        let new_synapses = self.config.new_synapses;
        for &coli in active_cols {
            let col = &mut self.columns[coli];
            let mut predicted = false;
            let mut chosen = false;
            for (celli, cell) in col.cells.iter_mut().enumerate() {
                if !predictive_cells[coli*self.depth + celli] { continue; }
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
                        .fold1(|(c1, l1), (c2, l2)| if l1 < l2 { (c1, l1) } else { (c2, l2) } )
                        .map(|(c, _)| (None, c))
                        .unwrap()
                });
                col.cells[cindex].learning = true;
                col.cells[cindex].add_update(opt_sindex, active_cells, learning_cells, connected_perm, new_synapses, true, rng);
            }
        }
    }

    fn cortical_temporal_phase_2(&mut self) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        let active_cells = self.dump_active_cells();
        for col in self.columns.iter_mut() {
        for cell in col.cells.iter_mut() {
        for s in cell.segments.iter() {
            if s.active(&active_cells, connected_perm, activation_thresold) {
                cell.predictive = true;
            }
        }}}
    }

    fn cortical_temporal_phase_learning_2<R: Rng>(&mut self, training_cells: &[bool], rng: &mut R) {
        let connected_perm = self.config.connected_perm;
        let activation_thresold = self.config.activation_thresold;
        let new_synapses = self.config.new_synapses;
        let learning_thresold = self.config.learning_thresold;
        let active_cells = self.dump_active_cells();
        for col in self.columns.iter_mut() {
        for cell in col.cells.iter_mut() {
        let active_segments = cell.segments.iter().enumerate()
                                   .filter_map(|(si, s)| if s.active(&active_cells, connected_perm, activation_thresold) { Some(si) } else { None })
                                   .collect::<Vec<_>>();
        for si in active_segments {
                cell.predictive = true;
                cell.add_update(Some(si), &active_cells, training_cells, connected_perm, 0, false, rng);

                let opt = cell.segments.iter().map(|s| s.raw_activity(&active_cells))
                                     .enumerate()
                                     .filter(|&(_, a)| a >= learning_thresold)
                                     .fold1(|(i1, a1), (i2, a2)| if a1 > a2 { (i1, a1) } else { (i2, a2) })
                                     .map(|(i, _)| i);
                cell.add_update(opt, &active_cells, training_cells, connected_perm, new_synapses, false, rng);
        }}}
    }

    fn cortical_temporal_phase_learning_3(&mut self, predictive_cells: &[bool]) {
        let inc = self.config.permanence_inc;
        let dec = self.config.permanence_dec;
        let init = self.config.initial_perm;
        for (cell, pred) in self.columns.iter_mut().flat_map(|c| c.cells.iter_mut()).zip(predictive_cells.iter()) {
            if cell.learning {
                cell.process_updates(true, inc, dec, init)
            } else if (!cell.predictive) && *pred {
                cell.process_updates(false, inc, dec, init)
            }
            cell.update_list.clear();
        }
    }

    fn update_anomaly(&mut self, previous_predictive_cells: &[bool]) {
        let (tot_act, tot_act_pred) = self.columns.iter().map(|col| col.cells.iter().any(|c| c.active)).zip(
                previous_predictive_cells.iter().chunks_lazy(self.depth).into_iter().map(|mut col| col.any(|x| *x))
            ).fold((0usize, 0usize), |(ta, tap), (a, p)| (ta + a as usize, tap + (a ^ (a&p)) as usize));
        self.last_anomaly = tot_act_pred as f64 / tot_act as f64
    }
}

#[cfg(all(test, feature = "nightly"))]
mod benches {
    use test::Bencher;
    use super::{TransitionMemoryConfig, TransitionMemory};
    use Pooling;

    static COL_COUNT: usize = 2048;
    static DEPTH: usize = 32;

    #[bench]
    fn bench_pool(b: &mut Bencher) {
        let mut rng = ::rand::weak_rng();
        let input = (0..8).map(|_|
            ::rand::sample(&mut rng, 0..COL_COUNT, COL_COUNT/50)
        ).collect::<Vec<_>>();

        let mut pooler = TransitionMemory::new(
            COL_COUNT,
            DEPTH,
            TransitionMemoryConfig {
                initial_perm: 0.21,
                connected_perm: 0.2,
                permanence_inc: 0.1,
                permanence_dec: 0.1,
                activation_thresold: 13,
                learning_thresold: 10,
                new_synapses: 20,
                initial_dev: 0.2,
                initial_segment_count: 4,
                initial_distal_segment_size: 20
            }
        );

        let mut i = 0;

        b.bench_n(3, |b| b.iter(|| { pooler.pool(&input[i]); i = (i+1) % 8; }));
    }

    #[bench]
    fn bench_train(b: &mut Bencher) {
        let mut rng = ::rand::weak_rng();
        let input = (0..8).map(|_|
            ::rand::sample(&mut rng, 0..COL_COUNT, COL_COUNT/50)
        ).collect::<Vec<_>>();

        let mut pooler = TransitionMemory::new(
            COL_COUNT,
            DEPTH,
            TransitionMemoryConfig {
                initial_perm: 0.21,
                connected_perm: 0.2,
                permanence_inc: 0.1,
                permanence_dec: 0.1,
                activation_thresold: 13,
                learning_thresold: 10,
                new_synapses: 20,
                initial_dev: 0.2,
                initial_segment_count: 4,
                initial_distal_segment_size: 20
            }
        );

        let mut i = 0;

        b.bench_n(3, |b| b.iter(|| { pooler.pool_train(&input[i]); i = (i+1) % 8; }));
    }
}