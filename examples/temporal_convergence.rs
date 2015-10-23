extern crate rand;
extern crate htm;
extern crate itertools;

use itertools::Itertools;

use htm::Pooling;
use htm::region::temporal::{TemporalPooler, TemporalPoolerConfig};
use htm::region::spatial::{SpatialPooler, SpatialPoolerConfig};
use htm::topology::OneDimension;

static INPUT_SIZE: usize = 128;
static COL_COUNT: usize = 2048;
static DEPTH: usize = 32;
static PERIOD: usize = 32;

fn main() {
    let input = (0..PERIOD).map(|i|
        (0..INPUT_SIZE).map(|j| j*3*PERIOD + INPUT_SIZE > 3*i*INPUT_SIZE && j*3*PERIOD < (3*i+4)*INPUT_SIZE).collect::<Vec<_>>()
    ).collect::<Vec<_>>();

    let mut spooler = SpatialPooler::new(
        COL_COUNT,
        OneDimension::new(COL_COUNT),
        SpatialPoolerConfig {
            input_size: INPUT_SIZE,
            connected_perm: 0.2,
            permanence_inc: 0.003,
            permanence_dec: 0.0005,
            sliding_average_factor: 0.01,
            min_overlap: 1,
            desired_local_activity: 40,
            initial_dev: 0.1,
            initial_proximal_segment_size: 16,
        }
    );

    let mut tpooler = TemporalPooler::new(
        COL_COUNT,
        DEPTH,
        TemporalPoolerConfig {
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

    for t in 0..400 {
        let cols = spooler.pool_train(&input[i]);
        let out = tpooler.pool_train(&cols[..]);
        i = (i+1) % PERIOD;
        println!("{},{:.6}", t, tpooler.anomaly());
    }

    // then, skip some beats and continue !
    i = (i+4) % PERIOD;

    for t in 0..400 {
        let cols = spooler.pool_train(&input[i]);
        let out = tpooler.pool_train(&cols[..]);
        i = (i+1) % PERIOD;
        println!("{},{:.6}", 400+t, tpooler.anomaly());
    }
}