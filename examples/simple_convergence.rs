extern crate rand;
extern crate htm;
extern crate itertools;

use htm::Pooling;
use htm::cla::{Region, TransitionMemoryConfig, PatternMemoryConfig};
use htm::topology::OneDimension;

static INPUT_SIZE: usize = 1024;
static COL_COUNT: usize = 1024;
static DEPTH: usize = 8;
static PERIOD: usize = 8;

fn main() {
    let input = (0..PERIOD).map(|i|
        (
            if i > 0 { (3*i-1)*INPUT_SIZE/(3*PERIOD) } else { 0 }
            ..
            ::std::cmp::min((3*i+4)*INPUT_SIZE/(3*PERIOD), INPUT_SIZE)
        ).collect::<Vec<_>>()
    ).collect::<Vec<_>>();

    let mut region = Region::new(
        INPUT_SIZE,
        COL_COUNT,
        DEPTH,
        OneDimension::new(COL_COUNT),
        PatternMemoryConfig {
            connected_perm: 0.2,
            permanence_inc: 0.1,
            permanence_dec: 0.01,
            sliding_average_factor: 0.001,
            min_overlap: 5,
            desired_local_activity: 20,
            initial_dev: 0.1,
            proximal_segment_size: 16,
            proximal_segment_density: 0.5
        },
        TransitionMemoryConfig {
            initial_perm: 0.21,
            connected_perm: 0.2,
            permanence_inc: 0.01,
            permanence_dec: 0.01,
            activation_thresold: 4,
            learning_thresold: 5,
            new_synapses: 8,
            initial_dev: 0.2,
            initial_segment_count: 4,
            initial_distal_segment_size: 8,
            max_segment_count: 8,
            max_distal_segment_size: 16
        }
    );

    let mut i = 0;

    for t in 0..400 {
        let _out = region.pool_train(&input[i]);
        i = (i+1) % PERIOD;
        println!("{},{:.6}", t, region.anomaly());
    }

    // then, skip some beats and continue !
    i = (i+4) % PERIOD;

    for t in 0..400 {
        let _out = region.pool_train(&input[i]);
        i = (i+1) % PERIOD;
        println!("{},{:.6}", 400+t, region.anomaly());
    }
}
