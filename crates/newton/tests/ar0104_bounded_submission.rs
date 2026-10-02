// SPDX-License-Identifier: MIT

//! Back-pressure must not deadlock. With a submission bound far below the number
//! of concurrent producers, every producer still finishes and every result is
//! right.
//!
//! The bound is held from submission until the outcome arrives, so with a bound
//! of `B` at most `B` messages are ever queued or batched. Producers past it wait
//! on the accelerator, not on each other; the worker flushes as soon as its
//! channel runs dry, which frees the permits. A deadlock here would show as the
//! timeout below firing.

use aristotle::{World, WorldKey};
use futures::future::join_all;
use newton::Accelerator;
use std::sync::Arc;
use std::time::Duration;

const PRODUCERS: usize = 32;
const JOBS_PER_PRODUCER: usize = 64;

type Case = (WorldKey<f32>, WorldKey<f32>, WorldKey<f32>, f32);

async fn hammer(bound: usize, batch: Option<usize>) {
    let world = Arc::new(World::builder().usual::<f32>());
    let cases: Arc<Vec<Vec<Case>>> = {
        let mut map = world.write::<f32>();
        Arc::new(
            (0..PRODUCERS)
                .map(|p| {
                    (0..JOBS_PER_PRODUCER)
                        .map(|j| {
                            let a = (p * JOBS_PER_PRODUCER + j) as f32;
                            (map.add(a), map.add(0.5), map.add(0.0), a + 0.5)
                        })
                        .collect()
                })
                .collect(),
        )
    };
    let builder = Accelerator::<f32>::builder(world.clone()).max_pending(bound);
    let builder = match batch {
        Some(rows) => builder.batch_size(rows),
        None => builder,
    };
    let accel = Arc::new(builder.build());

    // Real OS-thread producers on the multi-thread runtime, each also
    // submitting concurrently within itself.
    let tasks: Vec<_> = (0..PRODUCERS)
        .map(|p| {
            let accel = accel.clone();
            let cases = cases.clone();
            tokio::spawn(async move {
                let futs = cases[p].iter().map(|c| accel.simple_sum(&c.0, &c.1, &c.2));
                for r in join_all(futs).await {
                    r.unwrap();
                }
            })
        })
        .collect();

    tokio::time::timeout(Duration::from_secs(120), join_all(tasks))
        .await
        .unwrap_or_else(|_| panic!("bound {bound}: producers did not finish — deadlock"))
        .into_iter()
        .for_each(|r| r.unwrap());

    for (p, row) in cases.iter().enumerate() {
        for (j, c) in row.iter().enumerate() {
            assert_eq!(c.2.read(), c.3, "bound {bound}: producer {p}, job {j}");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_producers_finish_under_a_bound_of_one() {
    hammer(1, None).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_producers_finish_under_a_tiny_bound() {
    hammer(3, None).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_producers_finish_under_a_tiny_bound_and_tiny_batches() {
    hammer(3, Some(2)).await;
}

/// The single-threaded runtime too: there the producers cannot run while one
/// of them is blocked, so a bound that BLOCKED the thread instead of awaiting
/// would hang here.
#[tokio::test]
async fn many_producers_finish_under_a_tiny_bound_on_one_thread() {
    hammer(2, None).await;
}
