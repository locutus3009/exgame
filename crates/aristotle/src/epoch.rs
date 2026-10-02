// SPDX-License-Identifier: MIT

use crate::World;
use std::{
    cell::UnsafeCell,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

/// A callback registered on an [`EpochBuilder`] and run at the end of every
/// epoch, on the coordinator thread, after every worker has dropped its
/// [`Epoch`].
type Broker<T> = Box<dyn Fn(&Epoch<T>) + Send + Sync + 'static>;

pub struct EpochBuilder<T: Copy> {
    brokers: Vec<Broker<T>>,
    world: Option<Arc<World>>,
}

impl<T: Copy> EpochBuilder<T> {
    /// Register a callback to run at the end of every epoch, after all workers
    /// have dropped their `Epoch<T>` and before `enter` returns to the caller.
    ///
    /// **Keep brokers lightweight.** They execute on the coordinator thread
    /// inside `enter`, in the gap between "all workers finished" and "next
    /// epoch may start". Their wall time directly extends the inter-epoch
    /// interval and blocks the next `enter`. Use them only for work that
    /// requires a consistent snapshot at the epoch boundary: double-buffer
    /// swaps, refcount resets, snapshot-pointer updates. Substantive work —
    /// charge construction, physics, rendering — belongs in the worker body
    /// where it parallelises.
    pub fn with_broker<F>(mut self, f: F) -> Self
    where
        F: Fn(&Epoch<T>) + Send + Sync + 'static,
    {
        self.brokers.push(Box::new(f));
        self
    }

    pub fn with_world(mut self, world: Arc<World>) -> Self {
        self.world = Some(world);
        self
    }

    pub fn build(self, n: usize) -> (Vec<EpochHandle<T>>, EpochCoordinator<T>) {
        let arc = Arc::new(EpochInner {
            brokers: self.brokers,
            n_workers: AtomicUsize::new(0),
            phase: AtomicBool::new(false),
            finished: AtomicUsize::new(0),
            dt: UnsafeCell::new(None),
            warp: UnsafeCell::new(None),
            world: self.world,
        });

        arc.n_workers.store(n, Ordering::SeqCst);
        let mut out = Vec::new();
        for _ in 0..n {
            out.push(EpochHandle {
                h: arc.clone(),
                last_phase: false,
            });
        }
        (out, EpochCoordinator(arc, Epoch::<T>::empty()))
    }
}

struct EpochInner<T: Copy> {
    /// Invariant: set of brokers is fixed after initialization.
    brokers: Vec<Broker<T>>,
    n_workers: AtomicUsize,
    /// Flipped by the coordinator on each `enter`/`finish` to publish a fresh
    /// signal. Acts as the Release/Acquire fence that makes the preceding
    /// non-atomic write to `dt` visible to workers.
    phase: AtomicBool,
    finished: AtomicUsize,
    dt: UnsafeCell<Option<T>>,
    warp: UnsafeCell<Option<T>>,
    world: Option<Arc<World>>,
}

impl<T: Copy> EpochInner<T> {
    fn work(&self, epoch: &Epoch<T>) {
        for f in &self.brokers {
            (*f)(epoch);
        }
    }
}

// SAFETY: the two `UnsafeCell`s, `dt` and `warp`, are what keep this from being
// `Sync` on its own. Both are written only by the coordinator (`enter`, and its
// `Drop`), in the window between "every worker dropped its `Epoch`" and the next
// Release flip of `phase`; a worker reads them only after an Acquire load has
// seen that flip (`EpochHandle::enter`) and only through an `Epoch` it holds,
// which the coordinator waits out before writing again. Reads may run on many
// threads at once — `dt()` and `warp()` hand out `&T` — hence `T: Sync`.
unsafe impl<T: Copy + Send + Sync> Sync for EpochInner<T> {}

/// Adaptive busy-wait used inside the epoch hand-off loops. Each iteration
/// emits a CPU spin-loop hint (cheap, SMT-friendly, low power); every 256
/// iterations the OS quantum is yielded so a long-tail epoch (heavy broker,
/// scheduler jitter, a stalled worker) does not pin the core.
#[inline]
fn back_off(spins: &mut u32) {
    *spins = spins.wrapping_add(1);
    if *spins & 0xFF == 0 {
        std::thread::yield_now();
    } else {
        std::hint::spin_loop();
    }
}

pub struct EpochCoordinator<T: Copy>(Arc<EpochInner<T>>, Epoch<T>);

impl<T: Copy> Drop for EpochCoordinator<T> {
    fn drop(&mut self) {
        // SAFETY: Drop runs outside an epoch (caller contract): no `Epoch<T>`
        // is alive, all workers spin on `phase` and do not touch `dt`. The
        // Release flip below publishes this write before any worker's next
        // Acquire-load → dt read.
        unsafe { *self.0.dt.get() = None };
        self.0.phase.fetch_xor(true, Ordering::Release);
    }
}

impl<T: Copy> EpochCoordinator<T> {
    pub fn enter(&self, dt: T, warp: T) {
        // SAFETY: between epochs — workers from the previous epoch have all
        // dropped their `Epoch<T>` (precondition: `wait()` returned), and now
        // spin on `phase` without touching `dt` or `warp`. The Release flip below
        // publishes these writes before any worker observes the new phase.
        unsafe {
            *self.0.dt.get() = Some(dt);
            *self.0.warp.get() = Some(warp);
        }

        // SAFETY: `self.1` is the coordinator's own broker epoch. Its `Arc` is
        // never handed out — brokers borrow it only inside `wait`, below, on
        // this same thread — so nothing else can be reading these cells.
        unsafe {
            *self.1.0.dt.get() = Some(dt);
            *self.1.0.warp.get() = Some(warp);
        }

        self.0.phase.fetch_xor(true, Ordering::Release);
        self.wait();
    }

    fn wait(&self) {
        let max = self.0.n_workers.load(Ordering::SeqCst);
        let mut spins = 0u32;
        loop {
            if self
                .0
                .finished
                .compare_exchange(max, 0, Ordering::SeqCst, Ordering::Acquire)
                .is_ok()
            {
                // self.1 is the epoch for which brokers are being called.
                // it is not "active" in a sense that its destruction will lead for brokers call.
                self.0.work(&self.1);
                break;
            }
            back_off(&mut spins);
        }
    }
}

/// Requirement: only one per-thread
pub struct EpochHandle<T: Copy> {
    h: Arc<EpochInner<T>>,
    last_phase: bool,
}

impl<T: Copy> EpochHandle<T> {
    fn enter(&mut self) -> Option<Epoch<T>> {
        let mut spins = 0u32;
        let p = loop {
            let p = self.h.phase.load(Ordering::Acquire);
            if p != self.last_phase {
                break p;
            }
            back_off(&mut spins);
        };
        self.last_phase = p;
        // SAFETY: the Acquire load above saw the coordinator's Release flip, so
        // its write to `dt` is visible; and the coordinator writes `dt` again
        // only after this worker's `Epoch` (created below) is dropped, or —
        // when it was cleared to `None` on `Drop` — never again.
        let _ = unsafe { *self.h.dt.get() }?;
        Some(Epoch(self.h.clone()))
    }

    pub fn run<F>(&mut self, mut f: F)
    where
        F: FnMut(Epoch<T>),
    {
        while let Some(epoch) = self.enter() {
            f(epoch);
        }
    }
}

/// Invariant: if this struct exists for some thread, then all other threads are either on the same epoch or waiting for the entrance to a new epoch.
pub struct Epoch<T: Copy>(Arc<EpochInner<T>>);

impl<T: Copy> Epoch<T> {
    pub fn builder() -> EpochBuilder<T> {
        EpochBuilder {
            brokers: Vec::new(),
            world: None,
        }
    }

    pub fn world(&self) -> Option<&Arc<World>> {
        self.0.world.as_ref()
    }

    pub fn dt(&self) -> &T {
        // SAFETY: while this `Epoch` lives the coordinator does not write `dt`
        // (it waits for every worker's `Epoch` to drop first), and the borrow
        // returned cannot outlive `self`. A standalone or empty epoch has no
        // coordinator at all.
        unsafe { &*self.0.dt.get() }.as_ref().unwrap()
    }

    pub fn warp(&self) -> &T {
        // SAFETY: as `dt` — `warp` is written in the same window.
        unsafe { &*self.0.warp.get() }.as_ref().unwrap()
    }

    /// Standalone epoch without coordinator/workers/brokers. For tests and
    /// single-threaded demos that drive `newton::Mechanism::step` directly.
    /// `dt()` returns the supplied value; no broker callbacks fire on drop.
    /// The «one thread = one epoch» invariant is the caller's to uphold.
    pub fn standalone(dt: T, warp: T) -> Self {
        Self(Arc::new(EpochInner {
            brokers: Vec::new(),
            n_workers: AtomicUsize::new(0),
            phase: AtomicBool::new(false),
            finished: AtomicUsize::new(0),
            dt: UnsafeCell::new(Some(dt)),
            warp: UnsafeCell::new(Some(warp)),
            world: None,
        }))
    }

    pub fn empty() -> Self {
        Self(Arc::new(EpochInner {
            brokers: Vec::new(),
            n_workers: AtomicUsize::new(0),
            phase: AtomicBool::new(false),
            finished: AtomicUsize::new(0),
            dt: UnsafeCell::new(None),
            warp: UnsafeCell::new(None),
            world: None,
        }))
    }
}

impl<T: Copy> Drop for Epoch<T> {
    fn drop(&mut self) {
        self.0.finished.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_epoch_simple_separate_coordinator() {
        const N: usize = 10;
        let test_accumulator = Arc::new(AtomicUsize::new(0));
        let accumulator = test_accumulator.clone();

        let (mut handles, coordinator) = Epoch::builder()
            .with_broker(move |epoch| {
                accumulator.fetch_add(*epoch.dt(), Ordering::SeqCst);
            })
            .build(N);

        let counter = thread::scope(move |s| {
            let coordinator_thread_handle = s.spawn(move || {
                let counter = Arc::new(AtomicUsize::new(0));
                for i in 0..10 {
                    // The next function blocks until completion of all workers
                    // Hey, we can use different time steps!
                    coordinator.enter(i + 1, 1);
                    counter.fetch_add(1, Ordering::SeqCst);
                    println!("Finish step {i}");
                }
                counter
            });

            while let Some(mut handle) = handles.pop() {
                s.spawn(move || {
                    handle.run(|epoch| {
                        // Do some work with epoch, publish to a propagator etc.
                        // Work with mechanisms etc. etc.
                        let _dt = epoch.dt();
                        // After all workers has finished, all propagator-related callbacks
                        // are automatically called by the epoch coordinator.
                        // In theory, we can even send this epoch to some another thread,
                        // the outer loop will never enter its body again while current
                        // epoch exists.
                        // But it is advised to keep things clean and try not to expose epoch
                        // outside of this callback body.
                    })
                });
            }

            coordinator_thread_handle.join().unwrap()
        });

        // We must take exactly ten steps, even if 1000 workers have updated their states
        assert_eq!(counter.load(Ordering::SeqCst), 10);
        assert_eq!(test_accumulator.load(Ordering::SeqCst), 55);
    }

    #[test]
    fn test_epoch_simple_coordinator_in_main() {
        const N: usize = 10;
        let test_accumulator = Arc::new(AtomicUsize::new(0));
        let accumulator = test_accumulator.clone();

        let (mut handles, coordinator) = Epoch::builder()
            .with_broker(move |epoch| {
                accumulator.fetch_add(*epoch.dt(), Ordering::SeqCst);
            })
            .build(N);

        while let Some(mut handle) = handles.pop() {
            thread::spawn(move || {
                handle.run(|epoch| {
                    let _dt = epoch.dt();
                })
            });
        }

        let counter = Arc::new(AtomicUsize::new(0));
        for i in 0..10 {
            coordinator.enter(i + 1, 1);
            counter.fetch_add(1, Ordering::SeqCst);
            println!("Finish step {i}");
        }
        // Must ensure! This will command all worker threads to exit
        drop(coordinator);
        // It is advised to join all worker threads here or use scope as in previous test to ensure
        // a proper resource cleanup from these worker threads; for now it is irrelevant.

        assert_eq!(counter.load(Ordering::SeqCst), 10);
        assert_eq!(test_accumulator.load(Ordering::SeqCst), 55);
    }
}
