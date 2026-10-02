// SPDX-License-Identifier: MIT

// ============================================================================
// ARISTOTLE — shared primary notions of the world: WorldId
// ============================================================================
//
// WorldId — a stable unique identifier of any world entity (body,
// mechanism, agent). Implemented.
//
// The epoch / temporal frame of observers lives in the `epoch` module.
//
// (A generic `Broker<P>` publish/subscribe over WorldId, lifted out of
// newton::gravity::GravityPropagator, was planned here earlier. CANCELLED — not needed.)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

mod epoch;
mod world;

pub use epoch::{Epoch, EpochBuilder};
pub use world::{World, WorldKey};

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// World identifier of a body. Stable and unique: a sequence number plus an epoch.
/// The combination is unique enough that there is no need to persist the counter state between program runs.
#[derive(PartialEq, Debug, Copy, Clone, Eq, Hash)]
pub struct WorldId(usize, u64);

impl WorldId {
    pub fn get() -> Self {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self(ID_COUNTER.fetch_add(1, Ordering::SeqCst), epoch)
    }
}
