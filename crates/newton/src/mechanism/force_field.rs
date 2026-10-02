// SPDX-License-Identifier: MIT

use super::component::Component;
use aristotle::{Epoch, WorldId, WorldKey};
use async_trait::async_trait;
use bytemuck::Pod;
use clifford::pga3::Wrench;
use indexmap::IndexMap;
use peano::prelude::*;

// ============================================================================
// FORCE FIELD — external force (N-body gravity, aero, thrust)
// ============================================================================

/// External force field. Sees ALL bodies (N-body uses this for pairwise
/// forces), writes the wrench contribution into `out[i]`, matching `bodies[i]`.
///
/// `&self`: the field only READS the outside world — body state has already been
/// updated in pre-step. Internal mutation of the field (atomics, write-locks inside
/// the propagator) is allowed via interior mutability.
/// `epoch: &Epoch<T>` — snapshot of the current tick: broker reads are indexed by it,
/// the integrator step is taken from `epoch.dt()`. A field may ignore it (gravity, aero
/// — `_epoch`), but time-varying fields and implicit-form springs (Padé with `(1+ω·dt)²`
/// in the denominator) read `epoch.dt()` directly, and fuel consumption `Δm = ṁ·dt`
/// also goes here — no separate Actuator abstraction is needed. dt is NOT passed
/// as a separate parameter: a single Epoch handle rules out epoch/dt desync.
/// The mechanism does NOT know about gravity separately — it is one implementation of the trait.
///
/// TWO-PHASE PROTOCOL. In one step `Mechanism::step` first calls `accumulate`
/// for all fields, then — AFTER integrating positions — `publish`.
/// `publish` is the cache phase: the field walks the bodies with FRESH (just
/// integrated) positions and records its per-body state (charges,
/// medium density, direction to the Sun, etc.) in the BACK buffer. An external
/// `advance_epoch` swaps BACK→FRONT between rounds, and the `accumulate` of
/// the next step reads charges corresponding to the CURRENT positions — there is no
/// structural lag-1: the force `F(x_k)` is computed from the same `x_k` that are integrated
/// (see `gravity::GravityPropagator`). Publishing after integration makes the
/// coupling synchronous without touching the double-buffer machinery. For static fields
/// (uniform thrust, constant wind) `publish` is a no-op (the default).
#[async_trait]
pub trait ForceField<T: Scalar + Pod, S: Scalar + From<T>>: Send + Sync {
    /// Cache phase: walk the bodies and record per-body state in the
    /// field's internal buffers. Called AFTER `accumulate` (and after
    /// integrating positions). Default is a no-op.
    fn publish(&self, _bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>, _origin: &Vector3<S>) {
    }

    /// Write the field's contribution into `out[i]` for each `bodies[i]`. Relies on
    /// the state recorded by `publish` of the PREVIOUS step (via the double-buffer
    /// swap). `epoch` carries the step via `epoch.dt()` and
    /// identifies the tick for broker reads.
    async fn accumulate(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        out: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        epoch: &Epoch<T>,
        origin: &Vector3<S>,
    );
}
