// SPDX-License-Identifier: MIT

use super::component::Component;
use super::force_field::ForceField;
use super::islands::Islands;
use crate::{Accelerator, EvalError, RigidBody, integrator::ImplicitIntegrator};
use aristotle::{Epoch, WorldId};
use bytemuck::Pod;
use clifford::Lift;
use clifford::pga3::{Motor, Point, Twist, Wrench};
use futures::future::join_all;
use joints::Joint;
use peano::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use tokio::sync::{RwLock, RwLockWriteGuard};

/// Why a structural change, or a driver registration, was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureError {
    /// The mechanism (by id) is inside an epoch: its structure is frozen until
    /// the epoch ends (ACCELERATOR.md Part II, the epoch invariant). Refused
    /// rather than queued — retry between epochs.
    EpochInProgress(WorldId),
    /// The body (by world id) already belongs to a mechanism. A body has
    /// exactly one owner, which is what keeps every slot single-writer across
    /// mechanisms (ACCELERATOR.md Part III).
    AlreadyOwned(WorldId),
    /// The mechanism (by id) runs on a different accelerator than the driver.
    ForeignAccelerator(WorldId),
    /// The mechanism (by id) is already registered with the driver.
    DuplicateMechanism(WorldId),
}

impl std::fmt::Display for StructureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EpochInProgress(m) => write!(
                f,
                "mechanism {m:?} is inside an epoch; its structure is frozen until the epoch ends"
            ),
            Self::AlreadyOwned(b) => write!(
                f,
                "duplicate world id {b:?}: the body already belongs to a mechanism"
            ),
            Self::ForeignAccelerator(m) => write!(
                f,
                "mechanism {m:?} runs on a different accelerator than the driver"
            ),
            Self::DuplicateMechanism(m) => {
                write!(f, "mechanism {m:?} is already registered with the driver")
            }
        }
    }
}

impl std::error::Error for StructureError {}

/// Every body world id currently owned by a live mechanism, process-wide.
/// World ids are unique per process (`WorldId::get`), so one set serves every
/// world and every accelerator. A body is claimed when it is added and released
/// when it is detached or its mechanism is dropped; `split` and `merge` move
/// bodies between mechanisms without releasing them.
static OWNED: LazyLock<Mutex<HashSet<WorldId>>> = LazyLock::new(Default::default);

/// The owned set. Nothing panics while holding it, so a poisoned lock can
/// only come from an unrelated panic elsewhere; the set itself is intact.
fn owned() -> std::sync::MutexGuard<'static, HashSet<WorldId>> {
    OWNED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Keeps a mechanism's structure frozen while alive — see [`Mechanism::freeze`].
pub(crate) struct Frozen<'a>(&'a AtomicUsize);

impl Drop for Frozen<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

// ============================================================================
// MECHANISM
// ============================================================================
//
// ── PLANNED: publish stage 5.5 and MechanismSummary ─────────────────────────
// Between phase 5 (integrate) and the return from step() a phase 5.5 is added: from
// the list of registered publishers a payload is formed and pushed into the
// corresponding aristotle::Broker<P>. This is the "mechanism snapshot outward" channel
// for rendering, UI, the camera — observers read the payload with a one-step lag,
// without taking the RigidBody.
//
// MechanismSummary<T> — the overview payload type:
//   pose: Motor<T>           — pose of the "primary body" (see below)
//   centroid: Point<T>       — mass-weighted CoM (the method already exists)
//   momentum: Wrench<T>      — Σ body.world_momentum() — for UI velocities
//   mass: T                  — Σ body.inertia.mass() — for HUD/dv computation
//   bounding_radius: T       — max ‖body_pos − centroid‖ — for camera auto-zoom
//
// PRIMARY BODY. A multi-body mechanism has no natural "pose of the whole" —
// a convention is needed. Mechanism gets a field primary_body: Option<WorldId> +
// a setter. With None — pose = identity (or a panic when trying to publish the
// summary). The variant with principal axes from the total inertia tensor
// (eigen-decomposition) is deferred; a primary body gives the user explicit
// control (a ship's "cockpit") without additional math.
//
// The existing GravityPropagator already works on the same scheme (only the charge
// = COM+GM, not a full summary) — after migrating to aristotle::Broker<P> both
// publishers are unified under the common infrastructure.

/// A multigraph of bodies (fully connected, with cycles, parallel joints allowed,
/// self-loops forbidden). Orchestrates the step: pre-step → wrench accumulation →
/// integration. Joint physics lives in `Joint`, fields in `ForceField`,
/// the integrator is encapsulated (statically, as a type parameter).
pub struct Mechanism<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    id: WorldId,
    // The published snapshot of the mechanism summary (mass-weighted CoM and its world
    // velocity). Held under THEIR OWN locks, separate from `inner`: an observer
    // (the camera) reads them WITHOUT touching `inner`, so it can live in one async
    // batch with the target's `step` — the target's step holds `inner.write()` across its `.await`,
    // but the snapshot under its own lock is updated synchronously at the end of the step, without
    // being held across an await. `None` — empty mechanism (CoM undefined).
    // Updated EVERYWHERE the composition/state changes: `step` and the graph mutators
    // (`add_body`/`connect`/`detach`/`split`/`merge`/`update_body`), — so that
    // a reader after construction (before the first step) sees up-to-date data.
    centroid: RwLock<Option<Point<S>>>,
    centroid_velocity: RwLock<Option<Twist<T>>>,
    inner: RwLock<MechanismInner<T, S>>,
    /// Epochs (and bare steps) currently holding the structure frozen.
    frozen: AtomicUsize,
    /// Epochs (and bare steps) ever begun. A structural change compares it
    /// before and after waiting for `inner`, so one that started waiting just
    /// before an epoch began is refused instead of running after it.
    epochs: AtomicU64,
}

struct MechanismInner<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    /// Bodies, partitioned into connectivity islands. The partition is the storage itself:
    /// each `Island` is exactly one component, the invariant is upheld by the mutators of
    /// `Islands`. There is no longer a separate partition cache that one could forget
    /// to recompute.
    islands: Islands<T, S>,
    fields: Vec<Box<dyn ForceField<T, S>>>,
    // The integrator carries its own clone of the ONE accelerator (see
    // `ImplicitIntegrator`); the mechanism no longer holds it.
    integrator: ImplicitIntegrator<T>,
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod, S: Scalar + From<T> + Into<T>>
    Mechanism<T, S>
{
    /// Build a mechanism around a ready integrator (which already holds a clone of
    /// the one accelerator the app created explicitly):
    /// `Mechanism::new(ImplicitIntegrator::SymplecticEuler(gpu.clone()))`.
    pub fn new(integrator: ImplicitIntegrator<T>) -> Self {
        Self {
            centroid: RwLock::new(None),
            centroid_velocity: RwLock::new(None),
            id: WorldId::get(),
            inner: RwLock::new(MechanismInner {
                islands: Islands::new(),
                fields: Vec::new(),
                integrator,
            }),
            frozen: AtomicUsize::new(0),
            epochs: AtomicU64::new(0),
        }
    }

    /// Freeze the structure until the returned guard drops: every structural
    /// change (`try_add_body`, `try_connect`, `try_remove_body`, `try_split`,
    /// `try_merge`) is refused with [`StructureError::EpochInProgress`]
    /// meanwhile. `step` freezes for its own duration; the driver freezes all
    /// its mechanisms for the whole epoch.
    pub(crate) fn freeze(&self) -> Frozen<'_> {
        self.frozen.fetch_add(1, Ordering::SeqCst);
        self.epochs.fetch_add(1, Ordering::SeqCst);
        Frozen(&self.frozen)
    }

    /// The structure lock, for a structural change — refused, never queued,
    /// while an epoch holds the structure frozen or if one began while this
    /// call was waiting for the lock.
    async fn structure(
        &self,
    ) -> Result<RwLockWriteGuard<'_, MechanismInner<T, S>>, StructureError> {
        let refused = StructureError::EpochInProgress(self.id);
        let began = self.epochs.load(Ordering::SeqCst);
        if self.frozen.load(Ordering::SeqCst) > 0 {
            return Err(refused);
        }
        let guard = self.inner.write().await;
        if self.frozen.load(Ordering::SeqCst) > 0 || self.epochs.load(Ordering::SeqCst) != began {
            return Err(refused);
        }
        Ok(guard)
    }

    /// This mechanism's accelerator — pull it to build a sibling integrator that
    /// shares the same GPU point (e.g. `split`).
    pub async fn accelerator(&self) -> Arc<Accelerator<T>> {
        self.inner.read().await.integrator.accelerator()
    }

    // --- Graph composition (public — world ids) ---

    /// Add a body under its world id. Panics where [`Self::try_add_body`]
    /// refuses.
    pub async fn add_body(&self, entity: Box<dyn Component<T, S>>) -> WorldId {
        self.try_add_body(entity)
            .await
            .unwrap_or_else(|e| panic!("add_body: {e}"))
    }

    /// Add a body under its world id. Refused (and the entity dropped) while
    /// an epoch is in progress, or if a mechanism — this one or another —
    /// already owns a body with that id.
    pub async fn try_add_body(
        &self,
        entity: Box<dyn Component<T, S>>,
    ) -> Result<WorldId, StructureError> {
        let mut guard = self.structure().await?;
        let id = entity.id();
        if !owned().insert(id) {
            return Err(StructureError::AlreadyOwned(id));
        }
        guard.islands.add_body(entity);
        self.store_snapshot(&guard.islands).await;
        Ok(id)
    }

    /// Alias for Self::add_body()
    #[inline]
    pub async fn attach(&self, entity: Box<dyn Component<T, S>>) -> WorldId {
        self.add_body(entity).await
    }

    /// Connect body `world_id` by joints to targets. Each element is (joint,
    /// target world id). Panics on a self-loop, an unknown id, or where
    /// [`Self::try_connect`] refuses. World ids are translated into keys ONCE
    /// here.
    pub async fn connect(&self, world_id: WorldId, links: Vec<(Joint<T>, WorldId)>) {
        self.try_connect(world_id, links)
            .await
            .unwrap_or_else(|e| panic!("connect: {e}"))
    }

    /// [`Self::connect`], refused while an epoch is in progress.
    pub async fn try_connect(
        &self,
        world_id: WorldId,
        links: Vec<(Joint<T>, WorldId)>,
    ) -> Result<(), StructureError> {
        let mut guard = self.structure().await?;
        guard.islands.connect(world_id, links);
        self.store_snapshot(&guard.islands).await;
        Ok(())
    }

    /// Remove a body and ALL its joints. Expensive: O(joints) — a linear pass without
    /// adjacency lists (a deliberate trade-off, removal is rare). Panics where
    /// [`Self::try_remove_body`] refuses.
    pub async fn remove_body(&self, world_id: WorldId) {
        self.try_remove_body(world_id)
            .await
            .unwrap_or_else(|e| panic!("remove_body: {e}"))
    }

    /// [`Self::remove_body`], refused while an epoch is in progress. An unknown
    /// id is not an error.
    pub async fn try_remove_body(&self, world_id: WorldId) -> Result<(), StructureError> {
        self.try_detach(world_id).await.map(|_| ())
    }

    /// Internal migration primitive: extract a body whole (with its behavior), cutting
    /// its joints. `pub(crate)` — not public API (from outside only split/merge),
    /// but the integration tests in `mod.rs` call it. None if the id is unknown.
    #[cfg(test)]
    pub(crate) async fn detach(&self, world_id: WorldId) -> Option<Box<dyn Component<T, S>>> {
        self.try_detach(world_id)
            .await
            .unwrap_or_else(|e| panic!("detach: {e}"))
    }

    /// [`Self::detach`], refused while an epoch is in progress. The body is
    /// released: it may be added to a mechanism again.
    async fn try_detach(
        &self,
        world_id: WorldId,
    ) -> Result<Option<Box<dyn Component<T, S>>>, StructureError> {
        let mut guard = self.structure().await?;
        let out = guard.islands.detach(world_id);
        if out.is_some() {
            owned().remove(&world_id);
        }
        self.store_snapshot(&guard.islands).await;
        Ok(out)
    }

    /// Register a force field.
    #[inline]
    pub async fn add_field(&self, field: Box<dyn ForceField<T, S>>) {
        let mut guard = self.inner.write().await;
        guard.fields.push(field);
    }

    // --- Splitting and merging mechanisms ---

    /// Split bodies `world_ids` off into a NEW mechanism with an explicitly given integrator
    /// `J` (it may differ in type — "we landed on a planet, it has its own
    /// solver"). The new mechanism is born with an EMPTY set of fields — forces for it
    /// are set explicitly. `self` keeps its integrator and fields.
    ///
    /// Joints:
    /// - internal (both ends in the group) — move together with the bodies;
    /// - boundary (one end in the group, the other not) — are CUT;
    /// - external (both outside) — stay in `self`.
    ///
    /// Panics on an unknown world id, or where [`Self::try_split`] refuses.
    pub async fn split(
        &self,
        world_ids: &[WorldId],
        integrator: ImplicitIntegrator<T>,
    ) -> Mechanism<T, S> {
        self.try_split(world_ids, integrator)
            .await
            .unwrap_or_else(|e| panic!("split: {e}"))
    }

    /// [`Self::split`], refused while an epoch is in progress.
    pub async fn try_split(
        &self,
        world_ids: &[WorldId],
        integrator: ImplicitIntegrator<T>,
    ) -> Result<Mechanism<T, S>, StructureError> {
        let mut guard = self.structure().await?;
        for &id in world_ids {
            assert!(
                guard.islands.island_of(id).is_some(),
                "split: unknown world id {id:?}"
            );
        }
        // Extract the bodies (with behavior) and internal joints into a new set
        // of islands; boundary ones are cut, external ones and the affected islands of self
        // are rebuilt inside extract.
        let islands = guard.islands.extract(world_ids);
        // The bodies left self → its snapshot is stale; recompute under the held
        // lock. The new mechanism is born with its own snapshot from its own islands.
        self.store_snapshot(&guard.islands).await;
        let centroid = Self::compute_centroid(&islands);
        let centroid_velocity = Self::compute_centroid_velocity(&islands);
        Ok(Mechanism {
            centroid: RwLock::new(centroid),
            centroid_velocity: RwLock::new(centroid_velocity),
            id: WorldId::get(),
            inner: RwLock::new(MechanismInner {
                islands,
                fields: Vec::new(),
                integrator,
            }),
            frozen: AtomicUsize::new(0),
            epochs: AtomicU64::new(0),
        })
    }

    /// Merge `other` into `self`, creating exactly one joint that CROSSES the boundary
    /// (one end was in `self`, the other in `other`). `self` keeps its
    /// integrator and fields; the integrator and fields of `other` are destroyed with it.
    /// `other` may have an integrator of ANY type `J` — it does not get
    /// into `self` (only the integrator-agnostic contents are carried over).
    ///
    /// Panics if the joint does not cross the boundary (both ends on one side is
    /// not a merge but a connect). World id collisions are impossible by construction of
    /// `WorldId`, so there is no separate check. Panics where
    /// [`Self::try_merge`] refuses.
    pub async fn merge(&self, other: Mechanism<T, S>, link: (Joint<T>, WorldId, WorldId)) {
        if let Err((e, _)) = self.try_merge(other, link).await {
            panic!("merge: {e}");
        }
    }

    /// [`Self::merge`], refused while `self` is inside an epoch; `other` is
    /// handed back untouched with the refusal. (`other` itself is taken by
    /// value, so nothing can be stepping it.)
    pub async fn try_merge(
        &self,
        mut other: Mechanism<T, S>,
        link: (Joint<T>, WorldId, WorldId),
    ) -> Result<(), (StructureError, Box<Mechanism<T, S>>)> {
        let (joint, wa, wb) = link;

        assert!(self.id != other.id, "Cannot merge self");

        let mut self_guard = match self.structure().await {
            Ok(guard) => guard,
            Err(e) => return Err((e, Box::new(other))),
        };
        // Take other's islands; `other` then drops empty, releasing no body —
        // its bodies change owner, they are not given up.
        let other_islands = std::mem::replace(&mut other.inner.get_mut().islands, Islands::new());

        // Check boundary crossing BEFORE consuming: one end here, the other there.
        let a_here = self_guard.islands.island_of(wa).is_some();
        let b_here = self_guard.islands.island_of(wb).is_some();
        let a_there = other_islands.island_of(wa).is_some();
        let b_there = other_islands.island_of(wb).is_some();
        let crosses = (a_here && b_there) || (b_here && a_there);
        assert!(
            crosses,
            "merge link must cross the two mechanisms (got {wa:?}, {wb:?})"
        );

        // Merge in other's islands as they are, then link across the boundary — union
        // will merge the two islands into one.
        self_guard.islands.absorb(other_islands);
        self_guard.islands.connect(wa, vec![(joint, wb)]);
        // The composition of self grew by other's bodies → recompute the snapshot.
        self.store_snapshot(&self_guard.islands).await;
        // other's integrator and fields are destroyed here together with other.
        Ok(())
    }

    // --- Read access (world ids) ---

    #[inline]
    /// `‖I − A·X‖²` for a single island — how far the cached
    /// approximate inverse is from the true one. A TEST probe for the property
    /// "a warm start stays warm"; it takes no part in the step path.
    #[cfg(test)]
    pub(crate) async fn inspect_solver_residual(&self) -> Option<T> {
        let guard = self.inner.read().await;
        guard
            .islands
            .iter()
            .next()
            .map(|i| i.solver_inverse_residual())
    }

    pub async fn inspect_body<F, R>(&self, world_id: WorldId, f: F) -> Option<R>
    where
        F: AsyncFn(&RigidBody<T>) -> R,
    {
        let guard = self.inner.read().await;
        match guard.islands.get_body(world_id) {
            Some(e) => Some(f(e.body()).await),
            None => None,
        }
    }

    /// Mutating counterpart of `inspect_body`. The closure is `FnOnce`: it allows returning
    /// a value and/or capturing move-owned input. Returns `None` for
    /// an unknown id, like `inspect_body`.
    pub async fn update_body<F, R>(&self, world_id: WorldId, f: F) -> Option<R>
    where
        F: AsyncFn(&mut RigidBody<T>) -> R,
    {
        let mut guard = self.inner.write().await;
        let out = match guard.islands.get_body_mut(world_id) {
            Some(e) => Some(f(e.body_mut()).await),
            None => None,
        };
        self.store_snapshot(&guard.islands).await;
        out
    }

    // TODO: add id for JointEdge, add inspect_joint() and update_joint()

    // --- Connectivity diagnostics (NOT an invariant: a disconnected mechanism is legal —
    // a mechanism is an orchestration container, not a physical entity). The returned
    // partition can be fed by the user into split to separate the components. ---

    pub async fn components(&self) -> Vec<Vec<WorldId>> {
        let guard = self.inner.read().await;
        guard.islands.components()
    }

    /// Whether the mechanism is connected (≤1 connectivity island). Empty and single-body ones are
    /// degenerately connected. This is now just the number of islands — the partition is the storage.
    pub async fn is_connected(&self) -> bool {
        let guard = self.inner.read().await;
        guard.islands.len() <= 1
    }
}

impl<T, S> Mechanism<T, S>
where
    T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod,
    S: Scalar + From<T> + Into<T>,
{
    // ─── Only `step` requires `Lift<T>` (via `ImplicitIntegrator::step_all`).
    //     The accessors (centroid, body_absolute_position, …) live in the block without
    //     `Lift` below, so that observers (the camera) read the mechanism without dragging
    //     the integrator bound along. ───

    /// One simulation step. `epoch` carries the step via `epoch.dt()` and is forwarded
    /// to `ForceField::accumulate` (broker reads, time-varying fields, implicit
    /// springs). Encapsulates the integrator: from outside only Mechanism::step is visible,
    /// not a per-body Integrator.
    ///
    /// The step runs BY ISLANDS: each island is a connected subsystem that the
    /// integrator solves as a whole (the implicit scheme sees exactly one connected set,
    /// not a disconnected "mush"). Fields are called per island; for gravity this is
    /// equivalent — the coupling flows through the propagator's global slab, not through
    /// the island's body map, so cross-component attraction is intact.
    ///
    /// Phases inside an island (the order matters — all forces at the start of the step):
    ///   1. pre-step: behavior.pre_step of each body (mutates state/inertia)
    ///   2. zero the wrench buffer over the island's keys
    ///   4. fields.accumulate(island bodies, &mut island wrenches, epoch): read the
    ///      state recorded by publish of the PREVIOUS step (via swap)   (WORLD)
    ///   5. integrator.step_all(bodies, joints, island wrenches, epoch) — the pullback
    ///      world→body lives inside the integrator; Mechanism only hands out islands.
    ///   6. fields.publish: cache of per-body state (charges etc.) AFTER
    ///      integration — into the BACK buffer; an external advance_epoch swaps into FRONT,
    ///      the next step's accumulate reads it → there is no structural lag-1.
    ///   7. publish the mechanism summary (CoM + its world velocity) into
    ///      `centroid`/`centroid_velocity` — under THEIR locks, not `inner`.
    ///      An observer (the camera) reads this summary without touching `inner`, so it
    ///      can sit in one async batch with this step (see hitchcock `simple`).
    pub async fn step(&self, epoch: &Epoch<T>) -> Result<(), EvalError> {
        // This `write` lock is held across the accelerator `.await` below. For the summary
        // observer (the camera) this is no longer a problem: `centroid()`/`centroid_velocity()`
        // read the snapshot from under SEPARATE locks (updated in phase 7, synchronously,
        // without being held across an await), so all mechanisms of an epoch can be
        // `join_all`-ed. A restriction remains for an observer of a LIVE body
        // (`inspect_body`/`body_absolute_pose` take `inner.read()`) — such a
        // reader will still block on the target's step; the summary is enough.
        // Frozen for the whole step: a structural change that arrives meanwhile
        // is refused, not queued behind the lock below.
        let _frozen = self.freeze();
        let mut guard = self.inner.write().await;

        let MechanismInner {
            ref mut islands,
            ref fields,
            ref integrator,
        } = *guard;

        let futs = islands.iter_mut().map(|island| {
            let fields = &*fields; // shared, copied into each closure
            let integrator = &*integrator;

            async move {
                // Snapshot of the island's S anchor before the disjoint borrow of the maps (Vector3<S> is Copy).
                let origin = island.origin();
                // Separate locals for the island's disjoint fields.
                let (bodies, joints, wrenches, solver) = island.with_mut();

                // --- Phase 1: pre-step (the body sees the island's S anchor). ---
                for e in bodies.values_mut() {
                    e.pre_step(epoch, &origin).await;
                }

                // --- Phase 2: zero the island's wrench buffer. The map is persistent
                // while the island topology is stable (keys are registered on body insertion),
                // so here only the values → zero, without clear/reallocation. ---
                for wrench in wrenches.values() {
                    wrench.write(Wrench::zero());
                }

                // --- Phase 4: fields.accumulate over the island's bodies. ---
                for field in fields.iter() {
                    field.accumulate(bodies, wrenches, epoch, &origin).await;
                }

                // --- Phase 5: coupled implicit integration of this island. ---
                integrator
                    .step_all(bodies, joints, wrenches, solver, epoch)
                    .await?;

                // --- Phase 5.5: re-anchor origin onto the COM (strict centre of mass). The bodies
                // moved during integration → the COM drifted from origin; restore origin≡COM
                // and shift the motors. Absolute positions are invariant. ---
                island.recompute_centroid();
                let origin = island.origin();

                // --- Phase 6: fields.publish over the island's bodies (BACK buffer), already with
                // the fresh anchor. ---
                let (bodies, _, _) = island.with();
                for field in fields.iter() {
                    field.publish(bodies, &origin);
                }
                Ok(())
            }
        });
        // Islands step concurrently; the first failed island fails the whole step.
        for r in join_all(futs).await {
            r?;
        }

        // --- Phase 7: publish the summary from the fresh (re-anchored) islands.
        // Still under `inner.write()`, but we write into OUR OWN locks synchronously — a summary
        // reader does not block on `inner`. ---
        self.store_snapshot(islands).await;
        Ok(())
    }
}

impl<T, S> Mechanism<T, S>
where
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
{
    /// Mass-weighted centre of mass of the mechanism in ABSOLUTE coordinates — the "centroid
    /// of centroids": inside an island the local offsets sum to zero (origin≡
    /// COM), so an island's contribution = `origin·M`. Returned in S, since absolute
    /// positions are ALWAYS expressed in S: cross-mechanism references (camera→target)
    /// take the anchor difference in S and only then lower the small remainder into T, avoiding
    /// catastrophic cancellation for scenes far from zero. `None` for an empty one.
    ///
    /// Reads the PUBLISHED snapshot from under its own lock — does NOT touch `inner`.
    /// The snapshot is updated in `step` (phase 7) and in all graph mutators, so
    /// an observer (the camera) can read it in the same async batch as the target's step.
    pub async fn centroid(&self) -> Option<Point<S>> {
        *self.centroid.read().await
    }

    /// Recompute the summary (CoM + world velocity of the CoM) from the islands and write it into
    /// the snapshot under THEIR locks. Called under the held `inner` lock from everywhere the
    /// composition/state of the mechanism changes. Writes synchronously (without await), so
    /// it does not block a snapshot reader for long.
    async fn store_snapshot(&self, islands: &Islands<T, S>) {
        *self.centroid.write().await = Self::compute_centroid(islands);
        *self.centroid_velocity.write().await = Self::compute_centroid_velocity(islands);
    }

    /// Pure computation of the CoM over islands (see `centroid` on the S semantics). `None`
    /// if there are no bodies or the total mass is effectively zero (e.g. all bodies are
    /// kinematic).
    fn compute_centroid(islands: &Islands<T, S>) -> Option<Point<S>> {
        let mut total_m = T::ZERO;
        let mut acc = Vector3::ZERO;
        let mut any = false;
        for (origin, m) in islands.island_anchors() {
            any = true;
            total_m += m;
            let ms = S::from(m);
            for k in 0..3 {
                acc[k] += origin[k] * ms;
            }
        }
        if !any || total_m.standard_part().is_effective_zero() {
            return None;
        }
        let denom = S::from(total_m);
        Some(Point::new(acc.scale(S::ONE / denom)))
    }

    /// Absolute position of a body: `island origin ⊕ local pose`. Convenient for
    /// rendering/observers reading poses (which are now LOCAL to origin).
    pub async fn body_absolute_position(&self, world_id: WorldId) -> Option<Vector3<T>> {
        self.body_world_point(world_id, Vector3::ZERO).await
    }

    /// Absolute pose of a body: `T(origin) ∘ local pose` (left spatial
    /// translation onto the island anchor). The rotation is the same, the translation is world-frame. For
    /// the view matrix of the camera eye (rendering from the absolute pose).
    pub async fn body_absolute_pose(&self, world_id: WorldId) -> Option<Motor<T>> {
        let guard = self.inner.read().await;
        let i = guard.islands.island_of(world_id)?;
        let o = guard.islands.origin_of(i);
        let o: [S; 3] = o.into();
        let o = Vector3::from(o.map(|v| v.into()));
        let body = guard.islands.get_body(world_id)?.body();
        let shift = Twist::new(&o, &Vector3::ZERO).exp(T::ONE);
        Some(body.pose.read().compose(&shift))
    }

    /// Absolute world point of a body-local offset `local` on a body:
    /// `island origin ⊕ pose·local`. For rendering rig markers/spokes (poses are now
    /// local to origin; origin is lowered into T — precision is sufficient for visualization).
    pub async fn body_world_point(
        &self,
        world_id: WorldId,
        local: Vector3<T>,
    ) -> Option<Vector3<T>> {
        let guard = self.inner.read().await;
        let i = guard.islands.island_of(world_id)?;
        let o = guard.islands.origin_of(i);
        let o: [S; 3] = o.into();
        let o = Vector3::from(o.map(|v| v.into()));
        let body = guard.islands.get_body(world_id)?.body();
        let p = body.pose.read().conjugate(&Point::new(local)).coords();
        Some(p + o)
    }

    /// World-frame linear velocity of the centroid, mass-weighted across all
    /// bodies. Angular part is always zero (callers consume only `.linear()`,
    /// and a meaningful centroid angular velocity for separately-oriented
    /// rigid bodies requires an effective inertia tensor we have no use for).
    ///
    /// Per-body order is deliberate: weight = m_k / M first (≤ 1, dimensionless),
    /// then world-frame velocity = world_momentum / m_k. Direct division each
    /// step — precomputed 1/M or 1/m_k would underflow under fixed-point
    /// I88F40 at solar-mass scales.
    ///
    /// Like `centroid`, reads the published snapshot from under its own lock — does not
    /// touch `inner`. The snapshot is updated in `step` (phase 7) and the graph mutators.
    pub async fn centroid_velocity(&self) -> Option<Twist<T>> {
        *self.centroid_velocity.read().await
    }

    /// Pure computation of the world velocity of the CoM over islands (see `centroid_velocity`
    /// on the order of division). `None` if there are no bodies.
    fn compute_centroid_velocity(islands: &Islands<T, S>) -> Option<Twist<T>> {
        if islands.is_empty() {
            return None;
        }

        // Kinematic reference bodies are not part of the mechanism's mass/momentum —
        // their velocity is prescribed, and mass()=0 would give 0·(1/0)=NaN below.
        let sum_m = islands
            .all_bodies()
            .filter(|e| !e.body().inertia.is_kinematic())
            .fold(T::ZERO, |acc, e| acc + e.body().inertia.mass());

        let mut lin = Vector3::ZERO;
        for e in islands.all_bodies() {
            if e.body().inertia.is_kinematic() {
                continue;
            }
            let mass = e.body().inertia.mass();
            let weight = mass / sum_m; // 1) m_k / M  (≤ 1, dimensionless)
            let wm = e.body().world_momentum().read().force();
            let v_world = wm.scale(T::ONE / mass);
            lin += v_world.scale(weight);
        }

        Some(Twist::new(&lin, &Vector3::ZERO))
    }

    pub fn id(&self) -> WorldId {
        self.id
    }
}

impl<T, S> Drop for Mechanism<T, S>
where
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
{
    /// Release every body this mechanism still owns, so its world ids may be
    /// added again.
    fn drop(&mut self) {
        let islands = &self.inner.get_mut().islands;
        let mut owned = owned();
        for body in islands.all_bodies() {
            owned.remove(&body.id());
        }
    }
}
