// SPDX-License-Identifier: MIT

use crate::{Component, ForceField, RigidBody};
use aristotle::{Epoch, WorldId, WorldKey};
use async_trait::async_trait;
use bytemuck::Pod;
use clifford::pga3::{Point, Wrench};
use indexmap::IndexMap;
use peano::prelude::*;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

// ============================================================================
// GRAVITY — cross-mechanism "all-with-all" gravitation propagator
// ============================================================================
//
// MODEL. A shared intermediary (Arc<GravityPropagator>) that mechanisms
// access via &self from any number of threads. Only the CHARGE flows through it
// (CoM + GM), never a RigidBody — encapsulation: the body does not stick out.
//
// TWO EPOCHS (double-buffered charges). publish/update write into BACK; reading a row
// (subscribe) goes from FRONT. The gravity of round t is computed from charges
// published in round t−1 — A ONE-ROUND LAG. This is not a delay but a natural
// consistent frame: rendering/UI/observers look at the previous round's front
// in parallel with new publications accumulating in the current round's back.
//
// TWO MUTATIONS, SEPARATED IN TIME AND IN MECHANISM:
//   * STRUCTURAL (resizing the cell/cursor plate on register) — RARE,
//     EXCLUSIVE, ONLY OUTSIDE THE STEP. Done by swapping the Arc pointer to a new
//     frozen plate. add/remove during simulation is a PROGRAMMER ERROR.
//   * CONTENT (tags/cursors/wrenches inside cells) — FREQUENT, ATOMIC, DURING THE
//     STEP. Happens inside the immutable plate, without locks.
// The RwLock sits ONLY on the plate's Arc pointer and is held for a nanosecond on .clone()
// in subscribe — NOT on the hot traversal. A client grabs the plate Arc ONCE in
// subscribe and traverses its row through it: the next epoch's resize will swap the
// pointer, while a live cursor finishes looking at its own plate (the Arc refcount keeps it
// alive).
//
// EXPLICIT EPOCH LIFECYCLE. The propagator does not detect "end of round" itself.
// An external caller calls advance_epoch exactly once between rounds:
// an atomic swap back↔front + plate reset (tags/cursors). In single-threaded
// use this is a call between accumulate_for; in multi-threaded use via
// aristotle::epoch — a broker callback registered in the EpochCoordinator,
// fired after all Epoch<T> are dropped. Round synchronization is moved entirely
// into the caller — the propagator remains pure storage.
//
// PAIR WRITE-BACK. Canonical key (lo,hi)=(min slot,max slot). A cell stores
// ONE wrench "force on lo from hi" + an atomic tag. Whoever gets there first claims it
// (CAS Vacant→InWork), computes, publishes (Release-store Done); the second reader
// takes −w. Charges are NOT duplicated into the cell — they live in the front snapshot, the cell
// refers to them by slots.
//
// ⚠ CONCURRENCY. unsafe — only the single-writer publish of a cell (UnsafeCell under
// a CAS tag, Release/Acquire pair). RACE CORRECTNESS IS NOT VERIFIED here —
// loom/miri is needed before merging. Single-threaded correctness is under tests; at N=1
// the protocol degenerates into a sequential one (InWork/Wait are unreachable).

// ── Charge ────────────────────────────────────────────────────────────────

/// Gravitational charge of a body: world CoM + GM (weight·G precomputed; zero = does not
/// gravitate). The position is stored in S (high precision): the pairwise distance between
/// distant islands is computed in S so as not to lose digits. `local` is the position
/// of the body relative to the origin of ITS OWN island (the moment arm, small, in T): from it
/// the row reader assembles the wrench in the island's local frame (force transport).
#[derive(Debug, Clone, Copy)]
pub struct GravityCharge<T: Ring, S: Ring = T> {
    pub com: Point<S>,
    pub local: Vector3<T>,
    pub grav: T,
    pub size: T,
    // /// Radius of the sphere of influence (broad-phase cutoff of distant ones before the expensive computation,
    // /// like `influence` in sim.rs from ACCEL_MIN). Disabled for now.
    // pub influence: T,
}

impl<T: Scalar, S: Scalar + From<T> + Into<T>> GravityCharge<T, S> {
    #[inline]
    pub fn new(com: Point<S>, local: Vector3<T>, grav: T, size: T) -> Self {
        Self {
            com,
            local,
            grav,
            size,
        }
    }

    #[inline]
    fn zero() -> Self {
        Self {
            com: Point::new(Vector3::ZERO),
            local: Vector3::ZERO,
            grav: T::ZERO,
            size: T::ONE,
        }
    }
}

// ── Pair physics ─────────────────────────────────────────────────────────────

/// Force on `lo` from `hi` as a vector in world axes (∝ force; the 1/G factor is removed by
/// the caller, since grav=Gm for both). Distance and direction are in S (distant
/// islands do not lose digits); `Wrench::from_line` is kept for the exact
/// scale (its |direction|=d turns k/d³ into k/d²), we take ONLY the
/// force part from it and lower it into T — the (huge) moment about the global zero is left alone,
/// the reader rebuilds it in the island's local frame. None — degeneracy
/// (coincident CoMs) or underflow.
#[inline]
fn pair_force<T: Scalar + StandardPart, S: Scalar + StandardPart + From<T> + Into<T>>(
    lo: &GravityCharge<T, S>,
    hi: &GravityCharge<T, S>,
) -> Option<Vector3<T>>
where
    <S as StandardPart>::Real: PartialOrd,
{
    let line = lo.com.join(&hi.com);
    let d = line.weight_norm();
    if d.standard_part().is_effective_zero() {
        return None;
    }
    let size_hi = S::from(hi.size);
    let g_lo = S::from(lo.grav);
    let g_hi = S::from(hi.grav);
    let g = if d.standard_part() >= size_hi.standard_part() {
        let g1 = g_lo / d;
        let g2 = g_hi / d;
        let mut tmp = g1 * g2;
        tmp = tmp / d;
        tmp
    } else {
        let g1 = g_lo / size_hi;
        let g2 = g_hi / size_hi;
        let mut tmp = g1 * g2;
        tmp = tmp / size_hi;
        tmp
    };
    let f = (Wrench::<S>::from_line(&line) * (-g)).force();
    Some(Vector3::from([S::into(f[0]), S::into(f[1]), S::into(f[2])]))
}

/// Moment of force `f` on arm `r` (both in the island's local frame): r × f.
#[inline]
fn moment<T: Scalar>(r: &Vector3<T>, f: &Vector3<T>) -> Vector3<T> {
    r.cross(*f)
}

// ── Tags ──────────────────────────────────────────────────────────────────────

mod tag {
    pub const VACANT: u8 = 0;
    pub const IN_WORK: u8 = 1;
    pub const DONE: u8 = 2;
    pub const NEGLIGIBLE: u8 = 3;
}

#[derive(Debug)]
struct Cell<T: Ring> {
    tag: AtomicU8,
    force: UnsafeCell<Vector3<T>>, // force on `lo` (world axes); valid when tag==DONE
}

impl<T: Scalar> Cell<T> {
    fn vacant() -> Self {
        Self {
            tag: AtomicU8::new(tag::VACANT),
            force: UnsafeCell::new(Vector3::ZERO),
        }
    }
    #[inline]
    fn reset(&self) {
        self.tag.store(tag::VACANT, Ordering::Relaxed);
    }
}

// SAFETY: exactly one thread writes `force` — the winner of CAS VACANT→IN_WORK;
// publication is tag.store(DONE,Release), reading is under tag.load(Acquire)==DONE.
unsafe impl<T: Scalar> Sync for Cell<T> {}

// ── Frozen plate: pair cells + opposing row cursors ───────────────────────────
//
// Immutable in STRUCTURE during an epoch (length and addresses do not move);
// only the contents of the atomics inside change. New size => new plate (outside
// the step). Cursors per row — a pair of opposing ones (lo crawls up, hi down), atomic
// fetch_add hands out cells among parallel traversers of one row.

#[derive(Debug)]
struct Plate<T: Ring> {
    n: u32,                    // number of slots the plate is built for
    cells: Vec<Cell<T>>,       // length tri(0,n)
    lo_cursor: Vec<AtomicU32>, // per row
    hi_cursor: Vec<AtomicU32>,
}

impl<T: Scalar> Plate<T> {
    fn new(n: u32) -> Self {
        let ncells = tri(0, n);
        let mut cells = Vec::with_capacity(ncells);
        for _ in 0..ncells {
            cells.push(Cell::vacant());
        }
        let mut lo_cursor = Vec::with_capacity(n as usize);
        let mut hi_cursor = Vec::with_capacity(n as usize);
        for _ in 0..n {
            lo_cursor.push(AtomicU32::new(0));
            hi_cursor.push(AtomicU32::new(0));
        }
        Self {
            n,
            cells,
            lo_cursor,
            hi_cursor,
        }
    }

    /// Reset the contents at the start of an epoch (the structure is not touched).
    fn reset(&self) {
        for c in &self.cells {
            c.reset();
        }
        for x in &self.lo_cursor {
            x.store(0, Ordering::Relaxed);
        }
        for x in &self.hi_cursor {
            x.store(0, Ordering::Relaxed);
        }
    }
}

// ── id↔slot register ──────────────────────────────────────────────────────────

#[derive(Debug)]
struct Registry {
    slots: HashMap<WorldId, u32>,
    free: Vec<u32>,
    next: u32,
    live: u32,
}

impl Registry {
    fn new() -> Self {
        Self {
            slots: HashMap::new(),
            free: Vec::new(),
            next: 0,
            live: 0,
        }
    }
    fn add(&mut self, id: WorldId) -> u32 {
        let slot = self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        });
        self.slots.insert(id, slot);
        self.live += 1;
        slot
    }
    fn remove(&mut self, id: WorldId) {
        if let Some(slot) = self.slots.remove(&id) {
            self.free.push(slot);
            self.live -= 1;
        }
    }
    #[inline]
    fn slot(&self, id: WorldId) -> Option<u32> {
        self.slots.get(&id).copied()
    }
}

#[inline]
fn tri(lo: u32, hi: u32) -> usize {
    let (lo, hi) = (lo as usize, hi as usize);
    hi * (hi.wrapping_sub(1)) / 2 + lo
}

// ── Cache ─────────────────────────────────────────────────────────────────────

/// Charges hold their position in S (`com: Point<S>`): the pairwise distance between
/// distant islands is computed in high precision, and the force is lowered into T.
/// By default `S = T` (single-precision mode — everything as before).
#[derive(Debug)]
pub struct GravityPropagator<T: Ring, S: Ring = T> {
    /// G and 1/G side by side: the first is needed in `publish` for `charge_of(body, g)`, the second
    /// inside the pairwise computation (the charges already carry `Gm`, we divide back once).
    g: T,
    g_inv: T,

    reg: RwLock<Registry>,

    charges: [RwLock<Vec<GravityCharge<T, S>>>; 2],
    front: AtomicU8,

    /// The cell/cursor plate. The RwLock is ONLY on the pointer: the read lock is held for the
    /// .clone() of the Arc in subscribe (nanoseconds), the write lock for a resize outside the step.
    /// Row traversal goes over the captured Arc WITHOUT a lock.
    plate: RwLock<Arc<Plate<T>>>,
}

impl<T: Scalar + StandardPart + Pod, S: Scalar + StandardPart + From<T> + Into<T>>
    GravityPropagator<T, S>
where
    <T as StandardPart>::Real: PartialOrd,
    <S as StandardPart>::Real: PartialOrd,
{
    pub fn new(g: T) -> Self {
        Self {
            g,
            g_inv: T::ONE / g,
            reg: RwLock::new(Registry::new()),
            charges: [RwLock::new(Vec::new()), RwLock::new(Vec::new())],
            front: AtomicU8::new(0),
            plate: RwLock::new(Arc::new(Plate::new(0))),
        }
    }

    // ── Registration (OUTSIDE the step) ──────────────────────────────────────

    /// Register a body as gravitating. Grows the charge buffers and BUILDS
    /// A NEW PLATE of the required size (a structural mutation — an Arc swap). Not on
    /// the hot path; do NOT call during a step.
    pub fn register(&self, id: WorldId) -> u32 {
        let slot = self.reg.write().unwrap().add(id);
        let cap = self.reg.read().unwrap().next;
        for buf in &self.charges {
            let mut b = buf.write().unwrap();
            if b.len() < cap as usize {
                b.resize(cap as usize, GravityCharge::zero());
            }
        }
        // a new frozen plate for the new size (tags are reset anyway
        // at the start of the epoch, there is no point copying the old contents)
        let need_n = {
            let p = self.plate.read().unwrap();
            p.n
        };
        if cap > need_n {
            *self.plate.write().unwrap() = Arc::new(Plate::new(cap));
        }
        slot
    }

    pub fn unregister(&self, id: WorldId) {
        self.reg.write().unwrap().remove(id);
    }

    // ── publish/update: BACK ─────────────────────────────────────────────────

    /// Publish/update a body's charge in BACK (visible on the next step after
    /// the swap). Single-writer-per-slot: only the owner of the id, for its own body.
    pub fn update(&self, id: WorldId, charge: GravityCharge<T, S>) {
        let slot = match self.reg.read().unwrap().slot(id) {
            Some(s) => s as usize,
            None => return,
        };
        let back = 1 - self.front.load(Ordering::Acquire);
        let mut buf = self.charges[back as usize].write().unwrap();
        if slot < buf.len() {
            buf[slot] = charge;
        }
    }

    // ── Explicit epoch lifecycle ────────────────────────────────────────────
    //
    // The propagator does not detect "end of round" itself. An external lifecycle (a test
    // or aristotle::epoch::EpochCoordinator) calls advance_epoch exactly once
    // between rounds: an atomic swap back↔front + plate reset. This
    // moves synchronization out of the propagator into the caller and makes the round boundary
    // explicit — where the old refcount barrier gave a spurious lag for the
    // single-mechanism case.
    //
    // IMPORTANT: the swap by itself gives the shift "published in BACK → visible in FRONT
    // one round later". This does NOT mean lag-1 in the dynamics: Mechanism::step publishes
    // charges AFTER integration (phase 6), so after the swap the accumulate of
    // the next step reads the charges of the CURRENT positions — the coupling is synchronous,
    // F(x_k) is computed from the same x_k. See mechanism.rs and the regression test
    // tests/energy_drift_repro.rs (previously publish ran BEFORE accumulate, which is what
    // produced the anti-damping lag-1 that pumped orbital energy).

    /// Move the propagator to the next round: atomic swap back↔front
    /// of the charges + plate reset (tags → VACANT, cursors → 0). After the call
    /// `subscribe`/`accumulate_for` read the freshly published charges from
    /// front; the new back is ready to accept the next round's publications.
    ///
    /// **Lifecycle contract.** Called BETWEEN rounds — when no
    /// `RowCursor` of the propagator is alive on any thread. In single-threaded
    /// use — between calls to `accumulate_for`. In multi-threaded use
    /// via `aristotle::epoch` — registered as a broker in
    /// `EpochCoordinator`:
    /// `Builder::new().with_broker(move |_dt| prop.advance_epoch()).build(N)`
    /// — fired at the epoch boundary after all `Epoch<T>` are dropped.
    pub fn advance_epoch(&self) {
        let f = self.front.load(Ordering::Acquire);
        self.front.store(1 - f, Ordering::Release);
        let plate = self.plate.read().unwrap().clone();
        plate.reset();
    }

    /// Read the charge of body `id` from the FRONT snapshot. `None` if the body is not
    /// registered in this propagator. A pure read, without locks on
    /// the hot path (a read lock for a nanosecond).
    pub fn charge_of_id(&self, id: WorldId) -> Option<GravityCharge<T, S>> {
        let slot = self.reg.read().unwrap().slot(id)? as usize;
        let f = self.front.load(Ordering::Acquire);
        let buf = self.charges[f as usize].read().unwrap();
        buf.get(slot).copied()
    }

    // ── subscribe + accumulation ────────────────────────────────────────────

    /// Open a cursor over the row of body `me`. Grabs the plate Arc and the snapshot of
    /// the front charges ONCE — traversal without locks. Front/plate are assumed
    /// ready (the last `advance_epoch` has already published the needed charges).
    pub fn subscribe(&self, me: WorldId) -> Option<RowCursor<T, S>> {
        let my_slot = self.reg.read().unwrap().slot(me)?;
        let f = self.front.load(Ordering::Acquire);
        let front_charges = self.charges[f as usize].read().unwrap().clone();
        let plate = self.plate.read().unwrap().clone(); // Arc clone, lock for a nanosecond
        let n = front_charges.len() as u32;
        // Moment arm for `me` — its local position relative to the origin of its own
        // island (constant along the whole row).
        let my_local = front_charges
            .get(my_slot as usize)
            .map(|c| c.local)
            .unwrap_or(Vector3::ZERO);
        Some(RowCursor {
            g_inv: self.g_inv,
            my_slot,
            my_local,
            front_charges,
            plate,
            n,
            other: 0,
        })
    }

    /// Walk the whole row (spinning on Wait), return the total world wrench on `me`.
    pub fn accumulate_for(&self, me: WorldId) -> Wrench<T> {
        let mut acc = Wrench::zero();
        if let Some(mut cur) = self.subscribe(me) {
            loop {
                match cur.advance() {
                    Step::Force(w) => acc += w,
                    Step::Wait => std::hint::spin_loop(),
                    Step::Drained => break,
                }
            }
        }
        acc
    }

    /// Total world wrench on a PROBE charge `probe` from all registered
    /// bodies (front snapshot). The probe does NOT have to be registered and does not take part
    /// in the force exchange — this is a pure point query of the field for observers
    /// (the camera) and trajectory prediction. The lag-1 is the same as for `accumulate_for`
    /// (the same front is read). Zero charges (weight 0) are filtered out by `pair_wrench`.
    pub fn force_on_probe(&self, probe: &GravityCharge<T, S>) -> Wrench<T> {
        let f = self.front.load(Ordering::Acquire);
        let front = self.charges[f as usize].read().unwrap();
        let mut acc = Vector3::ZERO;
        for c in front.iter() {
            if let Some(raw) = pair_force(probe, c) {
                for k in 0..3 {
                    acc[k] += raw[k] * self.g_inv;
                }
            }
        }
        // The force is applied at the probe position; the moment about its island's origin = r × F.
        let m = moment(&probe.local, &acc);
        Wrench::new(&acc, &m)
    }

    /// A body's charge under this propagator's G in the S frame of its island (`origin`) —
    /// a wrapper over the free `charge_of`, so that the caller does not have to keep G around.
    pub fn charge_of(&self, e: &dyn Component<T, S>, origin: &Vector3<S>) -> GravityCharge<T, S> {
        Self::charge_of_impl(e.body(), self.g, origin)
    }

    pub fn charge_of_body(&self, body: &RigidBody<T>, origin: &Vector3<S>) -> GravityCharge<T, S> {
        Self::charge_of_impl(body, self.g, origin)
    }

    /// Charge from a RigidBody: absolute CoM in S (origin ⊕ raise(local)) + G·m, plus
    /// the local position in T (moment arm). Called by the body owner on publish.
    fn charge_of_impl(body: &RigidBody<T>, g: T, origin: &Vector3<S>) -> GravityCharge<T, S> {
        let local = body.position();
        let com = Point::<S>::new(
            Vector3::from([S::from(local[0]), S::from(local[1]), S::from(local[2])]) + *origin,
        );
        GravityCharge::new(
            com,
            local,
            g * body.inertia.mass(),
            body.effective_size.read(),
        )
    }
}

// ── Row cursor ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum Step<T: Ring> {
    Force(Wrench<T>),
    Wait,
    Drained,
}

/// Traversal of the row of body `me`. Holds the plate Arc (the structure is frozen for the epoch) and
/// a snapshot of the front charges. `next()` takes the next pair, computes/reads its force,
/// and assembles the wrench on `me` in its island's local frame (moment = r_me × F).
#[derive(Debug)]
pub struct RowCursor<T: Scalar, S: Scalar + From<T> = T> {
    g_inv: T,
    my_slot: u32,
    /// Local position of `me` (the moment arm in its island's frame).
    my_local: Vector3<T>,
    front_charges: Vec<GravityCharge<T, S>>,
    plate: Arc<Plate<T>>,
    n: u32,
    other: u32,
}

impl<T: Scalar + StandardPart, S: Scalar + StandardPart + From<T> + Into<T>> RowCursor<T, S>
where
    <S as StandardPart>::Real: PartialOrd,
{
    /// Named `advance`, not `next`: it yields a `Step`, whose `Drained` variant
    /// is the terminator, so it is not an `Iterator::next` and must not read like
    /// one.
    pub fn advance(&mut self) -> Step<T> {
        loop {
            if self.other >= self.n {
                return Step::Drained;
            }
            let other = self.other;
            self.other += 1;
            if other == self.my_slot {
                continue;
            }

            let (lo, hi) = if self.my_slot < other {
                (self.my_slot, other)
            } else {
                (other, self.my_slot)
            };
            let cell = &self.plate.cells[tri(lo, hi)]; // direct index, no lock

            match cell.tag.load(Ordering::Acquire) {
                tag::DONE => {
                    // SAFETY: DONE was published by a Release-store after writing force.
                    let f = unsafe { *cell.force.get() };
                    return Step::Force(self.wrench_for(lo, f));
                }
                tag::NEGLIGIBLE => continue,
                tag::IN_WORK => return Step::Wait,
                _ => {
                    if cell
                        .tag
                        .compare_exchange(
                            tag::VACANT,
                            tag::IN_WORK,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        let c_lo = &self.front_charges[lo as usize];
                        let c_hi = &self.front_charges[hi as usize];
                        match pair_force(c_lo, c_hi) {
                            Some(raw) => {
                                let f = raw.scale(self.g_inv);
                                // SAFETY: the only writer (won the CAS).
                                unsafe { *cell.force.get() = f };
                                cell.tag.store(tag::DONE, Ordering::Release);
                                return Step::Force(self.wrench_for(lo, f));
                            }
                            None => {
                                cell.tag.store(tag::NEGLIGIBLE, Ordering::Release);
                                continue;
                            }
                        }
                    }
                    self.other -= 1; // lost the CAS — re-read the cell
                }
            }
        }
    }

    /// Assemble the wrench on `me` from the cell's shared force (`force_lo` — the force on `lo`).
    /// `me==lo` → +F, otherwise −F (the 3rd law on the vector). Moment = r_me × F about the origin
    /// of `me`'s island.
    #[inline]
    fn wrench_for(&self, lo: u32, force_lo: Vector3<T>) -> Wrench<T> {
        let f = if self.my_slot == lo {
            force_lo
        } else {
            -force_lo
        };
        let m = moment(&self.my_local, &f);
        Wrench::new(&f, &m)
    }
}

// ── ForceField: the propagator as a force field ───────────────────────────────
//
// TWO PHASES. `accumulate` reads the front snapshot and assembles pairwise forces;
// `publish` (AFTER integration, phase 6 in Mechanism::step) runs over the bodies
// of the mechanism and publishes their fresh charges into the back buffer. Between rounds an external
// `advance_epoch` swaps back→front. The user does not need to call
// publication manually — Mechanism does it itself via the two-phase ForceField protocol.
//
// In `accumulate`, `bodies` is used ONLY to enumerate WHICH world_ids
// to serve and into which `out[i]` to write. Force data (including OTHER bodies of other
// mechanisms) is taken from the propagator's front snapshot, NOT from `bodies`. Bodies not
// registered in the propagator give a zero contribution (accumulate_for → 0).
//
// SYNCHRONICITY. Since publish runs AFTER integration, after the swap the
// accumulate of the next step reads the charges of the positions at the START of that step — the coupling
// is synchronous (F(x_k) from the same x_k), there is no structural lag-1 (previously publish ran
// before accumulate and gave an anti-damping lag-1, see tests/energy_drift_repro.rs).
// Cross-mechanism, with a shared propagator all `step`s of a frame read one and the
// same front and write into the shared back; the single `advance_epoch` at the end of the frame
// swaps all at once — there is no inter-mechanism lag either. The third law is preserved
// structurally (a `Cell` stores ONE wrench per pair, the second reader takes −w).
// On the bootstrap frame the front is zero: a new body does not take part in gravitation for a step —
// this is accepted behavior, an initial charge publication is deliberately not done.
#[async_trait]
impl<T: Scalar + StandardPart + Pod, S: Scalar + StandardPart + From<T> + Into<T>> ForceField<T, S>
    for GravityPropagator<T, S>
where
    <T as StandardPart>::Real: PartialOrd,
    <S as StandardPart>::Real: PartialOrd,
{
    fn publish(&self, bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>, origin: &Vector3<S>) {
        for (id, e) in bodies {
            self.update(*id, self.charge_of(&**e, origin));
        }
    }

    async fn accumulate(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        out: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        _epoch: &Epoch<T>,
        _origin: &Vector3<S>,
    ) {
        for (world_id, _) in bodies.iter() {
            let o = &out[world_id];
            o.write(o.read() + self.accumulate_for(*world_id));
        }
    }
}

/// A bridge for shared ownership: the mechanism takes the field as `Box<Arc<…>>`, while
/// the owner-publisher holds a clone of the same `Arc` and calls `publish`/`register`.
/// One propagator — many mechanisms (cross-mechanism) + external access.
#[async_trait]
impl<T: Scalar + StandardPart + Pod, S: Scalar + StandardPart + From<T> + Into<T>> ForceField<T, S>
    for Arc<GravityPropagator<T, S>>
where
    <T as StandardPart>::Real: PartialOrd,
    <S as StandardPart>::Real: PartialOrd,
{
    fn publish(&self, bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>, origin: &Vector3<S>) {
        (**self).publish(bodies, origin)
    }

    async fn accumulate(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        out: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        epoch: &Epoch<T>,
        origin: &Vector3<S>,
    ) {
        (**self).accumulate(bodies, out, epoch, origin).await;
    }
}

// ============================================================================
// Tests (single-threaded; the protocol degenerates into a sequential one, Wait is unreachable)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Accelerator, Inert, Inertia, Integrator, Mechanism,
        integrator::{ImplicitIntegrator, SymplecticEuler},
    };
    use aristotle::World;
    use std::sync::Arc;

    fn accel(world: Arc<World>) -> Arc<Accelerator<f32>> {
        Arc::new(Accelerator::builder(world).build())
    }
    use clifford::pga3::Twist;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn body_at(world: Arc<World>, x: f32, y: f32, z: f32, mass: f32) -> RigidBody<f32> {
        let b = RigidBody::new(world.clone(), Inertia::isotropic(world, mass, 1.0));
        b.pose
            .write(Twist::new(&Vector3::from([x, y, z]), &Vector3::ZERO).exp(1.0));
        b
    }

    fn com_of(b: &RigidBody<f32>) -> Point<f32> {
        b.pose.read().conjugate(&Point::new(Vector3::ZERO))
    }

    #[test]
    fn charge_of_body_has_com_and_gm() {
        let world = Arc::new(World::builder().usual::<f32>());
        let b = body_at(world, 3.0, 0.0, 0.0, 2.0);
        let c = GravityPropagator::<f32>::charge_of_impl(&b, 1.0, &Vector3::ZERO);
        assert!(approx(c.com.coords()[0], 3.0, 1e-9));
        assert!(approx(c.grav, 2.0, 1e-9));
    }

    #[test]
    fn pair_attracts_inverse_square() {
        let g = 2.0;
        // at origin=0 the local position equals the absolute one
        let lo = GravityCharge::<f32>::new(Point::new(Vector3::ZERO), Vector3::ZERO, g, 1.0);
        let hi = GravityCharge::<f32>::new(
            Point::new(Vector3::from([3.0, 0.0, 0.0])),
            Vector3::from([3.0, 0.0, 0.0]),
            g,
            1.0,
        );
        let raw = pair_force(&lo, &hi).unwrap();
        let f = [raw[0] / g, raw[1] / g, raw[2] / g];
        assert!(f[0] > 0.0, "attraction toward +X: {f:?}");
        assert!(approx(f[0], 2.0 / 9.0, 1e-6), "{}", f[0]);
        assert!(approx(f[1], 0.0, 1e-6) && approx(f[2], 0.0, 1e-6))
    }

    #[test]
    fn coincident_none() {
        let p = Point::new(Vector3::from([1.0, 2.0, 3.0]));
        assert!(
            pair_force(
                &GravityCharge::<f32>::new(p, Vector3::from([1.0, 2.0, 3.0]), 1.0, 1.0),
                &GravityCharge::<f32>::new(p, Vector3::from([1.0, 2.0, 3.0]), 1.0, 1.0)
            )
            .is_none(),
        );
    }

    /// The plate survives a resize: we register a third body AFTER the start — an old
    /// cursor (if it were alive) would see the old plate; a new subscribe sees the new one.
    /// We check that after register the row sizes grow.
    #[test]
    fn plate_grows_on_register() {
        let cache = GravityPropagator::<f32>::new(1.0);
        let id1 = WorldId::get();
        let id2 = WorldId::get();
        cache.register(id1);
        cache.register(id2);
        let world = Arc::new(World::builder().usual::<f32>());
        cache.update(
            id1,
            cache.charge_of_body(&body_at(world.clone(), 0.0, 0.0, 0.0, 1.0), &Vector3::ZERO),
        );
        cache.update(
            id2,
            cache.charge_of_body(&body_at(world.clone(), 2.0, 0.0, 0.0, 1.0), &Vector3::ZERO),
        );
        assert_eq!(cache.plate.read().unwrap().n, 2);
        let id3 = WorldId::get();
        cache.register(id3);
        assert_eq!(cache.plate.read().unwrap().n, 3);
    }

    /// End-to-end third law through the cache (non-monotonic ids). publish → back; the first
    /// accumulate_for of the round swaps back→front and reads; the refcount closes the round
    /// on the second body.
    #[test]
    fn two_body_third_law_through_cache() {
        let g = 1.0;
        let world = Arc::new(World::builder().usual::<f32>());
        let a = body_at(world.clone(), -1.0, 0.0, 0.0, 1.0);
        let b = body_at(world.clone(), 1.0, 0.0, 0.0, 1.0);

        let cache = GravityPropagator::<f32>::new(g);
        let id1 = WorldId::get();
        let id2 = WorldId::get();
        cache.register(id1);
        cache.register(id2);
        cache.update(id1, cache.charge_of_body(&a, &Vector3::ZERO));
        cache.update(id2, cache.charge_of_body(&b, &Vector3::ZERO));

        // explicit lifecycle: after the publications advance_epoch swaps back→front,
        // and accumulate_for reads the fresh charges.
        cache.advance_epoch();
        let wa = cache.accumulate_for(id1);
        let wb = cache.accumulate_for(id2);

        for k in 0..3 {
            assert!(
                approx(wa.force()[k] + wb.force()[k], 0.0, 1e-12),
                "Σ force along axis {k}"
            );
            assert!(
                approx(wa.torque()[k] + wb.torque()[k], 0.0, 1e-12),
                "Σ moment along axis {k}"
            );
        }
        assert!(wa.force()[0] > 0.0, "10 (x=-1) toward +X");
        assert!(wb.force()[0] < 0.0, "20 (x=+1) toward −X");
    }

    #[test]
    fn unregistered_body_no_force() {
        let cache = GravityPropagator::<f32>::new(1.0);
        let id1 = WorldId::get();
        cache.register(id1);
        let world = Arc::new(World::builder().usual::<f32>());
        cache.update(
            id1,
            cache.charge_of_body(&body_at(world.clone(), 0.0, 0.0, 0.0, 1.0), &Vector3::ZERO),
        );
        cache.advance_epoch();
        let w = cache.accumulate_for(WorldId::get());
        for k in 0..3 {
            assert!(approx(w.force()[k], 0.0, 1e-15));
        }
    }

    #[test]
    fn charge_of_id_returns_published_com_from_front() {
        let g = 1.0;
        let cache = GravityPropagator::<f32>::new(g);
        let id = WorldId::get();
        cache.register(id);
        let world = Arc::new(World::builder().usual::<f32>());
        cache.update(
            id,
            cache.charge_of_body(&body_at(world.clone(), 3.0, 0.0, 0.0, 2.0), &Vector3::ZERO),
        );
        cache.advance_epoch();

        let read = cache
            .charge_of_id(id)
            .expect("registered id should be readable");
        assert!(approx(read.com.coords()[0], 3.0, 1e-12));
        assert!(approx(read.grav, 2.0, 1e-12));
    }

    #[test]
    fn charge_of_id_unknown_returns_none() {
        let cache = GravityPropagator::<f32>::new(1.0);
        let unknown = WorldId::get();
        assert!(cache.charge_of_id(unknown).is_none());
    }

    #[test]
    fn charge_of_id_before_advance_is_zero_default() {
        let cache = GravityPropagator::<f32>::new(1.0);
        let id = WorldId::get();
        cache.register(id);
        // No publish or advance_epoch — FRONT holds zero-initialised charge.
        let read = cache
            .charge_of_id(id)
            .expect("registered id should be Some");
        assert_eq!(read.grav, 0.0);
    }

    /// Three-body: the total wrench (force and moment about the origin) is zero — the third law
    /// over all pairs structurally.
    #[test]
    fn three_body_total_wrench_zero() {
        let g = 1.0;
        let world = Arc::new(World::builder().usual::<f32>());
        let bodies = [
            (WorldId::get(), body_at(world.clone(), 0.0, 0.0, 0.0, 1.0)),
            (WorldId::get(), body_at(world.clone(), 2.0, 0.0, 0.0, 2.0)),
            (WorldId::get(), body_at(world.clone(), 0.0, 2.0, 0.0, 3.0)),
        ];
        let cache = GravityPropagator::<f32>::new(g);
        for (id, _) in &bodies {
            cache.register(*id);
        }
        for (id, b) in &bodies {
            cache.update(*id, cache.charge_of_body(b, &Vector3::ZERO));
        }
        // explicit lifecycle: after all publications we swap, then read
        cache.advance_epoch();
        let mut sf = [0.0; 3];
        let mut st = [0.0; 3];
        for (id, _) in &bodies {
            let w = cache.accumulate_for(*id);
            for k in 0..3 {
                sf[k] += w.force()[k];
                st[k] += w.torque()[k];
            }
        }
        for k in 0..3 {
            assert!(approx(sf[k], 0.0, 1e-6), "Σ force {k}: {sf:?}");
            assert!(approx(st[k], 0.0, 1e-6), "Σ moment {k}: {st:?}");
        }
    }

    /// Infall: momentum is conserved to machine precision over the whole run. Each step: publish
    /// the current charges into back; two accumulate_for calls make up ONE round (the first
    /// swaps back→front and reads, the second closes the refcount). A one-step lag:
    /// the step's forces are computed from the charges published in the previous back.
    #[test]
    fn two_body_infall_momentum_conserved() {
        let g = 1.0;
        let world = Arc::new(World::builder().usual::<f32>());
        let a = body_at(world.clone(), -1.0, 0.0, 0.0, 1.0);
        let b = body_at(world.clone(), 1.0, 0.0, 0.0, 1.0);
        let cache = GravityPropagator::<f32>::new(g);
        let id1 = WorldId::get();
        let id2 = WorldId::get();
        cache.register(id1);
        cache.register(id2);

        let integ = SymplecticEuler;
        let dt = 0.001;

        for _ in 0..500 {
            let epoch = Epoch::standalone(dt, 1.0);
            cache.update(id1, cache.charge_of_body(&a, &Vector3::ZERO));
            cache.update(id2, cache.charge_of_body(&b, &Vector3::ZERO));
            // explicit lifecycle, lag-1: read front (publications of the previous round),
            // at the end of the iteration call advance_epoch for the next round.
            let wa = cache.accumulate_for(id1);
            let wb = cache.accumulate_for(id2);

            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);

            let pf = [
                a.world_momentum().read().force()[0] + b.world_momentum().read().force()[0],
                a.world_momentum().read().force()[1] + b.world_momentum().read().force()[1],
                a.world_momentum().read().force()[2] + b.world_momentum().read().force()[2],
            ];
            assert!(
                pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2] < 1e-12,
                "{pf:?}"
            );
            cache.advance_epoch();
        }
        assert!(com_of(&a).join(&com_of(&b)).weight_norm() < 2.0);
    }

    // ── Integration tests via Mechanism (field + step) ───────────────────────

    /// Gravity as a ForceField INSIDE one mechanism. Two bodies are registered
    /// in a shared propagator, which is also added to the mechanism as a field (via the Arc bridge).
    /// **No manual publications** in the hot loop: the two-phase protocol of
    /// `ForceField` publishes charges itself in `publish` (after integration),
    /// the next step's `accumulate` reads them. The bodies must approach each other, the world
    /// momentum must hold to machine precision (the third law via field orchestration).
    #[tokio::test]
    async fn gravity_field_inside_mechanism_pulls_bodies_together() {
        let g = 1.0;
        let prop = Arc::new(GravityPropagator::new(g));

        let world = Arc::new(World::builder().usual::<f32>());
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id1 = m
            .add_body(Inert::new(body_at(world.clone(), -1.0, 0.0, 0.0, 1.0)))
            .await;
        let id2 = m
            .add_body(Inert::new(body_at(world.clone(), 1.0, 0.0, 0.0, 1.0)))
            .await;
        prop.register(id1);
        prop.register(id2);
        m.add_field(Box::new(prop.clone())).await; // the mechanism owns the field; the Arc is shared

        let dist_abs = async |m: &Mechanism<f32, f32>| {
            let p1 = m.body_absolute_position(id1).await.unwrap();
            let p2 = m.body_absolute_position(id2).await.unwrap();
            ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt()
        };
        let dist0 = dist_abs(&m).await;

        let dt = 0.001;
        for _ in 0..400 {
            m.step(&Epoch::standalone(dt, 1.0)).await.unwrap(); // accumulate + integrate, then publishes charges ITSELF

            // the total world momentum stays at zero to machine precision
            let p1 = m
                .inspect_body(id1, async |b| b.world_momentum().read().force())
                .await
                .unwrap();
            let p2 = m
                .inspect_body(id2, async |b| b.world_momentum().read().force())
                .await
                .unwrap();
            let s = [p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]];
            assert!(
                s[0] * s[0] + s[1] * s[1] + s[2] * s[2] < 1e-12,
                "Σ momentum ≠ 0 through the mechanism: {s:?}"
            );
            prop.advance_epoch();
        }

        let dist1 = dist_abs(&m).await;
        assert!(
            dist1 < dist0,
            "the bodies must approach each other: {dist0} → {dist1}"
        );
    }

    /// CROSS-MECHANISM gravity — the essence of the architecture. Two bodies in DIFFERENT
    /// mechanisms attract each other through ONE shared propagator. Each mechanism
    /// holds a clone of the same Arc as a field and sees the other body only through the front snapshot
    /// of the charges (the other mechanism's RigidBody is inaccessible to it). A body is attracted to
    /// a body that is not in its slice — so the force came from the shared cache.
    ///
    /// **No manual publications**: each `ma.step` / `mb.step` publishes
    /// its own bodies in `publish` (after integration) by itself. Both `step`s of a frame read
    /// one and the same front and write into the shared back; the single `advance_epoch`
    /// at the end of the frame swaps all at once — the coupling is synchronous, there is no inter-mechanism lag
    /// (checks `x1_a > x0_a`/`x1_b < x0_b`).
    #[tokio::test]
    async fn gravity_propagator_couples_two_mechanisms() {
        let g = 1.0;
        let prop = Arc::new(GravityPropagator::new(g));

        let world = Arc::new(World::builder().usual::<f32>());
        let ma =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id1 = ma
            .add_body(Inert::new(body_at(world.clone(), -2.0, 0.0, 0.0, 1.0)))
            .await;
        ma.add_field(Box::new(prop.clone())).await;
        prop.register(id1); // body of mechanism A

        let mb =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id2 = mb
            .add_body(Inert::new(body_at(world.clone(), 2.0, 0.0, 0.0, 1.0)))
            .await;
        mb.add_field(Box::new(prop.clone())).await;
        prop.register(id2); // body of mechanism B

        let x0_a = ma.body_absolute_position(id1).await.unwrap()[0];
        let x0_b = mb.body_absolute_position(id2).await.unwrap()[0];

        let dt = 0.002;
        for _ in 0..300 {
            // both mechanisms are stepped independently; each publishes its own charges
            ma.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            mb.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            prop.advance_epoch();
        }

        let x1_a = ma.body_absolute_position(id1).await.unwrap()[0];
        let x1_b = mb.body_absolute_position(id2).await.unwrap()[0];

        // body A (x=-2) is pulled toward +X (toward body B from ANOTHER mechanism);
        // body B (x=+2) is pulled toward −X (toward body A). Approach across the boundary.
        assert!(
            x1_a > x0_a,
            "A must move toward +X (toward the other body): {x0_a} → {x1_a}"
        );
        assert!(
            x1_b < x0_b,
            "B must move toward −X (toward the other body): {x0_b} → {x1_b}"
        );
        assert!(
            (x1_b - x1_a) < (x0_b - x0_a),
            "the inter-mechanism distance must shrink"
        );
    }
}
