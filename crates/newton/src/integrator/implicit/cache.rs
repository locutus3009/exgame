// SPDX-License-Identifier: MIT

//! Per-island solver state that survives between steps.
//!
//! The approximate inverse `X` is what makes the method affordable: between
//! steps the system matrix changes slowly, so the previous `X` is already
//! nearly its inverse and a small iteration budget suffices. From a cold seed
//! the same accuracy costs an order of magnitude more work.
//!
//! `X` is a HINT, never a correctness input. A stale `X`, one left over from a
//! different `dt`, or one from a different configuration cannot produce a wrong
//! answer — at worst the line search rejects the step and `dt` is subdivided.
//! So the only thing that ever forces a rebuild is a change of DIMENSION.
//!
//! This lives on `Island` rather than on `Newton` because the integrator is
//! shared across islands stepped concurrently, while each island hands out a
//! disjoint `&mut` — no lock, and no need for an island identity (there is
//! none: `Islands` is a positional `Vec` and repartition reshuffles it).
//! Repartition rebuilds the island, and the cache dies with it.

use super::block::{Block, zero_block};
use crate::GatherTerm;
use crate::accelerator::MessageKind;
use crate::accelerator::row::Row;
use crate::accelerator::rows;
use crate::accelerator::rows::body_post::GatheredRows;
use aristotle::{World, WorldId, WorldKey};
use bytemuck::Pod;
use clifford::pga3::Wrench as W;
use clifford::pga3::{Motor, Twist, Wrench};
use indexmap::IndexMap;
use joints::JointEdge;
use peano::prelude::*;
use std::sync::Arc;

/// One contribution to a matrix block: a joint's Jacobian plus which of its ends gives the ROW
/// of the block and which gives the COLUMN.
///
/// The direct analogue of `GatherTerm` for wrenches, one level up. Contributions to a block
/// ACCUMULATE (the diagonal collects from every incident joint plus the mass block),
/// so the kernels cannot write into the matrix directly — they write into their own slots, and
/// a separate stage sums them according to this table. The table is baked when the
/// topology changes: `order` and the structure do not change between mechanism rebuilds.
#[derive(Clone)]
pub(crate) struct BlockTerm<T: Scalar + Pod> {
    pub key: WorldKey<[[W<T>; 24]; 2]>,
    /// The joint end that gives the block row (0 = a, 1 = b).
    pub row_end: u8,
    /// The joint end that gives the block column.
    pub col_end: u8,
}

/// Trial step fractions the line search evaluates AT ONCE.
///
/// Eight is not a tuning knob picked for roundness: it is the ladder the search
/// already walked (`α = 1, ½, … 1/128`), so a wave of eight settles on exactly
/// the fraction the sequential loop would have settled on. Measured over the
/// whole newton suite (10 642 searches) the sequential loop needed 1.71 probes on
/// average — 85.6% accept at `α = 1`, but 7.7% exhaust the ladder and fall
/// through to subdivision — so a wave replaces 1.71 dispatch round-trips with 1.
///
/// Sixteen was measured and rejected: buckets 8..15 collect a flat ~20
/// acceptances each, which is the signature of a float-noise coin flip rather
/// than of a descent direction (at `α = 1/32768` the trial iterate is
/// indistinguishable from its base), and accepting such a step only postpones the
/// subdivision that actually fixes the span — the cloth suite got 24% slower.
/// Four was rejected too: it fails more often (854 vs 824 exhausted searches),
/// and under a wave the extra four lanes cost no latency at all.
pub(crate) const LANES: usize = 8;

/// One line-search lane's scratch — everything a VALUE-only probe writes, and
/// nothing else. The Jacobian, the system matrix and the approximate inverse are
/// shared: a probe never touches them.
///
/// Lane 0 holds CLONES of the body-, edge- and cache-owned slots, so it addresses
/// exactly the storage the solver used before there were lanes; lanes 1.. own
/// fresh slots. That is what lets the accepted lane be adopted by copying into
/// lane 0 and leaves every consumer downstream (`assemble`, `block_matvec`)
/// pointing where it always did.
pub(crate) struct LaneSlots<T: Scalar + Pod> {
    /// The trial iterate this lane evaluates, per body (kinematic bodies included:
    /// their prescribed velocity is an input to the force laws).
    pub(crate) vmid: IndexMap<WorldId, WorldKey<Twist<T>>>,
    /// PRE's outputs, per body.
    pub(crate) midpoint: IndexMap<WorldId, WorldKey<Motor<T>>>,
    pub(crate) solve_vel: IndexMap<WorldId, WorldKey<Twist<T>>>,
    /// The connection kernels' value output, per JOINT id.
    ///
    /// Held, not read. For lanes past 0 these slots are allocated here and
    /// nowhere else, and `rows::fixed::bake_joints` bakes only their
    /// `raw_index()` into the GLSL row — a raw slot number, not an owning key.
    /// `WorldKey`'s `Drop` frees the slot, so this map is what keeps the slots
    /// a baked row still points at alive for as long as the lane is. The
    /// clones in `gather_terms` cover most joints but not one whose both ends
    /// are outside `order`, so they are not a substitute.
    #[expect(
        dead_code,
        reason = "owns the lane's connection slots; see the comment above"
    )]
    pub(crate) conn: IndexMap<WorldId, WorldKey<[Wrench<T>; 2]>>,
    /// The gather's output, per body.
    pub(crate) total: IndexMap<WorldId, WorldKey<Wrench<T>>>,
    /// Gather rows, round-major and island-wide, baked against this lane's
    /// `conn` and `total`.
    pub(crate) gather: Vec<Arc<[WorldKey<Row>]>>,
    /// `Pre` rows for every body, one round.
    pub(crate) pre: Vec<Arc<[WorldKey<Row>]>>,
    /// Connection rows, value-only and with the Jacobian block, GROUPED BY
    /// KERNEL: a joint family is a different pipeline, so its rows cannot share a
    /// message. Same edges and same order in both, so the two differ only by the
    /// block slot the value kernel has no output for.
    pub(crate) plain: KernelRows,
    pub(crate) jacobian: KernelRows,
    /// POST rows, baked against this lane's `midpoint`/`solve_vel`/`total` and
    /// its residual triple.
    body_post: PostRows,
    /// The same stage with the GATHER folded in — `None` when some body has more
    /// incident connections than one row carries, in which case the two-stage
    /// path stands. `Some` means `dispatch` must NOT gather: the fused kernel
    /// sums the terms itself and never writes `total`.
    gathered: Option<GatheredRows>,
    /// Kept for the fused bake, which needs the same terms the gather rows use.
    gather_terms: Vec<Vec<GatherTerm<T>>>,
    external: Vec<WorldKey<Wrench<T>>>,
    /// The residual and its self-scale, in `order` index order.
    rhs: Arc<[WorldKey<Wrench<T>>]>,
    scale: Arc<[WorldKey<Wrench<T>>]>,
    /// The world mass block — `assemble` reads lane 0's.
    mass_block: Arc<[WorldKey<Block<T>>]>,
}

impl<T: Scalar + Pod> LaneSlots<T> {
    pub(crate) fn gathered_rows(&self) -> Option<&GatheredRows> {
        self.gathered.as_ref()
    }
    pub(crate) fn rhs(&self) -> &Arc<[WorldKey<Wrench<T>>]> {
        &self.rhs
    }
    pub(crate) fn scale(&self) -> &Arc<[WorldKey<Wrench<T>>]> {
        &self.scale
    }
}

/// Baked POST rows, one per body, each carrying the kernel variant its inertia
/// shape selects.
type PostRows = Vec<(rows::body_post::Variant, WorldKey<Row>)>;
/// Baked connection rows for one lane, grouped by kernel: a joint family is a
/// different pipeline, so its rows cannot share a message.
type KernelRows = Vec<(MessageKind, Arc<[WorldKey<Row>]>)>;
/// The same, merged across every lane — each kernel's rounds concatenated in
/// lane order.
type MergedKernelRows = Vec<(MessageKind, Vec<Arc<[WorldKey<Row>]>>)>;

/// Every lane's rows of a phase, concatenated — the form the line-search wave is
/// actually dispatched in.
///
/// Eight lanes driven as eight concurrent `dispatch` calls did NOT stay in step:
/// each awaited its own completion, they woke in whatever order the executor
/// chose, and the worker flushed half a wave before the rest arrived. Observed on
/// a 6×12 curtain as batches of 360 and 216 PRE rows (five lanes, then three)
/// where one batch of 576 was due, and the probe wave costing eight or nine
/// flushes instead of four.
///
/// Concatenating at BAKE time fixes the order by construction and costs nothing
/// per step. Lane order is the row order inside a merged round; every row writes
/// its own slots, so the lanes cannot interfere.
pub(crate) struct Wave {
    pub(crate) pre: Vec<Arc<[WorldKey<Row>]>>,
    pub(crate) plain: MergedKernelRows,
    pub(crate) jacobian: MergedKernelRows,
    pub(crate) gather: Vec<Arc<[WorldKey<Row>]>>,
    /// Rebuilt per span, alongside the lanes' own POST rows.
    pub(crate) body_post: PostRows,
    /// The fused stage across every lane, grouped by kernel. `None` when any lane
    /// fell back to the two-stage path.
    pub(crate) gathered: Option<GatheredRows>,
}

impl Wave {
    fn empty() -> Self {
        Self {
            pre: Vec::new(),
            plain: Vec::new(),
            jacobian: Vec::new(),
            gather: Vec::new(),
            body_post: Vec::new(),
            gathered: None,
        }
    }
}

/// Merge round `k` of every lane into one round.
fn merge_rounds<'a>(
    per_lane: impl Iterator<Item = &'a Vec<Arc<[WorldKey<Row>]>>>,
) -> Vec<Arc<[WorldKey<Row>]>> {
    let mut out: Vec<Vec<WorldKey<Row>>> = Vec::new();
    for lane in per_lane {
        for (k, round) in lane.iter().enumerate() {
            if out.len() == k {
                out.push(Vec::new());
            }
            out[k].extend(round.iter().cloned());
        }
    }
    out.into_iter().map(Arc::from).collect()
}

pub(crate) struct NewtonCache<T: Scalar + Pod> {
    /// Row and column index of each DYNAMIC body. Kinematic bodies are absent:
    /// their velocity is prescribed, so they are not unknowns.
    order: IndexMap<WorldId, usize>,
    /// For block row `i`, the block columns of `A` that can be nonzero: the
    /// body itself plus every body it shares a connection with. Used to skip
    /// known-zero terms in `A·X`.
    neighbours: Vec<Arc<[usize]>>,
    /// All block columns, `0..m` — what a dense product sums over.
    all_columns: Arc<[usize]>,
    /// Sparsity of the left factor for `A·X` and the dense list for `X·R` —
    /// also constant, also used to be built on every pass.
    sparse_rows: Arc<[Arc<[usize]>]>,
    dense_rows: Arc<[Arc<[usize]>]>,
    /// For each block `(i, j)` — the list of joint contributions. Indexing `i*m + j`.
    block_terms: Arc<[Arc<[BlockTerm<T>]>]>,
    /// Identifiers of the joints `block_terms` was baked from — compared
    /// so as not to rebuild the table on every step.
    baked_joints: Vec<WorldId>,
    a: Arc<[WorldKey<Block<T>>]>,
    x: Arc<[WorldKey<Block<T>>]>,
    x_next: Arc<[WorldKey<Block<T>>]>,
    /// The last `x` that belonged to an ACCEPTED step. The live `x` is scratch:
    /// it is mutated by every Newton iteration of every span, including spans
    /// that get rejected and subdivided. Without this snapshot a single
    /// diverged attempt would leave its garbage in the persistent cache, be
    /// inherited by the subdivision retry that was supposed to recover from it,
    /// and — because the cache outlives the step — poison the island forever.
    x_accepted: Arc<[WorldKey<Block<T>>]>,
    /// `seeded` as of the last accepted step, restored alongside `x_accepted`.
    accepted_seeded: bool,
    r: Arc<[WorldKey<Block<T>>]>,
    /// World mass block `𝕀_s` per body — the adjoint sandwich `Ad*∘I∘Ad⁻¹`,
    /// computed ONCE per pass. It is also the diagonal block of `A`, and also what the
    /// residual uses to get `P_s = 𝕀_s·V_s`: previously both phases built it independently.
    mass_block: Arc<[WorldKey<Block<T>>]>,
    /// Residual per body, in `order` index order.
    rhs: Arc<[WorldKey<Wrench<T>>]>,
    /// Residual self-scale per body: the linear part in `force`, the angular part in
    /// `torque` (within each triple the value is the same — it is the scale of the block,
    /// not a vector).
    scale: Arc<[WorldKey<Wrench<T>>]>,
    /// Newton correction per body, in `order` index order.
    dv: Arc<[WorldKey<Twist<T>>]>,

    // ── Baked GPU rows ───────────────────────────────────────────────────────
    // A row names concrete slots, so each product carries a set baked against the
    // CURRENT arrays and an `_alt` set baked against the swapped ones; `swap_x`
    // exchanges the pair together with the block arrays, so the live set always
    // matches what `x()` points at.
    ns_ax: Vec<Arc<[WorldKey<Row>]>>,     // A · X       -> R
    ns_ax_alt: Vec<Arc<[WorldKey<Row>]>>, // A · X_next  -> R
    ns_xr: Vec<Arc<[WorldKey<Row>]>>,     // X · R       -> X_next
    ns_xr_alt: Vec<Arc<[WorldKey<Row>]>>, // X_next · R  -> X
    matvec_rows: Vec<Arc<[WorldKey<Row>]>>,
    matvec_rows_alt: Vec<Arc<[WorldKey<Row>]>>,
    /// POST rows, split by inertia variant. Names only stable slots plus the two
    /// scalar slots, so baked in `sync`.
    body_post_rows: Vec<(rows::body_post::Variant, WorldKey<Row>)>,
    assemble_rows: Vec<Arc<[WorldKey<Row>]>>,
    /// Partial sums of `‖I − A·X‖²`, one per reduce row, and the rows producing
    /// them. This is what replaced reading all `m²` blocks of `R` back to the host
    /// on every Newton–Schulz iteration.
    r_partials: Arc<[WorldKey<T>]>,
    r_reduce: Vec<Arc<[WorldKey<Row>]>>,
    /// `‖X‖²_F` — the finiteness probe. A sum of squares is finite exactly when
    /// every component is, so one scalar replaces the `m²` block reads
    /// `checkpoint` used to make. Swaps with `x`, hence the `_alt` set.
    x_partials: Arc<[WorldKey<T>]>,
    x_reduce: Vec<Arc<[WorldKey<Row>]>>,
    x_reduce_alt: Vec<Arc<[WorldKey<Row>]>>,
    /// `x → x_accepted` and back. Both swap with `x`.
    publish_rows: Vec<Arc<[WorldKey<Row>]>>,
    publish_rows_alt: Vec<Arc<[WorldKey<Row>]>>,
    restore_rows: Vec<Arc<[WorldKey<Row>]>>,
    restore_rows_alt: Vec<Arc<[WorldKey<Row>]>>,
    /// `‖A‖²_F` and `‖𝕀_s‖²_F` partials, and the rows folding them.
    a_partials: Arc<[WorldKey<T>]>,
    a_reduce: Vec<Arc<[WorldKey<Row>]>>,
    mass_partials: Arc<[WorldKey<T>]>,
    mass_reduce: Vec<Arc<[WorldKey<Row>]>>,
    /// Per-step scalar slots the kernels read by index. Written once per dispatch;
    /// the rows that name them are baked with the topology.
    half_slot: Option<WorldKey<T>>,
    floor2_slot: Option<WorldKey<T>>,
    /// Per-dispatch scalars the fixed-arity kernels read by index. They used to
    /// ride the batch table as immediates, which is what kept those rows from
    /// being bakeable: they change every step while the topology does not.
    /// `−half²` and `−half`, the assembly's midpoint factors. Slots rather than
    /// row words: see `rows::assemble`.
    pose_factor_slot: Option<WorldKey<T>>,
    vel_factor_slot: Option<WorldKey<T>>,
    retraction_slot: Option<WorldKey<T>>,
    dt_slot: Option<WorldKey<T>>,
    warp_slot: Option<WorldKey<T>>,

    /// The `LANES` line-search lanes. Rebuilt with the topology; lane 0 aliases
    /// the body/edge/cache slots, so an empty `lanes` and a fresh one address the
    /// same storage.
    lanes: Vec<LaneSlots<T>>,
    /// The lanes' rows merged phase by phase — see `Wave`.
    wave: Wave,
    /// Body and joint id sets the lanes were baked against, so a change of either
    /// forces a rebake — the same guard `baked_joints` gives the block table.
    baked_lane_bodies: Vec<WorldId>,
    baked_lane_joints: Vec<WorldId>,

    /// False until `x` holds a usable approximate inverse.
    seeded: bool,
    /// Sub-steps taken by the last step on this island. Internal diagnostic
    /// only — deliberately not exposed.
    spans: usize,
}

impl<T: Scalar + Pod> NewtonCache<T> {
    /// An empty cache. Allocation waits for the first `sync`, which is the only
    /// place that knows the island's dimension.
    pub(crate) fn new() -> Self {
        Self {
            order: IndexMap::new(),
            neighbours: Vec::new(),
            all_columns: Arc::from(Vec::new()),
            sparse_rows: Arc::from(Vec::new()),
            dense_rows: Arc::from(Vec::new()),
            block_terms: Arc::from(Vec::new()),
            baked_joints: Vec::new(),
            a: Arc::from(Vec::new()),
            x: Arc::from(Vec::new()),
            x_next: Arc::from(Vec::new()),
            x_accepted: Arc::from(Vec::new()),
            accepted_seeded: false,
            r: Arc::from(Vec::new()),
            mass_block: Arc::from(Vec::new()),
            rhs: Arc::from(Vec::new()),
            scale: Arc::from(Vec::new()),
            dv: Arc::from(Vec::new()),
            ns_ax: Vec::new(),
            ns_ax_alt: Vec::new(),
            ns_xr: Vec::new(),
            ns_xr_alt: Vec::new(),
            matvec_rows: Vec::new(),
            matvec_rows_alt: Vec::new(),
            body_post_rows: Vec::new(),
            assemble_rows: Vec::new(),
            r_partials: Arc::from(Vec::new()),
            r_reduce: Vec::new(),
            x_partials: Arc::from(Vec::new()),
            x_reduce: Vec::new(),
            x_reduce_alt: Vec::new(),
            publish_rows: Vec::new(),
            publish_rows_alt: Vec::new(),
            restore_rows: Vec::new(),
            restore_rows_alt: Vec::new(),
            a_partials: Arc::from(Vec::new()),
            a_reduce: Vec::new(),
            mass_partials: Arc::from(Vec::new()),
            mass_reduce: Vec::new(),
            half_slot: None,
            floor2_slot: None,
            pose_factor_slot: None,
            vel_factor_slot: None,
            retraction_slot: None,
            dt_slot: None,
            warp_slot: None,
            lanes: Vec::new(),
            wave: Wave::empty(),
            baked_lane_bodies: Vec::new(),
            baked_lane_joints: Vec::new(),
            seeded: false,
            spans: 0,
        }
    }

    pub(crate) fn lanes(&self) -> &[LaneSlots<T>] {
        &self.lanes
    }

    pub(crate) fn wave(&self) -> &Wave {
        &self.wave
    }

    pub(crate) fn order(&self) -> &IndexMap<WorldId, usize> {
        &self.order
    }
    pub(crate) fn m(&self) -> usize {
        self.order.len()
    }
    pub(crate) fn a(&self) -> &Arc<[WorldKey<Block<T>>]> {
        &self.a
    }
    pub(crate) fn x(&self) -> &Arc<[WorldKey<Block<T>>]> {
        &self.x
    }
    pub(crate) fn rhs(&self) -> &Arc<[WorldKey<Wrench<T>>]> {
        &self.rhs
    }
    pub(crate) fn scale(&self) -> &Arc<[WorldKey<Wrench<T>>]> {
        &self.scale
    }
    /// Test-only view of the world mass block.
    #[cfg(test)]
    pub(crate) fn mass_block(&self) -> &Arc<[WorldKey<Block<T>>]> {
        &self.mass_block
    }
    pub(crate) fn dv(&self) -> &Arc<[WorldKey<Twist<T>>]> {
        &self.dv
    }
    /// Test-only view of the sparsity actually used by `A.X`, as opposed to the
    /// private field.
    #[cfg(test)]
    pub(crate) fn sparse_rows(&self) -> &Arc<[Arc<[usize]>]> {
        &self.sparse_rows
    }
    /// Rows for `A · X → R` (sparse left factor), live phase.
    pub(crate) fn ns_ax(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.ns_ax
    }
    /// Rows for `X · R → X_next` (dense, `two_minus` folded), live phase.
    pub(crate) fn ns_xr(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.ns_xr
    }
    pub(crate) fn matvec_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.matvec_rows
    }
    pub(crate) fn body_post_rows(&self) -> &[(rows::body_post::Variant, WorldKey<Row>)] {
        &self.body_post_rows
    }

    /// Rounds folding `R = A·X` into `r_partials`.
    pub(crate) fn r_reduce_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.r_reduce
    }

    /// Rounds folding `‖A‖²_F` and `‖𝕀_s‖²_F` into their partials. Independent of
    /// each other and of the Newton–Schulz products, so they ride whatever wave is
    /// convenient once `assemble` and `BodyPost` have written their inputs.
    /// Rounds folding `‖X‖²_F`, and the copies publishing / restoring the hint.
    pub(crate) fn x_reduce_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.x_reduce
    }
    pub(crate) fn publish_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.publish_rows
    }
    pub(crate) fn restore_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.restore_rows
    }

    /// `‖X‖²_F`, from the partials the reduce kernel wrote.
    pub(crate) fn x_norm2(&self) -> T {
        self.x_partials
            .iter()
            .fold(T::ZERO, |acc, k| acc + k.read())
    }

    pub(crate) fn a_reduce_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.a_reduce
    }
    pub(crate) fn mass_reduce_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.mass_reduce
    }

    /// `‖A‖²_F / ‖𝕀_s‖²_F` — how far the assembled system is from being pure
    /// inertia at the current sub-step.
    ///
    /// `A = 𝕀_s − half·D − half²·K`, so dividing by the mass block leaves
    /// `1 + O(half·ζω) + O((half·ω)²)`: in the stiff regime this IS `(half·ω)²`,
    /// the group that decides whether the step is solvable at all. Unlike anything
    /// read off a failed attempt, it is a property of the SYSTEM and is known
    /// before the attempt is made.
    ///
    /// Zero when there is nothing to measure (no dynamic bodies).
    pub(crate) fn stiffness(&self) -> T {
        let sum = |p: &Arc<[WorldKey<T>]>| p.iter().fold(T::ZERO, |acc, k| acc + k.read());
        let mass = sum(&self.mass_partials);
        if mass.standard_part().is_effective_zero() {
            return T::ZERO;
        }
        sum(&self.a_partials) / mass
    }

    /// `‖I − A·X‖²`, added up from the partials the reduce kernel wrote.
    ///
    /// Summed in row order, so it is deterministic — but NOT bit-identical to the
    /// old host loop, which associated the `m²` block contributions differently.
    /// Nothing downstream needs it to be: the only consumer compares consecutive
    /// values of this quantity against a factor-of-four margin.
    pub(crate) fn contraction(&self) -> T {
        self.r_partials
            .iter()
            .fold(T::ZERO, |acc, k| acc + k.read())
    }

    /// The assembly rows. Baked with the topology and never again: the only
    /// per-sub-step numbers they carry are the two factor SLOTS, whose values
    /// `publish_scalars` rewrites.
    pub(crate) fn assemble_rows(&self) -> &[Arc<[WorldKey<Row>]>] {
        &self.assemble_rows
    }

    /// Publish this dispatch's scalars into their slots. A handful of stores, off
    /// the per-invocation path.
    pub(crate) fn publish_scalars(&self, half: T, floor2: T) {
        self.half_slot.as_ref().unwrap().write(half);
        self.floor2_slot.as_ref().unwrap().write(floor2);
        // The midpoint scheme's column factors, derived here rather than in the
        // kernel: they are parameters of the scheme, and two stores per span is
        // nothing against making every invocation recompute them.
        self.pose_factor_slot
            .as_ref()
            .unwrap()
            .write(-(half * half));
        self.vel_factor_slot.as_ref().unwrap().write(-half);
    }

    /// The three scalars the fixed-arity kernels read out of `plain`. Written once
    /// per dispatch — three stores, off the per-invocation path.
    pub(crate) fn publish_dispatch_scalars(&self, retraction: T, dt: T, warp: T) {
        self.retraction_slot.as_ref().unwrap().write(retraction);
        self.dt_slot.as_ref().unwrap().write(dt);
        self.warp_slot.as_ref().unwrap().write(warp);
    }

    pub(crate) fn is_seeded(&self) -> bool {
        self.seeded
    }
    pub(crate) fn mark_seeded(&mut self) {
        self.seeded = true;
    }
    pub(crate) fn set_spans(&mut self, spans: usize) {
        self.spans = spans;
    }

    /// After `X ← X·(2I − R)` the new iterate is in `x_next`; make it current.
    /// A baked row names concrete slots, so the row sets that reference `x` must
    /// swap with it — otherwise the next product would read the stale inverse.
    pub(crate) fn swap_x(&mut self) {
        std::mem::swap(&mut self.x, &mut self.x_next);
        std::mem::swap(&mut self.ns_ax, &mut self.ns_ax_alt);
        std::mem::swap(&mut self.ns_xr, &mut self.ns_xr_alt);
        std::mem::swap(&mut self.matvec_rows, &mut self.matvec_rows_alt);
        std::mem::swap(&mut self.x_reduce, &mut self.x_reduce_alt);
        std::mem::swap(&mut self.publish_rows, &mut self.publish_rows_alt);
        std::mem::swap(&mut self.restore_rows, &mut self.restore_rows_alt);
    }

    /// Settle the flags after `restore_rows` has run. Called when a span does NOT
    /// commit, so a rejected attempt's iterate never survives it. The copy itself
    /// is a dispatch the caller issues — `solve_step` owns the accelerator.
    pub(crate) fn finish_rollback(&mut self) {
        self.seeded = self.accepted_seeded;
    }

    /// Bring the cache in line with the island's current dynamic-body set and
    /// connection graph. Cheap and idempotent when nothing changed: the
    /// dimension check short-circuits and the existing `X` is kept warm.
    ///
    /// `pairs` lists the connected body pairs by `WorldId`; ends that are not
    /// dynamic are ignored.
    pub(crate) fn sync(
        &mut self,
        world: &Arc<World>,
        dynamic: impl Iterator<Item = WorldId>,
        joints: &IndexMap<WorldId, JointEdge<T>>,
    ) {
        let order: IndexMap<WorldId, usize> = dynamic.enumerate().map(|(i, id)| (id, i)).collect();
        let m = order.len();

        // Dimension is the ONLY rebuild trigger. Same size, same buffers, and
        // `X` stays warm even if bodies were swapped underneath — a wrong hint
        // costs a rejected line-search step, never a wrong answer.
        let rebuilt = m != self.m();
        if rebuilt {
            let mut a = Vec::with_capacity(m * m);
            let mut x = Vec::with_capacity(m * m);
            let mut x_next = Vec::with_capacity(m * m);
            let mut x_accepted = Vec::with_capacity(m * m);
            let mut r = Vec::with_capacity(m * m);
            {
                let mut map = world.write::<Block<T>>();
                for _ in 0..(m * m) {
                    a.push(map.add(zero_block::<T>()));
                    x.push(map.add(zero_block::<T>()));
                    x_next.push(map.add(zero_block::<T>()));
                    x_accepted.push(map.add(zero_block::<T>()));
                    r.push(map.add(zero_block::<T>()));
                }
            }
            let mut mass_block = Vec::with_capacity(m);
            {
                let mut map = world.write::<Block<T>>();
                for _ in 0..m {
                    mass_block.push(map.add(zero_block::<T>()));
                }
            }
            let mut rhs = Vec::with_capacity(m);
            let mut scale = Vec::with_capacity(m);
            {
                let mut map = world.write::<Wrench<T>>();
                for _ in 0..m {
                    rhs.push(map.add(Wrench::zero()));
                    scale.push(map.add(Wrench::zero()));
                }
            }
            let mut dv = Vec::with_capacity(m);
            {
                let mut map = world.write::<Twist<T>>();
                for _ in 0..m {
                    dv.push(map.add(Twist::zero()));
                }
            }
            self.a = Arc::from(a);
            self.x = Arc::from(x);
            self.x_next = Arc::from(x_next);
            self.x_accepted = Arc::from(x_accepted);
            self.r = Arc::from(r);
            self.mass_block = Arc::from(mass_block);
            self.rhs = Arc::from(rhs);
            self.scale = Arc::from(scale);
            self.dv = Arc::from(dv);
            {
                let mut map = world.write::<T>();
                self.half_slot = Some(map.add(T::ZERO));
                self.floor2_slot = Some(map.add(T::ZERO));
                self.pose_factor_slot = Some(map.add(T::ZERO));
                self.vel_factor_slot = Some(map.add(T::ZERO));
            }
            self.all_columns = Arc::from((0..m).collect::<Vec<_>>());
            // The contraction measure's partials, and the rows that fold `R` into
            // them. `R` is `m²` blocks whose target is the identity on the diagonal
            // — the cell index decides that, so the kernel never learns about `m`.
            let partials = {
                let mut map = world.write::<T>();
                (0..rows::reduce::partials_for(m * m))
                    .map(|_| map.add(T::ZERO))
                    .collect::<Vec<_>>()
            };
            self.r_partials = Arc::from(partials);
            self.r_reduce = rows::reduce::bake(
                world,
                &self.r,
                |k| k / m.max(1) == k % m.max(1),
                &self.r_partials,
            );
            // The stiffness signal's two norms. `diag = 0` everywhere, so the same
            // kernel yields a plain Frobenius norm rather than a distance from the
            // identity.
            let alloc = |n: usize| -> Arc<[WorldKey<T>]> {
                let mut map = world.write::<T>();
                Arc::from(
                    (0..rows::reduce::partials_for(n))
                        .map(|_| map.add(T::ZERO))
                        .collect::<Vec<_>>(),
                )
            };
            self.a_partials = alloc(m * m);
            self.mass_partials = alloc(m);
            self.x_partials = alloc(m * m);
            self.x_reduce = rows::reduce::bake(world, &self.x, |_| false, &self.x_partials);
            self.x_reduce_alt =
                rows::reduce::bake(world, &self.x_next, |_| false, &self.x_partials);
            self.publish_rows = rows::copy::bake(world, &self.x_accepted, &self.x);
            self.publish_rows_alt = rows::copy::bake(world, &self.x_accepted, &self.x_next);
            self.restore_rows = rows::copy::bake(world, &self.x, &self.x_accepted);
            self.restore_rows_alt = rows::copy::bake(world, &self.x_next, &self.x_accepted);
            self.a_reduce = rows::reduce::bake(world, &self.a, |_| false, &self.a_partials);
            self.mass_reduce =
                rows::reduce::bake(world, &self.mass_block, |_| false, &self.mass_partials);
            self.seeded = false;
            self.accepted_seeded = false;
        }

        // Rebuild the structure only when the topology changes: the joint set and
        // the dimension are constant between mechanism rebuilds.
        let ids: Vec<WorldId> = joints.keys().copied().collect();
        if rebuilt || ids != self.baked_joints {
            // Sparsity of `A`: the diagonal plus one block per joint end.
            let mut adjacency: Vec<Vec<usize>> = (0..m).map(|i| vec![i]).collect();
            // Contribution table: a joint gives a block for every pair (row end,
            // column end) whose both ends are dynamic.
            let mut terms: Vec<Vec<BlockTerm<T>>> = (0..m * m).map(|_| Vec::new()).collect();
            for e in joints.values() {
                let ends = [e.a(), e.b()];
                if let (Some(&ia), Some(&ib)) = (order.get(&ends[0]), order.get(&ends[1])) {
                    if !adjacency[ia].contains(&ib) {
                        adjacency[ia].push(ib);
                    }
                    if !adjacency[ib].contains(&ia) {
                        adjacency[ib].push(ia);
                    }
                }
                for (row_end, row_id) in ends.iter().enumerate() {
                    let Some(&row) = order.get(row_id) else {
                        continue; // kinematic end: no row
                    };
                    for (col_end, col_id) in ends.iter().enumerate() {
                        let Some(&col) = order.get(col_id) else {
                            continue; // kinematic end: not an unknown
                        };
                        terms[row * m + col].push(BlockTerm {
                            key: e.jacobian_key(),
                            row_end: row_end as u8,
                            col_end: col_end as u8,
                        });
                    }
                }
            }
            self.neighbours = adjacency.into_iter().map(Arc::from).collect();
            self.sparse_rows = Arc::from(self.neighbours.clone());
            self.dense_rows =
                Arc::from((0..m).map(|_| self.all_columns.clone()).collect::<Vec<_>>());
            self.block_terms = Arc::from(
                terms
                    .into_iter()
                    .map(Arc::from)
                    .collect::<Vec<Arc<[BlockTerm<T>]>>>(),
            );
            self.baked_joints = ids;

            // GPU rows that depend only on the cache's own arrays and the just-built
            // sparsity — bake them here, both swap phases (see `swap_x`). The
            // assembly rows join them now that their midpoint factors are slots.
            self.ns_ax = rows::gemm::bake(
                world,
                &self.a,
                &self.x,
                &self.r,
                m,
                &self.sparse_rows,
                false,
            );
            self.ns_ax_alt = rows::gemm::bake(
                world,
                &self.a,
                &self.x_next,
                &self.r,
                m,
                &self.sparse_rows,
                false,
            );
            self.ns_xr = rows::gemm::bake(
                world,
                &self.x,
                &self.r,
                &self.x_next,
                m,
                &self.dense_rows,
                true,
            );
            self.ns_xr_alt = rows::gemm::bake(
                world,
                &self.x_next,
                &self.r,
                &self.x,
                m,
                &self.dense_rows,
                true,
            );
            self.matvec_rows = rows::matvec::bake(world, &self.x, &self.rhs, &self.dv, m);
            self.matvec_rows_alt = rows::matvec::bake(world, &self.x_next, &self.rhs, &self.dv, m);
            self.assemble_rows = rows::assemble::bake(
                world,
                &self.block_terms,
                &self.mass_block,
                &self.a,
                m,
                self.pose_factor_slot.as_ref().unwrap(),
                self.vel_factor_slot.as_ref().unwrap(),
            );
        }
        self.order = order;
    }

    /// Allocate and bake the line-search lanes. Separate from `sync` because it
    /// needs the bodies and the island's external-wrench map, which `sync` does
    /// not see; guarded by the same "did the topology move" test, since every slot
    /// a lane row names is stable exactly while the topology is.
    ///
    /// Lane 0 is built from CLONES of the body-, edge- and cache-owned slots. That
    /// is deliberate: everything downstream of the search — `assemble` reading the
    /// mass block, `block_matvec` reading the residual, `commit_span` reading the
    /// iterate — keeps addressing the storage it always did, and adopting an
    /// accepted lane is a copy into lane 0 rather than a rebake of anything.
    pub(crate) fn bake_lanes<S: Ring>(
        &mut self,
        world: &Arc<World>,
        bodies: &IndexMap<WorldId, Box<dyn crate::Component<T, S>>>,
        joints: &IndexMap<WorldId, JointEdge<T>>,
        external: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        n_lanes: usize,
    ) {
        let body_ids: Vec<WorldId> = bodies.keys().copied().collect();
        let joint_ids: Vec<WorldId> = joints.keys().copied().collect();
        if !self.lanes.is_empty()
            && body_ids == self.baked_lane_bodies
            && joint_ids == self.baked_lane_joints
        {
            return;
        }
        self.baked_lane_bodies = body_ids;
        self.baked_lane_joints = joint_ids;
        if self.retraction_slot.is_none() {
            let mut map = world.write::<T>();
            self.retraction_slot = Some(map.add(T::ZERO));
            self.dt_slot = Some(map.add(T::ZERO));
            self.warp_slot = Some(map.add(T::ZERO));
        }
        let (retraction, dt, warp) = (
            self.retraction_slot.clone().unwrap(),
            self.dt_slot.clone().unwrap(),
            self.warp_slot.clone().unwrap(),
        );

        self.lanes = (0..n_lanes)
            .map(|lane| {
                // Lane 0 aliases; the rest allocate.
                let (vmid, midpoint, solve_vel, total) = if lane == 0 {
                    (
                        bodies
                            .iter()
                            .map(|(id, e)| (*id, e.body().world_velocity_slot()))
                            .collect(),
                        bodies
                            .iter()
                            .map(|(id, e)| (*id, e.body().midpoint_pose.clone()))
                            .collect(),
                        bodies
                            .iter()
                            .map(|(id, e)| (*id, e.body().solve_vel.clone()))
                            .collect(),
                        bodies
                            .iter()
                            .map(|(id, e)| (*id, e.body().total_wrench.clone()))
                            .collect(),
                    )
                } else {
                    let mut tw = world.write::<Twist<T>>();
                    let vmid: IndexMap<_, _> = bodies
                        .keys()
                        .map(|id| (*id, tw.add(Twist::zero())))
                        .collect();
                    let solve_vel: IndexMap<_, _> = bodies
                        .keys()
                        .map(|id| (*id, tw.add(Twist::zero())))
                        .collect();
                    drop(tw);
                    let mut mo = world.write::<Motor<T>>();
                    let midpoint: IndexMap<_, _> = bodies
                        .keys()
                        .map(|id| (*id, mo.add(Motor::identity())))
                        .collect();
                    drop(mo);
                    let mut wr = world.write::<Wrench<T>>();
                    let total: IndexMap<_, _> = bodies
                        .keys()
                        .map(|id| (*id, wr.add(Wrench::zero())))
                        .collect();
                    (vmid, midpoint, solve_vel, total)
                };
                let conn: IndexMap<WorldId, WorldKey<[Wrench<T>; 2]>> = if lane == 0 {
                    joints.iter().map(|(id, e)| (*id, e.wrench_key())).collect()
                } else {
                    let mut map = world.write::<[Wrench<T>; 2]>();
                    joints
                        .keys()
                        .map(|id| (*id, map.add([Wrench::zero(); 2])))
                        .collect()
                };
                // Gather terms in the SAME order `bake_incidence` uses — connection
                // order, `a` before `b` — so the sum is associated identically in
                // every lane and the reduction stays deterministic.
                let mut terms: IndexMap<WorldId, Vec<GatherTerm<T>>> =
                    bodies.keys().map(|id| (*id, Vec::new())).collect();
                for (jid, e) in joints {
                    let k = conn[jid].clone();
                    terms[&e.a()].push(GatherTerm {
                        key: k.clone(),
                        slot: 0,
                    });
                    terms[&e.b()].push(GatherTerm { key: k, slot: 1 });
                }
                // For the FUSED stage, in `order` sequence — the unknowns only, and
                // in the same sequence `bake_body_post` walks. `bake_gathered` pairs
                // these with its body rows by position, and those rows come from
                // `order`, so anything else here silently hands a body its
                // neighbour's external wrench and its neighbour's incident
                // connections. `bodies` is NOT that sequence: it holds the island's
                // kinematic bodies too, and an island whose kinematic body is not
                // last (the camera rig's despun tracker sits second, because the
                // springs that merge it in are declared before the anchor's) shifts
                // every body after it by one.
                let ordered_terms: Vec<Vec<GatherTerm<T>>> =
                    self.order.keys().map(|id| terms[id].clone()).collect();
                let ordered_external: Vec<WorldKey<Wrench<T>>> =
                    self.order.keys().map(|id| external[id].clone()).collect();
                // Round-major across the island: round `k` of every body goes out
                // together, which is where a lane's batch breadth comes from.
                let mut gather: Vec<Vec<WorldKey<Row>>> = Vec::new();
                for id in bodies.keys() {
                    let baked = rows::gather::bake(world, &total[id], &external[id], &terms[id]);
                    for (round, key) in baked.into_iter().enumerate() {
                        if gather.len() == round {
                            gather.push(Vec::new());
                        }
                        gather[round].push(key);
                    }
                }
                let gather: Vec<Arc<[WorldKey<Row>]>> = gather.into_iter().map(Arc::from).collect();
                let m = self.order.len();
                let (rhs, scale, mass_block) = if lane == 0 {
                    (
                        self.rhs.clone(),
                        self.scale.clone(),
                        self.mass_block.clone(),
                    )
                } else {
                    let mut wr = world.write::<Wrench<T>>();
                    let rhs: Vec<_> = (0..m).map(|_| wr.add(Wrench::zero())).collect();
                    let scale: Vec<_> = (0..m).map(|_| wr.add(Wrench::zero())).collect();
                    drop(wr);
                    let mut bm = world.write::<Block<T>>();
                    let mass: Vec<_> = (0..m).map(|_| bm.add(zero_block::<T>())).collect();
                    (Arc::from(rhs), Arc::from(scale), Arc::from(mass))
                };
                // `Pre` and the connection kernels, baked against THIS lane's
                // slots. The families ride along so the dispatch knows which
                // kernel each row belongs to without touching the edges again.
                let pre_in: Vec<rows::fixed::PreRow<T>> = bodies
                    .keys()
                    .map(|id| rows::fixed::PreRow {
                        vel: vmid[id].clone(),
                        pose: bodies[id].body().pose.clone(),
                        midpoint: midpoint[id].clone(),
                        solve_vel: solve_vel[id].clone(),
                    })
                    .collect();
                let pre = rows::fixed::bake_pre(world, &pre_in, &retraction);
                let joint_in: Vec<rows::fixed::JointRow<T>> = joints
                    .iter()
                    .map(|(jid, e)| rows::fixed::JointRow {
                        vels: [solve_vel[&e.a()].clone(), solve_vel[&e.b()].clone()],
                        poses: [midpoint[&e.a()].clone(), midpoint[&e.b()].clone()],
                        conn: conn[jid].clone(),
                        block: e.jacobian_key(),
                        params: e.params_keys(),
                    })
                    .collect();
                let group = |jac: bool| -> Vec<(MessageKind, Arc<[WorldKey<Row>]>)> {
                    let baked = rows::fixed::bake_joints(world, &joint_in, &dt, &warp, jac);
                    let flat: Vec<WorldKey<Row>> = baked
                        .into_iter()
                        .next()
                        .map(|r| r.to_vec())
                        .unwrap_or_default();
                    let mut out: Vec<(MessageKind, Vec<WorldKey<Row>>)> = Vec::new();
                    for (e, row) in joints.values().zip(flat) {
                        let kind: MessageKind = (e.shader_name(), jac).into();
                        match out.iter_mut().find(|(k, _)| *k == kind) {
                            Some((_, v)) => v.push(row),
                            None => out.push((kind, vec![row])),
                        }
                    }
                    out.into_iter().map(|(k, v)| (k, Arc::from(v))).collect()
                };
                let plain = group(false);
                let jacobian = group(true);

                LaneSlots {
                    vmid,
                    midpoint,
                    solve_vel,
                    conn,
                    total,
                    gather,
                    gathered: None,
                    gather_terms: ordered_terms,
                    external: ordered_external,
                    pre,
                    plain,
                    jacobian,
                    body_post: Vec::new(),
                    rhs,
                    scale,
                    mass_block,
                }
            })
            .collect();

        // The phase-merged view. Built here rather than per dispatch: the merge is
        // a function of the baked rows, and doing it per step would clone every
        // key on every wave.
        let group = |jac: bool| -> MergedKernelRows {
            let mut acc: Vec<(MessageKind, Vec<WorldKey<Row>>)> = Vec::new();
            for lane in &self.lanes {
                let src = if jac { &lane.jacobian } else { &lane.plain };
                for (kind, rows) in src {
                    match acc.iter_mut().find(|(k, _)| k == kind) {
                        Some((_, v)) => v.extend(rows.iter().cloned()),
                        None => acc.push((*kind, rows.to_vec())),
                    }
                }
            }
            acc.into_iter()
                .map(|(k, v)| (k, vec![Arc::from(v)]))
                .collect()
        };
        self.wave = Wave {
            pre: merge_rounds(self.lanes.iter().map(|l| &l.pre)),
            plain: group(false),
            jacobian: group(true),
            gather: merge_rounds(self.lanes.iter().map(|l| &l.gather)),
            body_post: Vec::new(),
            gathered: None,
        };
    }

    /// Bake every lane's POST rows.
    ///
    /// Separate from `bake_lanes` only because it needs `bodies` AND the caller's
    /// `snap_mom` map. Every slot it names is topology-stable — including
    /// `snap_mom`, which IS the per-body `world_momentum` slot, rewritten in place
    /// rather than reallocated — so this is idempotent and guarded like the rest.
    /// It used to run per span, allocating and dropping `m` row keys each time for
    /// rows that came out identical.
    pub(crate) fn bake_body_post<S: Ring>(
        &mut self,
        world: &Arc<World>,
        bodies: &IndexMap<WorldId, Box<dyn crate::Component<T, S>>>,
        snap_mom: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
    ) {
        if !self.wave.body_post.is_empty() {
            return;
        }
        let half = self.half_slot.as_ref().unwrap().clone();
        let floor2 = self.floor2_slot.as_ref().unwrap().clone();
        let order: Vec<(WorldId, usize)> = self.order.iter().map(|(id, &i)| (*id, i)).collect();
        for lane in &mut self.lanes {
            let rows_in: Vec<rows::body_post::BodyRow<T>> = order
                .iter()
                .map(|(id, idx)| rows::body_post::BodyRow {
                    midpoint_pose: lane.midpoint[id].clone(),
                    solve_vel: lane.solve_vel[id].clone(),
                    inertia: bodies[id].body().inertia.keys(),
                    snap_mom: snap_mom[id].clone(),
                    total_wrench: lane.total[id].clone(),
                    mass_out: lane.mass_block[*idx].clone(),
                    rhs_out: lane.rhs[*idx].clone(),
                    scale_out: lane.scale[*idx].clone(),
                })
                .collect();
            lane.body_post = rows::body_post::bake(world, &rows_in, &half, &floor2);
            // The fused stage, when every body's term list fits one row. All or
            // nothing: a mixed dispatch would need both stages anyway, which is
            // the wave the fusion exists to remove.
            lane.gathered = rows::body_post::bake_gathered(
                world,
                &rows_in,
                &lane.external,
                &lane.gather_terms,
                &half,
                &floor2,
            );
        }
        self.body_post_rows = self.lanes[0].body_post.clone();
        self.wave.body_post = self
            .lanes
            .iter()
            .flat_map(|l| l.body_post.iter().cloned())
            .collect();
        // Merge the fused rows across lanes, by kernel — one message per variant
        // for the whole wave, exactly as the unfused stage does.
        self.wave.gathered = self
            .lanes
            .iter()
            .map(|l| l.gathered.as_ref())
            .collect::<Option<Vec<_>>>()
            .map(|per_lane| {
                let mut acc: Vec<(rows::body_post::Variant, Vec<WorldKey<Row>>)> = Vec::new();
                for lane in per_lane {
                    for (v, rows) in lane {
                        match acc.iter_mut().find(|(k, _)| k == v) {
                            Some((_, out)) => out.extend(rows.iter().cloned()),
                            None => acc.push((*v, rows.to_vec())),
                        }
                    }
                }
                acc.into_iter().map(|(v, r)| (v, Arc::from(r))).collect()
            });
    }

    /// Adopt the accepted lane as the iterate: copy the slots anything downstream
    /// reads into lane 0. Lane 0 is where `assemble` looks for the mass block,
    /// `block_matvec` for the residual and `commit_span` for the iterate, so this
    /// is what makes a probe's result the next iteration's starting point.
    ///
    /// `total` and `scale` are deliberately absent: the gather's output is only
    /// ever consumed by the POST that already ran, and the self-scale only by the
    /// norm already taken.
    pub(crate) fn adopt_lane(&mut self, lane: usize) {
        if lane == 0 {
            return;
        }
        let (head, tail) = self.lanes.split_at_mut(1);
        let (dst, src) = (&head[0], &tail[lane - 1]);
        for (id, k) in &dst.vmid {
            k.write(src.vmid[id].read());
        }
        for (id, k) in &dst.midpoint {
            k.write(src.midpoint[id].read());
        }
        for (id, k) in &dst.solve_vel {
            k.write(src.solve_vel[id].read());
        }
        for (d, s) in dst.rhs.iter().zip(src.rhs.iter()) {
            d.write(s.read());
        }
        for (d, s) in dst.mass_block.iter().zip(src.mass_block.iter()) {
            d.write(s.read());
        }
    }
}

impl<T: Scalar + Pod + PartialOrd> NewtonCache<T> {
    /// Settle the flags after `publish_rows` and the `‖X‖²` probe have run.
    ///
    /// The COPY is unconditional now, where the host loop skipped it on a
    /// non-finite `X`. That is not a behaviour change: `accepted_seeded` goes
    /// false with it, `finish_rollback` then leaves `seeded` false, and
    /// `run_budget` reseeds from `A` — overwriting whatever was published before
    /// anything reads it. Making it unconditional is what lets the copy and the
    /// probe ride the same wave instead of the probe gating the copy.
    ///
    /// The test moved from per-component to the Frobenius norm and got STRICTER
    /// with it: it still rejects any infinity or NaN (both lose the comparison),
    /// and now also rejects an `X` that is finite component-wise but enormous in
    /// aggregate. That is the direction worth erring in — such a hint is garbage
    /// whether or not each number is representable.
    pub(crate) fn finish_checkpoint(&mut self) {
        let huge = T::from_u32(1_000_000_000);
        let bound = huge * huge;
        let n = self.x_norm2();
        // Negated `<`, as everywhere in this solver: NaN loses every comparison,
        // and the direct form would let it through.
        if n < bound {
            self.accepted_seeded = self.seeded;
        } else {
            self.seeded = false;
            self.accepted_seeded = false;
        }
    }
}

#[cfg(test)]
impl<T: Scalar + Pod> NewtonCache<T> {
    /// `‖I − A·X‖²` — how far the cached approximate inverse is from
    /// the true one. A test probe for the "warm start stays warm" property;
    /// it does not take part in the step and computes the product itself, without touching `R`.
    pub(crate) fn inverse_residual(&self) -> T {
        let m = self.m();
        let mut acc = T::ZERO;
        for i in 0..m {
            for j in 0..m {
                let mut block = [[T::ZERO; 6]; 6];
                for k in 0..m {
                    let a = self.a[i * m + k].read();
                    let x = self.x[k * m + j].read();
                    for r in 0..6 {
                        for c in 0..6 {
                            let mut sum = block[r][c];
                            for t in 0..6 {
                                sum += a[r][t] * x[t][c];
                            }
                            block[r][c] = sum;
                        }
                    }
                }
                for (r, row) in block.iter().enumerate() {
                    for (c, &v) in row.iter().enumerate() {
                        let want = if i == j && r == c { T::ONE } else { T::ZERO };
                        let e = want - v;
                        acc += e * e;
                    }
                }
            }
        }
        acc
    }
}

impl<T: Scalar + Pod> std::fmt::Debug for NewtonCache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NewtonCache(m={}, seeded={})", self.m(), self.seeded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use joints::{AxialSpringDamper, SimpleSpringDamper};

    fn ids(n: usize) -> Vec<WorldId> {
        (0..n).map(|_| WorldId::get()).collect()
    }

    /// We build real joints: `sync` bakes both the sparsity and the contribution
    /// table from them, so a fake would test the wrong path.
    fn edges(
        world: &Arc<World>,
        pairs: &[(WorldId, WorldId)],
    ) -> IndexMap<WorldId, JointEdge<f32>> {
        pairs
            .iter()
            .map(|&(a, b)| {
                let j = AxialSpringDamper::builder(world.clone(), SimpleSpringDamper)
                    .rest(1.0)
                    .stiffness(1.0)
                    .damping(0.0)
                    .build();
                (WorldId::get(), JointEdge::new(a, b, j))
            })
            .collect()
    }

    #[test]
    fn sync_allocates_on_first_call_and_records_sparsity() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mut cache = NewtonCache::<f32>::new();
        let b = ids(3);

        cache.sync(
            &world,
            b.iter().copied(),
            &edges(&world, &[(b[0], b[1]), (b[1], b[2])]),
        );

        assert_eq!(cache.m(), 3);
        assert_eq!(cache.a().len(), 9);
        assert_eq!(cache.x().len(), 9);
        assert_eq!(cache.rhs().len(), 3);
        assert_eq!(cache.dv().len(), 3);
        assert!(!cache.is_seeded());

        // Chain 0-1-2: body 1 touches everything, bodies 0 and 2 — two each. We check
        // `sparse_rows` — that is what `A·X` reads, not the internal field.
        let mut n1 = cache.sparse_rows()[1].to_vec();
        n1.sort();
        assert_eq!(n1, vec![0, 1, 2]);
        let mut n0 = cache.sparse_rows()[0].to_vec();
        n0.sort();
        assert_eq!(n0, vec![0, 1]);
    }

    #[test]
    fn resync_at_the_same_dimension_keeps_the_warm_inverse() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mut cache = NewtonCache::<f32>::new();
        let b = ids(2);

        cache.sync(&world, b.iter().copied(), &edges(&world, &[(b[0], b[1])]));
        cache.mark_seeded();
        let before = cache.x()[0].clone();
        before.write([[7.0; 6]; 6]);

        cache.sync(&world, b.iter().copied(), &edges(&world, &[(b[0], b[1])]));

        assert!(cache.is_seeded(), "same dimension must not drop the hint");
        assert_eq!(cache.x()[0].read()[0][0], 7.0);
    }

    #[test]
    #[ignore]
    fn dimension_change_reallocates_and_clears_the_hint() {
        // FIXME: deadlock?
        let world = Arc::new(World::builder().usual::<f32>());
        let mut cache = NewtonCache::<f32>::new();
        let b = ids(3);

        cache.sync(&world, b.iter().copied(), &edges(&world, &[(b[0], b[1])]));
        cache.mark_seeded();
        cache.sync(
            &world,
            b[..2].iter().copied(),
            &edges(&world, &[(b[0], b[1])]),
        );

        assert_eq!(cache.m(), 2);
        assert_eq!(cache.a().len(), 4);
        assert!(!cache.is_seeded(), "dimension change must clear the hint");
    }
}
