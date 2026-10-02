// SPDX-License-Identifier: MIT

use crate::Component;
use crate::EvalError;
use crate::GatherTerm;
use crate::integrator::implicit::cache::{LaneSlots, Wave};
use aristotle::{World, WorldId, WorldKey};
use bytemuck::Pod;
use clifford::Lift;
use clifford::pga3::Wrench;
use futures::channel::oneshot;
use futures::future::join_all;
use indexmap::IndexMap;
use joints::JointEdge;
use peano::prelude::*;
use rembrandt::GpuAccelerator;
use shaders::ShaderSetup;
use std::any::{TypeId, type_name};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use tokio::sync::Semaphore;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator,
    },
    pipeline::PipelineBindPoint,
    sync::{self, GpuFuture},
};

pub(crate) mod row;
pub(crate) mod rows;
pub(crate) mod shaders;

pub(crate) struct MessageInput {
    kind: MessageKind,
    payload: MessagePayload,
}

/// A message waiting on the worker, paired with the channel its outcome is
/// reported on. `None` for a fire-and-forget submission whose caller is not
/// waiting.
pub(crate) type Pending = (MessageInput, Option<oneshot::Sender<Result<(), EvalError>>>);

/// Pending messages grouped by the kernel that will run them: one batch per
/// [`MessageKind`], every kind registered up front so the push site is a plain
/// lookup.
pub(crate) type PendingByKind = HashMap<MessageKind, Vec<Pending>>;

impl MessageInput {
    /// Batch-table entries this message occupies. One for the fixed-arity kernels;
    /// a whole round for the row-driven ones.
    fn invocations(&self) -> usize {
        let MessagePayload::Rows { rows } = &self.payload;
        rows.len()
    }
}

#[derive(Hash, Debug, Eq, PartialEq, Clone, Copy)]
pub(crate) enum MessageKind {
    EvalSimpleSum,
    EvalCriticallyDampedWarpedPlain,
    EvalCriticallyDampedWarpedJacobian,
    EvalPerpendicularDamperWarpedPlain,
    EvalPerpendicularDamperWarpedJacobian,
    EvalSimpleSpringDamperPlain,
    EvalSimpleSpringDamperJacobian,
    EvalTorsionalDamperWarpedPlain,
    EvalTorsionalDamperWarpedJacobian,
    Pre,
    Gather,
    Gemm,
    /// POST splits by angular-inertia storage: a diagonal tensor and a full one
    /// bind different buffers, hence different kernels.
    BodyPostDiagonal,
    BodyPostFull,
    AssembleBlock,
    BlockMatVec,
    BlockReduce,
    BlockCopy,
    BodyPostGatheredDiagonal,
    BodyPostGatheredFull,
}

impl MessageKind {
    const fn all() -> &'static [Self] {
        &[
            MessageKind::EvalSimpleSum,
            MessageKind::EvalCriticallyDampedWarpedPlain,
            MessageKind::EvalCriticallyDampedWarpedJacobian,
            MessageKind::EvalPerpendicularDamperWarpedPlain,
            MessageKind::EvalPerpendicularDamperWarpedJacobian,
            MessageKind::EvalSimpleSpringDamperPlain,
            MessageKind::EvalSimpleSpringDamperJacobian,
            MessageKind::EvalTorsionalDamperWarpedPlain,
            MessageKind::EvalTorsionalDamperWarpedJacobian,
            MessageKind::Pre,
            MessageKind::Gather,
            MessageKind::Gemm,
            MessageKind::BodyPostDiagonal,
            MessageKind::BodyPostFull,
            MessageKind::AssembleBlock,
            MessageKind::BlockMatVec,
            MessageKind::BlockReduce,
            MessageKind::BlockCopy,
            MessageKind::BodyPostGatheredDiagonal,
            MessageKind::BodyPostGatheredFull,
        ]
    }
}

impl From<(&'static str, bool)> for MessageKind {
    fn from(s: (&'static str, bool)) -> Self {
        match s {
            ("CriticallyDampedWarped", false) => Self::EvalCriticallyDampedWarpedPlain,
            ("CriticallyDampedWarped", true) => Self::EvalCriticallyDampedWarpedJacobian,
            ("PerpendicularDamperWarped", false) => Self::EvalPerpendicularDamperWarpedPlain,
            ("PerpendicularDamperWarped", true) => Self::EvalPerpendicularDamperWarpedJacobian,
            ("SimpleSpringDamper", false) => Self::EvalSimpleSpringDamperPlain,
            ("SimpleSpringDamper", true) => Self::EvalSimpleSpringDamperJacobian,
            ("TorsionalDamperWarped", false) => Self::EvalTorsionalDamperWarpedPlain,
            ("TorsionalDamperWarped", true) => Self::EvalTorsionalDamperWarpedJacobian,
            _ => unreachable!(),
        }
    }
}

enum MessagePayload {
    /// A whole ROUND of baked rows. Everything a stage needs to know is inside the
    /// rows themselves; the keys are here to keep those slots alive from enqueue to
    /// flush (a clone is one atomic). The index arithmetic was done at bake time,
    /// so `setup` writes one number per row — its slot (accelerator/row.rs).
    ///
    /// The round, not the row, is the unit deliberately. A reduction round is
    /// `m²` rows for the block stages, and paying a message, a oneshot and a
    /// channel round-trip for each of them is what made the step grow as `m²`:
    /// at 45 unknowns 86% of all messages were block cells. One message per round
    /// also makes a round unsplittable across flushes, which is what used to tear
    /// them apart. Rounds from DIFFERENT mechanisms remain independent messages
    /// and still coalesce into one dispatch — see the `@row` arm in `shaders.rs`.
    Rows { rows: Arc<[WorldKey<row::Row>]> },
}

enum Message {
    Job {
        input: MessageInput,
        respond_to: oneshot::Sender<Result<(), EvalError>>,
    },
    Shutdown,
}

/// Bake the per-body incidence cache (`RigidBody::incident_terms`) from the joints —
/// the input of gather phase 3. For each body, the list of joint wrench slots
/// flowing into its `total_wrench` is collected, in `joints` traversal order (sum determinism).
/// O(joints), idempotent.
///
/// CONTRACT: called when the SET of joints/bodies changes. In production this is done by
/// the topology layer (`Islands` on connect/detach/split/merge). A direct caller of
/// `Accelerator::dispatch` bypassing `Mechanism` (integrator unit tests) must
/// call this ITSELF after assembling `bodies`+`joints` — otherwise the gather will sum an empty
/// list and lose the joint forces.
pub(crate) fn bake_incidence<T, S>(
    bodies: &mut IndexMap<WorldId, Box<dyn Component<T, S>>>,
    joints: &IndexMap<WorldId, JointEdge<T>>,
    external: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
) where
    T: Scalar + StandardPart + Pod,
    S: Scalar + From<T> + Into<T>,
{
    // Phase 1 (immutable): collect contributions per body, traversing the joints in order.
    let mut acc: IndexMap<WorldId, Vec<GatherTerm<T>>> =
        bodies.keys().map(|id| (*id, Vec::new())).collect();
    for e in joints.values() {
        // Both ends of a joint are bodies from `bodies` (island invariant), the unwrap is safe.
        let k = e.wrench_key();
        acc.get_mut(&e.a()).unwrap().push(GatherTerm {
            key: k.clone(),
            slot: 0,
        });
        acc.get_mut(&e.b())
            .unwrap()
            .push(GatherTerm { key: k, slot: 1 });
    }
    // Phase 2 (mutable): write the finished slices into the bodies — the borrows do not overlap.
    for (id, terms) in acc {
        let b = bodies.get_mut(&id).unwrap().body_mut();
        b.incident_terms = Arc::from(terms);
    }
    // Phase 3: incidence rows for the GPU — with the same trigger and in the same order
    // as the terms themselves, otherwise they could describe different topologies. A row holds
    // ONLY slot indices, so a step does not rewrite it (accelerator/row.rs).
    for (id, e) in bodies.iter_mut() {
        let b = e.body_mut();
        // The World comes from the body itself: the island does not hold it, and the slots the row
        // addresses live in it anyway.
        let world = b.world.clone();
        b.gather_rows = Arc::from(rows::gather::bake(
            &world,
            &b.total_wrench,
            &external[id],
            &b.incident_terms,
        ));
    }
}

struct Shaders {
    shaders: HashMap<MessageKind, Box<dyn ShaderSetup>>,
    /// The one-writer-per-slot ledger every shader of `shaders` claims into.
    ledger: Arc<shaders::Ledger>,
    cmd_allocator: Arc<StandardCommandBufferAllocator>,
    gpu: Arc<GpuAccelerator>,
    world: Arc<World>,
    #[cfg(test)]
    fault: Option<Arc<FaultInjector>>,
}

impl Shaders {
    fn backend(e: impl std::fmt::Display) -> EvalError {
        EvalError::Backend {
            shader: "unspecified",
            msg: e.to_string(),
        }
    }

    /// Times a flush is re-recorded because a storage it binds grew between
    /// recording and taking the guards, before it gives up with `Backend`.
    /// Growth is a rare structural event, so one retry nearly always suffices.
    const MAX_RERECORD: usize = 8;

    /// Record every non-empty batch into one command buffer and submit it.
    ///
    /// Order matters twice:
    /// 1. The one-writer check is a PRE-PASS over every kind of the flush, so a
    ///    collision refuses the whole flush before any kind is recorded.
    /// 2. `setup` binds the CURRENT buffer of each storage, taking read locks
    ///    on them, so it has to run before the write guards are taken — the
    ///    locks are not reentrant. A storage can therefore grow between
    ///    recording and guarding. Once the guards are held growth is excluded,
    ///    so the bindings are re-checked there, and a stale recording is thrown
    ///    away and recorded again instead of being submitted against an old
    ///    buffer.
    fn dispatch(&self, gpu_batches: &mut PendingByKind) -> Result<(), EvalError> {
        self.ledger.begin();
        for (kind, batch) in gpu_batches.iter() {
            if batch.is_empty() {
                continue;
            }
            self.shaders[kind]
                .claim(batch)
                .map_err(|msg| EvalError::Backend {
                    shader: "one-writer check",
                    msg,
                })?;
        }

        let queue = self.gpu.queue();
        let mut storages = HashSet::new();
        for (kind, batch) in gpu_batches.iter() {
            if !batch.is_empty() {
                storages.extend(self.shaders[kind].storages().iter().copied());
            }
        }

        // Declared BEFORE the guards so it outlives them: retiring a message drops
        // its `WorldKey`s, and the last handle to one takes `World::write` on its
        // storage — a lock the guards below are holding. Collecting here and
        // letting the vector die at the end of the call keeps that off the guarded
        // region no matter how the scopes below are later rearranged.
        let mut retired: Vec<MessageInput> = Vec::new();

        let mut attempt = 0;
        loop {
            let mut builder = AutoCommandBufferBuilder::primary(
                self.cmd_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(Self::backend)?;

            for (kind, batch) in gpu_batches.iter() {
                if batch.is_empty() {
                    continue;
                }
                // `setup` reports the INVOCATIONS the batch filled, which is not its
                // message count: a row-driven message carries a whole round.
                let count = self.shaders[kind].setup(&mut builder, batch)?;
                if count == 0 {
                    continue;
                }
                // SAFETY: one invocation per index-table row, rounded up to whole
                // workgroups; the kernel bounds-checks each against `count`, and every
                // row within `count` addresses a live slot whose key the batch holds.
                unsafe {
                    builder
                        .dispatch([count.div_ceil(GpuAccelerator::LOCAL_SIZE_X), 1, 1])
                        .map_err(Self::backend)?;
                }
            }

            #[cfg(test)]
            if attempt == 0
                && let Some(f) = &self.fault
            {
                f.trip(gpu_batches)?;
            }

            let command_buffer = builder.build().map_err(Self::backend)?;

            let _guards: Vec<_> = storages
                .iter()
                .map(|t| self.world.write_guard(*t))
                .collect();
            let grown = gpu_batches
                .iter()
                .filter(|(_, batch)| !batch.is_empty())
                .find_map(|(kind, _)| self.shaders[kind].stale());
            if let Some(t) = grown {
                attempt += 1;
                if attempt > Self::MAX_RERECORD {
                    return Err(Self::backend(format!(
                        "world storage {t:?} kept growing while the flush was recorded"
                    )));
                }
                continue;
            }
            sync::now(self.gpu.device().clone())
                .then_execute(queue.clone(), command_buffer)
                .map_err(Self::backend)?
                .then_signal_fence_and_flush()
                .map_err(Self::backend)?
                .wait(None)
                .map_err(Self::backend)?;
            break;
        }

        for (kind, batch) in gpu_batches.iter_mut() {
            if batch.is_empty() {
                continue;
            }
            // Sequential notify, off the compute path: one submission is
            // atomic, so its outcome is the outcome of every job in it.
            retired.extend(self.shaders[kind].check(batch));
        }

        Ok(())
    }

    /// Run one flush and report its outcome to every job it carried.
    ///
    /// On success `dispatch` has already notified each job through `check`. On
    /// failure it returns before `check` with every batch still intact, so the
    /// jobs are answered here: one submission is atomic, its error is the outcome
    /// of every job in it, and the worker survives to serve the next batch. The
    /// batches are left empty either way, which is what keeps the caller's
    /// occupancy counters exact after a reset.
    fn flush(&self, gpu_batches: &mut PendingByKind) {
        let Err(e) = self.dispatch(gpu_batches) else {
            return;
        };
        // Retired inputs die after the loop, with no world guard held — the same
        // reason `dispatch` collects them instead of dropping them in place.
        let mut retired: Vec<MessageInput> = Vec::new();
        for batch in gpu_batches.values_mut() {
            for (input, respond_to) in batch.drain(..) {
                if let Some(respond_to) = respond_to {
                    let _ = respond_to.send(Err(e.clone()));
                }
                retired.push(input);
            }
        }
    }
}

/// Test seam: fail chosen flushes AFTER their command buffer is recorded and
/// before it is submitted — the point where a real `build` or submit error
/// leaves the batch fully set up but never run.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct FaultInjector {
    /// Flushes still to fail, counted down one per injected failure.
    fail: std::sync::atomic::AtomicUsize,
    /// Messages each failed flush carried, in order.
    failed_jobs: std::sync::Mutex<Vec<usize>>,
}

#[cfg(test)]
impl FaultInjector {
    fn trip(&self, gpu_batches: &PendingByKind) -> Result<(), EvalError> {
        use std::sync::atomic::Ordering;
        let armed = self
            .fail
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok();
        if !armed {
            return Ok(());
        }
        let jobs = gpu_batches.values().map(Vec::len).sum();
        self.failed_jobs.lock().unwrap().push(jobs);
        Err(EvalError::Backend {
            shader: "injected",
            msg: "injected submission failure".into(),
        })
    }
}

// The `Accelerator` + async-dispatch design — the executor model (why the whole
// integrator goes async, one accelerator fed index records, heuristic/quiescence
// flush, await-per-block), result write-back by partitioned ownership, the
// structural-vs-data lock split, and the `GpuVec<T>` shared-memory storage — all
// lives in `/ACCELERATOR.md` (Parts II–IV). Kept out of source comments so the
// design is not lost.
//
// Every kernel runs on the GPU from a worker thread that owns the compiled
// pipelines and the batch tables; the handle below holds only the channel, the
// submission bound and the thread.
pub struct Accelerator<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> {
    tx: Sender<Message>,
    thread: Option<JoinHandle<()>>,
    /// Rows the batch table holds — every shader's `incidence`/`fatals` buffers
    /// were allocated at this size, so it bounds one message AND one flush.
    capacity: usize,
    /// Back-pressure: one permit per submitted message, held until its outcome
    /// arrives. A producer that finds none left WAITS (asynchronously) instead
    /// of growing the queue without limit.
    permits: Semaphore,
    /// The message path is fully type-erased — a payload is slot INDICES, and the
    /// scalar type only ever appears in this handle's own signatures.
    _marker: PhantomData<T>,
}

/// Configures an [`Accelerator`]. Every knob changes only TIMING — how messages
/// pack into flushes and how far producers may run ahead — never the result of
/// any kernel (ACCELERATOR.md Part V, the determinism invariant).
pub struct AcceleratorBuilder<T> {
    world: Arc<World>,
    batch_size: Option<usize>,
    max_pending: usize,
    log_flushes: bool,
    #[cfg(test)]
    fault: Option<Arc<FaultInjector>>,
    _marker: PhantomData<T>,
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> Accelerator<T> {
    pub fn builder(world: Arc<World>) -> AcceleratorBuilder<T> {
        AcceleratorBuilder {
            world,
            batch_size: None,
            max_pending: AcceleratorBuilder::<T>::DEFAULT_MAX_PENDING,
            log_flushes: false,
            #[cfg(test)]
            fault: None,
            _marker: PhantomData,
        }
    }

    /// Dispatch one connected island's connection kernels — the accelerator-owned
    /// PRE + connection-kernel + gather stage, shared by ALL integrators. The
    /// only per-family inputs are `vel` (the solve velocity per body) and
    /// `retraction` (the exp-time folded into the midpoint pose):
    /// - implicit (`Newton`): `vel` = current `vmid` iterate, `retraction` = ½dt,
    ///   `jacobian` = true (needs the block);
    /// - explicit (Euler/Symplectic/Lie): `vel` = world velocity, `retraction` = 0
    ///   (so `midpoint = pose`), `jacobian` = false.
    ///
    /// Returns `Err` if a connection kernel fails (Lua trap / fired fatal). A
    /// phase-2 error short-circuits before the gather (phase 3).
    /// Dispatch one island's PRE + connection + gather stages against ONE lane's
    /// slots. Everything it needs is baked: three rounds of rows, one message
    /// each, and the per-step scalars sit in slots the rows name.
    ///
    /// `jacobian` picks the connection kernel — value-only for a line-search
    /// probe or an explicit step, value plus derivative for a Newton iterate.
    pub(crate) async fn dispatch(
        &self,
        lane: &LaneSlots<T>,
        jacobian: bool,
        gather: bool,
    ) -> Result<(), EvalError> {
        // Phase 1 — PRE (per body, independent): midpoint pose + solve velocity.
        self.submit_rounds(MessageKind::Pre, &lane.pre).await?;
        // Phase 2 — connection kernels. One message per joint FAMILY, because a
        // family is a different pipeline; within a family the rows are one round.
        self.eval_joints(lane, jacobian).await?;
        // Phase 3 — GATHER (per body): `total_wrench = external + Sum` incident
        // connection wrenches. Outputs are disjoint; the deterministic reduction
        // order lives in the term order inside the row (ACCELERATOR.md §6).
        //
        // Skipped when the caller follows with the FUSED post, which sums the same
        // terms inside its own invocation and never materialises `total_wrench`.
        if !gather {
            return Ok(());
        }
        self.submit_rounds(MessageKind::Gather, &lane.gather).await
    }

    /// The same three phases as `dispatch`, but for EVERY line-search lane at
    /// once: one message per phase for the whole wave instead of one per lane.
    ///
    /// Driving the lanes as concurrent `dispatch` calls left them free to wake in
    /// any order, so the worker flushed partial waves — measured as a phase
    /// arriving in two or three pieces and the wave costing eight or nine
    /// submissions where four were due. The rows were already merged at bake time
    /// (`Wave`); this just sends them.
    pub(crate) async fn dispatch_wave(
        &self,
        wave: &Wave,
        jacobian: bool,
        gather: bool,
    ) -> Result<(), EvalError> {
        self.submit_rounds(MessageKind::Pre, &wave.pre).await?;
        let groups = if jacobian {
            &wave.jacobian
        } else {
            &wave.plain
        };
        let futs = groups
            .iter()
            .map(|(kind, rounds)| self.submit_rounds(*kind, rounds));
        for r in join_all(futs).await {
            r?;
        }
        if !gather {
            return Ok(());
        }
        self.submit_rounds(MessageKind::Gather, &wave.gather).await
    }

    /// One connection kernel    /// Phase 2 alone: every connection's kernel, grouped by family.
    async fn eval_joints(&self, lane: &LaneSlots<T>, jacobian: bool) -> Result<(), EvalError> {
        let groups = if jacobian {
            &lane.jacobian
        } else {
            &lane.plain
        };
        let futs = groups
            .iter()
            .map(|(kind, rows)| self.submit_rounds(*kind, std::slice::from_ref(rows)));
        for r in join_all(futs).await {
            r?;
        }
        Ok(())
    }

    /// Phase 2 ALONE    /// Phase 2 ALONE, Jacobian variant: re-evaluate every connection at the
    /// midpoint pose and solve velocity ALREADY in the slots.
    ///
    /// This is the Newton iteration's re-entry after an accepted line-search
    /// probe. The probe ran the full `dispatch` at the very iterate the next
    /// iteration is about to work from, so PRE's inputs (`pose`, `vmid`) have not
    /// moved and the gather's output (`total_wrench`) is the one `BodyPost`
    /// already consumed — re-running either would recompute its own result. The
    /// only thing no probe ever writes is the Jacobian block, because the probe
    /// needs a value, not a derivative. So the iteration costs ONE dispatch wave
    /// instead of four.
    ///
    /// The kernel also rewrites the per-connection value pair `conn`, which is
    /// then not re-gathered: `total_wrench` keeps what the probe's value kernel
    /// summed. That is deliberate — the residual, its scale and the mass block in
    /// the slots all came from that same pass, so the set stays self-consistent.
    pub(crate) async fn eval_jacobians(&self, lane: &LaneSlots<T>) -> Result<(), EvalError> {
        self.eval_joints(lane, true).await
    }

    /// Transport test: `out = a + b` over the flat scalar map. Bakes a one-off
    /// row (0 = a, 1 = b, 2 = out) inline — the one place a row is built at
    /// dispatch time, which is fine because the test allocates its slots inline
    /// anyway.
    pub async fn simple_sum(
        &self,
        a: &WorldKey<T>,
        b: &WorldKey<T>,
        out: &WorldKey<T>,
    ) -> Result<(), EvalError> {
        let world = a.world();
        let mut r: row::Row = [0; row::ROW];
        r[0] = a.raw_index() as u32;
        r[1] = b.raw_index() as u32;
        r[2] = out.raw_index() as u32;
        let row = world.write::<row::Row>().add(r);
        self.submit(MessageInput {
            kind: MessageKind::EvalSimpleSum,
            payload: MessagePayload::Rows {
                rows: Arc::from(vec![row]),
            },
        })
        .await
    }

    /// One block matrix product from pre-baked ROUNDS: `baked[k]` is round `k` of
    /// every output cell, and the rounds go out in order.
    ///
    /// The sparsity of the left factor, the `two_minus` fold and the `k == j`
    /// diagonal are all resolved at bake time — see `rows::gemm`.
    pub(crate) async fn gemm(&self, baked: &[Arc<[WorldKey<row::Row>]>]) -> Result<(), EvalError> {
        self.submit_rounds(MessageKind::Gemm, baked).await
    }

    /// POST — the per-body mass block, residual and self-scale, from pre-baked
    /// rows. There is no reduction, so every row is independent; the two inertia
    /// variants go to different kernels.
    ///
    /// Appended after `dispatch` rather than folded into it, so the explicit
    /// schemes — which have no residual — do not pay for it. It reads the
    /// gather's output, so it must follow phase 3; the barrier is inherent to the
    /// data dependency, not an artefact of splitting the call.
    pub(crate) async fn body_post(
        &self,
        baked: &[(rows::body_post::Variant, WorldKey<row::Row>)],
    ) -> Result<(), EvalError> {
        // The two inertia variants bind different storages, so they are different
        // kernels — but within a variant the rows are one round.
        let futs = [
            (
                rows::body_post::Variant::Diagonal,
                MessageKind::BodyPostDiagonal,
            ),
            (rows::body_post::Variant::Full, MessageKind::BodyPostFull),
        ]
        .map(|(want, kind)| {
            let rows: Vec<_> = baked
                .iter()
                .filter(|(v, _)| *v == want)
                .map(|(_, row)| row.clone())
                .collect();
            let rows: Arc<[WorldKey<row::Row>]> = Arc::from(rows);
            async move { self.submit_rounds(kind, std::slice::from_ref(&rows)).await }
        });
        for r in join_all(futs).await {
            r?;
        }
        Ok(())
    }

    /// Assemble the whole system matrix from pre-baked rounds.
    pub(crate) async fn assemble(
        &self,
        baked: &[Arc<[WorldKey<row::Row>]>],
    ) -> Result<(), EvalError> {
        self.submit_rounds(MessageKind::AssembleBlock, baked).await
    }

    /// GATHER for ONE body's baked rows, one round at a time — the unit tests
    /// drive a single body's reduction directly.
    #[cfg(test)]
    pub(crate) async fn gather_rows(&self, rows: &[WorldKey<row::Row>]) -> Result<(), EvalError> {
        let rounds: Vec<Arc<[WorldKey<row::Row>]>> =
            rows.iter().map(|r| Arc::from(vec![r.clone()])).collect();
        self.submit_rounds(MessageKind::Gather, &rounds).await
    }

    /// `Σ ‖diag·I − b‖²` over the blocks the rounds name, one partial per row.
    /// The host adds the partials up — see `rows::reduce`.
    pub(crate) async fn block_reduce(
        &self,
        baked: &[Arc<[WorldKey<row::Row>]>],
    ) -> Result<(), EvalError> {
        self.submit_rounds(MessageKind::BlockReduce, baked).await
    }

    /// POST with the gather folded in — one message per inertia variant, the same
    /// shape as `body_post`, but the rows carry each body's incident terms and the
    /// kernel sums them itself.
    pub(crate) async fn body_post_gathered(
        &self,
        baked: &[(rows::body_post::Variant, Arc<[WorldKey<row::Row>]>)],
    ) -> Result<(), EvalError> {
        let futs = baked.iter().map(|(v, rows)| {
            let kind = match v {
                rows::body_post::Variant::Diagonal => MessageKind::BodyPostGatheredDiagonal,
                rows::body_post::Variant::Full => MessageKind::BodyPostGatheredFull,
            };
            self.submit_rounds(kind, std::slice::from_ref(rows))
        });
        async move {
            for r in join_all(futs).await {
                r?;
            }
            Ok(())
        }
        .await
    }

    /// `dst = src` over the blocks the rounds name.
    pub(crate) async fn block_copy(
        &self,
        baked: &[Arc<[WorldKey<row::Row>]>],
    ) -> Result<(), EvalError> {
        self.submit_rounds(MessageKind::BlockCopy, baked).await
    }

    /// `dv = x · rhs`, from pre-baked rounds.
    pub(crate) async fn block_matvec(
        &self,
        baked: &[Arc<[WorldKey<row::Row>]>],
    ) -> Result<(), EvalError> {
        self.submit_rounds(MessageKind::BlockMatVec, baked).await
    }

    /// Dispatch pre-baked rows in ROUNDS: every row of round `k` goes out
    /// together and is awaited before round `k + 1` starts.
    ///
    /// The rounds exist because a reduction longer than one row is split across
    /// several, and each seeds from the previous one's output — so they must not
    /// overlap. Within a round the rows are independent (disjoint outputs), and
    /// there are as many of them as there are bodies or cells, which is where the
    /// batch breadth comes from.
    ///
    /// The rounds arrive pre-grouped from the bake, not regrouped here. Building
    /// them per call meant cloning every `WorldKey` — an `Arc` bump and a drop per
    /// cell, `m²` of them, on each of the five products an iteration runs — to
    /// re-derive a grouping that only changes when the topology does.
    ///
    /// NOT DONE — the round should be the unit of the MESSAGE, not just of the
    /// await. One message and one oneshot per ROW is why a round gets torn apart:
    /// the worker flushes on the first moment the channel looks empty, so it races
    /// the producer mid-round. Measured on `cloth_survives_frame_time_spikes`,
    /// 31% of all flushes carried exactly ONE job against a mean of 9.7, while
    /// `gpu_simple_sum` — which submits 6000 jobs with no await in between — packs
    /// them into 3 flushes. A submission costs ~52 µs regardless of how many jobs
    /// ride it, so the tearing is close to pure loss. Carrying `Vec<WorldKey<Row>>`
    /// in one `MessagePayload` would make a round unsplittable by construction and
    /// drop ~876 oneshots per step to ~51; the fiddly part is `ShaderSetup::check`,
    /// which indexes `fatals` per batch entry and would need a per-message base.
    async fn submit_rounds(
        &self,
        kind: MessageKind,
        rounds: &[Arc<[WorldKey<row::Row>]>],
    ) -> Result<(), EvalError> {
        for round in rounds {
            if round.is_empty() {
                continue;
            }
            // The round travels as a SHARED handle, not a copy. It is owned by the
            // cache for the whole step and only has to stay alive until the flush,
            // so cloning every key into the message — 21800 clone/drop pairs per
            // Newton iteration at 66 unknowns — bought nothing but atomics.
            //
            // A round wider than the batch table still has to be split, and that
            // path does copy; it is unreachable in practice (the table holds at
            // least 64Ki rows against `m²` cells) and exists so the contract does
            // not depend on that.
            if round.len() <= self.capacity {
                self.submit(MessageInput {
                    kind,
                    payload: MessagePayload::Rows {
                        rows: round.clone(),
                    },
                })
                .await?;
            } else {
                let futs = round.chunks(self.capacity).map(|chunk| {
                    self.submit(MessageInput {
                        kind,
                        payload: MessagePayload::Rows {
                            rows: Arc::from(chunk.to_vec()),
                        },
                    })
                });
                for r in join_all(futs).await {
                    r?;
                }
            }
        }
        Ok(())
    }

    /// Send one job to the worker and await its result. A dead channel (worker
    /// gone) is a recoverable `EvalError::WorkerGone`, not a panic; the kernel's
    /// own failure comes back as the inner `Err(EvalError)`.
    ///
    /// Waits for a submission permit first (`AcceleratorBuilder::max_pending`)
    /// and holds it until the outcome arrives. That cannot deadlock: a held
    /// permit only waits on the worker, and the worker flushes whenever its
    /// channel runs dry, which it does once every permit is taken.
    async fn submit(&self, input: MessageInput) -> Result<(), EvalError> {
        // The semaphore is never closed, so `acquire` cannot fail; map it anyway
        // rather than unwrap on the producer's side.
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| EvalError::WorkerGone)?;
        let (respond_to, rx) = oneshot::channel();
        self.tx
            .send(Message::Job { input, respond_to })
            .map_err(|_| EvalError::WorkerGone)?;
        rx.await.map_err(|_| EvalError::WorkerGone)?
    }
}

impl<T> AcceleratorBuilder<T> {
    /// Default for [`Self::max_pending`]. Far above what one step of the
    /// largest test scene keeps in flight, so it only bites on a runaway
    /// producer.
    pub const DEFAULT_MAX_PENDING: usize = 1 << 16;

    /// Flush threshold, in batch-table ROWS per kernel: a flush goes out as soon
    /// as one kernel's queued rows reach it, or when the submission channel runs
    /// dry, whichever comes first. Defaults to — and is capped at — the batch
    /// table's capacity (`GpuAccelerator::gpu_in_flight`); `0` is taken as `1`.
    ///
    /// A hint, not a split: a message is one whole round and is never torn
    /// across flushes, so a round wider than the hint simply flushes alone.
    pub fn batch_size(mut self, rows: usize) -> Self {
        self.batch_size = Some(rows.max(1));
        self
    }

    /// Back-pressure bound: at most this many messages submitted and not yet
    /// answered. A producer past it waits for an outcome to free a slot. `0`
    /// is taken as `1`. Defaults to [`Self::DEFAULT_MAX_PENDING`].
    pub fn max_pending(mut self, messages: usize) -> Self {
        self.max_pending = messages.max(1);
        self
    }

    /// Print one line per flush to stderr — batch count, rows, the fullest
    /// kernel. Off by default.
    pub fn log_flushes(mut self, on: bool) -> Self {
        self.log_flushes = on;
        self
    }

    #[cfg(test)]
    fn inject_faults(mut self, fault: Arc<FaultInjector>) -> Self {
        self.fault = Some(fault);
        self
    }
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod + Sync + Send + 'static>
    AcceleratorBuilder<T>
{
    pub fn build(self) -> Accelerator<T> {
        // The kernel declares `buf.data[]` as `float[]`, so binding 0 — the
        // world's flat scalar map — is only meaningful for f32. Same shape
        // of guard as the Lua backend's `t_to_f64`. Real typing under `T`
        // is the future viete -> SPIR-V codegen's job.
        assert!(
            TypeId::of::<T>() == TypeId::of::<f32>(),
            "gpu backend: only f32, got {}",
            type_name::<T>()
        );
        let w = &self.world;
        let g = w.gpu();
        let capacity = g.gpu_in_flight();
        let gpu_batch_size = self.batch_size.unwrap_or(capacity).min(capacity);
        let log_flushes = self.log_flushes;

        let ledger = Arc::new(shaders::Ledger::default());
        let shaders = Shaders {
            shaders: shaders::all_shaders(w, &ledger),
            ledger,
            cmd_allocator: g.cmd_allocator().clone(),
            gpu: g,
            world: w.clone(),
            #[cfg(test)]
            fault: self.fault,
        };

        // Unbounded on purpose: the bound is the `permits` semaphore, which a
        // producer awaits instead of blocking its executor thread on a full
        // `sync_channel`.
        let (tx, rx) = mpsc::channel::<Message>();
        let thread = thread::spawn(move || {
            // Register every message kind's batch up front, so the push site is a
            // plain lookup. A message whose kind is missing here is a wiring bug (a
            // new `MessageKind` left unregistered) and panics rather than silently
            // spawning a stray batch. This list is the single source of GPU kinds
            // and must mirror `Shaders::dispatch`.
            let mut gpu_batches: PendingByKind = MessageKind::all()
                .iter()
                .map(|k| (*k, Vec::with_capacity(gpu_batch_size)))
                .collect();

            // Batch occupancy, carried instead of recomputed. These were three
            // separate folds over the kind map evaluated on EVERY turn of the loop,
            // including the turns that found nothing — `Shaders::dispatch` empties
            // every batch it touches, so the pair is exact if it is reset there.
            //
            // Counted in INVOCATIONS, not messages: a row-driven message carries a
            // whole round, and it is rows that have to fit the batch table.
            let mut pending = 0usize;
            let mut max_len = 0usize;
            // Rows already queued per kind, for the same reason.
            let mut rows_in: HashMap<MessageKind, usize> =
                MessageKind::all().iter().map(|k| (*k, 0usize)).collect();

            loop {
                // With nothing queued there is nothing a timeout could decide, so
                // BLOCK. The former `recv_timeout(1 ns)` spun a whole core through
                // the gaps in which the solver does its CPU work, and the deadline's
                // `clock_gettime` alone was 5% of the process. While a batch IS
                // filling we keep the same short poll, because "the channel went
                // empty" is exactly what decides the flush — changing that would
                // change how work packs, which is a separate question.
                let msg = if pending == 0 {
                    match rx.recv() {
                        Ok(m) => Ok(m),
                        // Every sender gone: nothing more can arrive.
                        Err(_) => break,
                    }
                } else {
                    rx.recv_timeout(std::time::Duration::from_nanos(1))
                        .map_err(|_| ())
                };
                let mut empty = false;
                match msg {
                    Ok(Message::Job { input, respond_to }) => {
                        // Every kind is GPU-dispatched now. Its batch is
                        // pre-registered above; the lookup panics if this kind has
                        // none (a variant left unwired).
                        let kind = input.kind;
                        let n = input.invocations();
                        // A message that would overrun the threshold flushes what
                        // is already queued FIRST, then joins the empty batch. The
                        // threshold never exceeds the table, and the producer never
                        // sends a message wider than the table (`submit_rounds`
                        // chunks), so one always fits.
                        let full = *rows_in.get(&kind).unwrap() + n > gpu_batch_size;
                        if full && pending > 0 {
                            shaders.flush(&mut gpu_batches);
                            pending = 0;
                            max_len = 0;
                            rows_in.values_mut().for_each(|v| *v = 0);
                        }
                        gpu_batches
                            .get_mut(&kind)
                            .unwrap_or_else(|| panic!("no GPU batch registered for {kind:?}"))
                            .push((input, Some(respond_to)));
                        let queued = rows_in.get_mut(&kind).unwrap();
                        *queued += n;
                        pending += n;
                        max_len = max_len.max(*queued);
                    }
                    Ok(Message::Shutdown) => break,
                    _ => {
                        empty = true;
                    }
                };

                // `>=`, not `==`: under a batch-size hint smaller than a round,
                // one message alone can overshoot the threshold.
                let has_full = max_len >= gpu_batch_size;
                if pending > 0 && (has_full || empty) {
                    if log_flushes {
                        let non_zero = gpu_batches.values().filter(|v| !v.is_empty()).count();
                        eprintln!(
                            "[GPU] Total size of all {non_zero} batches: {pending:5} (max {max_len:5} in batch)"
                        );
                    }
                    shaders.flush(&mut gpu_batches);
                    pending = 0;
                    max_len = 0;
                    rows_in.values_mut().for_each(|v| *v = 0);
                }
            }
        });

        Accelerator {
            thread: Some(thread),
            tx,
            capacity,
            permits: Semaphore::new(self.max_pending),
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> Drop for Accelerator<T> {
    fn drop(&mut self) {
        // Ask the worker to stop (unblocks recv()) and wait for it to actually
        // finish before drop returns.
        let _ = self.tx.send(Message::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::implicit::block::{Block, block_mul_acc, zero_block};
    use aristotle::Epoch;
    use clifford::pga3::Twist;

    use aristotle::World;

    /// A run of 6x6 blocks held in world storage — the shape every block test
    /// here fills, dispatches and reads back.
    type Blocks = Vec<WorldKey<Block<f32>>>;
    /// One `simple_sum` case: the two addends, the destination, and the sum
    /// expected in it.
    type SumCase = (WorldKey<f32>, WorldKey<f32>, WorldKey<f32>, f32);

    /// A block product routed through the accelerator must equal the same
    /// product computed directly. Guards the row index arithmetic and the factor
    /// order across the dispatch boundary. The CPU implementation is gone, so the
    /// oracle is this test's own arithmetic.
    #[tokio::test]
    async fn accelerator_gemm_matches_direct_product() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = 2;
        let fill = |seed: f32| -> Vec<WorldKey<Block<f32>>> {
            let mut map = world.write::<Block<f32>>();
            (0..m * m)
                .map(|t| {
                    let mut b = zero_block::<f32>();
                    for (r, row) in b.iter_mut().enumerate() {
                        for (c, cell) in row.iter_mut().enumerate() {
                            *cell = seed + (t * 36 + r * 6 + c) as f32 * 0.25;
                        }
                    }
                    map.add(b)
                })
                .collect()
        };
        let a = fill(1.0);
        let b = fill(-2.0);
        let out: Vec<_> = {
            let mut map = world.write::<Block<f32>>();
            (0..m * m).map(|_| map.add(zero_block::<f32>())).collect()
        };
        let columns: Arc<[usize]> = Arc::from((0..m).collect::<Vec<_>>());
        let rows_of: Vec<Arc<[usize]>> = (0..m).map(|_| columns.clone()).collect();
        let baked = rows::gemm::bake(&world, &a, &b, &out, m, &rows_of, false);

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.gemm(&baked).await.unwrap();

        for i in 0..m {
            for j in 0..m {
                let mut want = zero_block::<f32>();
                for k in 0..m {
                    let (ab, bb) = (a[i * m + k].read(), b[k * m + j].read());
                    block_mul_acc(&mut want, &ab, &bb);
                }
                let got = out[i * m + j].read();
                for r in 0..6 {
                    for c in 0..6 {
                        assert!(
                            (got[r][c] - want[r][c]).abs() < 1e-3,
                            "gemm[{i}][{j}][{r}][{c}] = {} want {}",
                            got[r][c],
                            want[r][c]
                        );
                    }
                }
            }
        }
    }

    /// The `2I − R` fold: with it enabled the product must equal the product
    /// against `2I − b`, computed explicitly.
    #[tokio::test]
    async fn accelerator_gemm_folds_two_minus() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = 1;
        let (a, b, out): (Blocks, Blocks, Blocks) = {
            let mut map = world.write::<Block<f32>>();
            let mut ab = zero_block::<f32>();
            let mut bb = zero_block::<f32>();
            for r in 0..6 {
                for c in 0..6 {
                    ab[r][c] = 1.0 + (r * 6 + c) as f32;
                    bb[r][c] = 0.5 - (r * 6 + c) as f32 * 0.1;
                }
            }
            (
                vec![map.add(ab)],
                vec![map.add(bb)],
                vec![map.add(zero_block::<f32>())],
            )
        };
        let rows_of: Vec<Arc<[usize]>> = vec![Arc::from(vec![0usize])];
        let baked = rows::gemm::bake(&world, &a, &b, &out, m, &rows_of, true);

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.gemm(&baked).await.unwrap();

        let mut folded = b[0].read();
        for (r, row) in folded.iter_mut().enumerate() {
            for cell in row.iter_mut() {
                *cell = -*cell;
            }
            row[r] += 2.0;
        }
        let mut want = zero_block::<f32>();
        block_mul_acc(&mut want, &a[0].read(), &folded);

        let got = out[0].read();
        for r in 0..6 {
            for c in 0..6 {
                assert!(
                    (got[r][c] - want[r][c]).abs() < 1e-3,
                    "fold mismatch at [{r}][{c}]"
                );
            }
        }
    }

    /// A block copy through the accelerator must reproduce the source exactly and
    /// touch nothing else. Exact, not approximate: a copy that changes a bit is
    /// not a copy, and the solver publishes its warm hint through this.
    #[tokio::test]
    async fn gpu_block_copy_is_exact_and_leaves_neighbours_alone() {
        let world = Arc::new(World::builder().usual::<f32>());
        let n = 5;
        let (src, dst, untouched) = {
            let mut map = world.write::<Block<f32>>();
            let src: Vec<_> = (0..n)
                .map(|t| {
                    let mut b = zero_block::<f32>();
                    for (r, row) in b.iter_mut().enumerate() {
                        for (c, cell) in row.iter_mut().enumerate() {
                            *cell = (t * 36 + r * 6 + c) as f32 * 0.375 - 7.0;
                        }
                    }
                    map.add(b)
                })
                .collect();
            let dst: Vec<_> = (0..n).map(|_| map.add(zero_block::<f32>())).collect();
            // A block adjacent in the same storage, to catch a row that walks off
            // the end of its own slot.
            let mut sentinel = zero_block::<f32>();
            sentinel[2][3] = 42.0;
            (src, dst, map.add(sentinel))
        };

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel
            .block_copy(&rows::copy::bake(&world, &dst, &src))
            .await
            .unwrap();

        for (i, (d, s)) in dst.iter().zip(&src).enumerate() {
            let (got, want) = (d.read(), s.read());
            for r in 0..6 {
                for c in 0..6 {
                    assert_eq!(got[r][c], want[r][c], "block {i} at [{r}][{c}]");
                }
            }
        }
        assert_eq!(untouched.read()[2][3], 42.0, "copy wrote outside its slots");
    }

    /// `Σ ‖diag·I − b‖²` through the accelerator must equal the same sum computed
    /// here. Deliberately spans more blocks than fit one row, because the split
    /// into partials is the only thing between this kernel and a plain loop: a row
    /// that folded the wrong chunk, or a partial the host forgot to add, would
    /// still look right on a single-row reduction.
    #[tokio::test]
    async fn gpu_block_reduce_matches_direct_sum() {
        let world = Arc::new(World::builder().usual::<f32>());
        let n = rows::reduce::MAX_TERMS + 37;
        let blocks: Vec<WorldKey<Block<f32>>> = {
            let mut map = world.write::<Block<f32>>();
            (0..n)
                .map(|t| {
                    let mut b = zero_block::<f32>();
                    for (r, row) in b.iter_mut().enumerate() {
                        for (c, cell) in row.iter_mut().enumerate() {
                            // Spread of magnitudes and signs, and near 1 on the
                            // diagonal so the identity subtraction is not swamped.
                            *cell = ((t * 7 + r * 6 + c) % 13) as f32 * 0.25 - 1.5;
                        }
                        row[r] += 1.0;
                    }
                    map.add(b)
                })
                .collect()
        };
        // Every third block is a "diagonal" cell — an arbitrary but uneven pattern,
        // so a kernel ignoring the flag cannot pass by symmetry.
        let is_diag = |k: usize| k.is_multiple_of(3);
        let partials: Vec<WorldKey<f32>> = {
            let mut map = world.write::<f32>();
            (0..rows::reduce::partials_for(n))
                .map(|_| map.add(0.0))
                .collect()
        };
        assert!(partials.len() > 1, "test must actually span partials");
        let baked = rows::reduce::bake(&world, &blocks, is_diag, &partials);

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.block_reduce(&baked).await.unwrap();

        let mut want = 0.0f64;
        for (k, key) in blocks.iter().enumerate() {
            let b = key.read();
            let d = if is_diag(k) { 1.0f32 } else { 0.0 };
            for (r, row) in b.iter().enumerate() {
                for (c, &v) in row.iter().enumerate() {
                    let e = if r == c { d - v } else { -v };
                    want += (e * e) as f64;
                }
            }
        }
        let got: f64 = partials.iter().map(|k| k.read() as f64).sum();
        assert!(
            (got - want).abs() < 1e-3 * (1.0 + want.abs()),
            "reduce = {got}, want {want}"
        );
    }

    /// A joint family's two kernels must agree on the connection VALUE.
    ///
    /// Load-bearing for `eval_jacobians`: the Newton iteration after an accepted
    /// line-search probe re-runs only the Jacobian kernel and keeps the
    /// `total_wrench` the probe's PLAIN kernel produced, where it used to keep the
    /// one the Jacobian kernel produced. That swap is only invisible if the two
    /// kernels return the same `[wa, wb]` for the same inputs — which is not free:
    /// they are traced through viete SEPARATELY, one over the scalar carrier and
    /// one over `Tangent<24>`, so nothing but this test stops the folds from
    /// drifting apart.
    ///
    /// The comparison is EXACT because that is what the hardware does: measured on
    /// this state, every component matches bit for bit. A tolerance here would be
    /// a guess dressed as a bound, and would let a real divergence in one of the
    /// two traces through unnoticed.
    #[tokio::test]
    async fn plain_and_jacobian_kernels_agree_on_the_connection_value() {
        use crate::{Inert, RigidBody};
        use joints::{AxialSpringDamper, JointEdge, SimpleSpringDamper};

        let world = Arc::new(World::builder().usual::<f32>());
        // Off-axis, moving, and stretched well off the rest length: a state where
        // every term of the force law is live, not a degenerate one.
        let mk = |p: [f32; 3], v: [f32; 3]| {
            RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from(p),
                &Vector3::from(v),
                1.5,
            )
        };
        let (ida, idb) = (WorldId::get(), WorldId::get());
        let mut bodies: IndexMap<WorldId, Box<dyn Component<f32, f32>>> = IndexMap::new();
        bodies.insert(ida, Inert::new(mk([1.3, -0.4, 0.2], [0.0, 2.5, -1.0])));
        bodies.insert(idb, Inert::new(mk([-0.7, 0.9, -0.5], [1.0, -0.5, 0.75])));

        let joint = AxialSpringDamper::builder(world.clone(), SimpleSpringDamper)
            .rest(1.0)
            .stiffness(37.0)
            .damping(2.5)
            .build();
        let mut joints: IndexMap<WorldId, JointEdge<f32>> = IndexMap::new();
        joints.insert(WorldId::get(), JointEdge::new(ida, idb, joint));

        let mut external: IndexMap<WorldId, WorldKey<Wrench<f32>>> = IndexMap::new();
        {
            let mut map = world.write::<Wrench<f32>>();
            external.insert(ida, map.add(Wrench::zero()));
            external.insert(idb, map.add(Wrench::zero()));
        }
        bake_incidence(&mut bodies, &joints, &external);

        for e in bodies.values() {
            e.body().world_velocity();
        }
        let conn = joints.values().next().unwrap().wrench_key();
        let epoch = Epoch::standalone(1.0 / 60.0, 1.0);

        // One lane is enough: this compares two KERNELS, not two iterates.
        let mut cache = crate::integrator::implicit::cache::NewtonCache::<f32>::new();
        cache.bake_lanes(&world, &bodies, &joints, &external, 1);
        cache.publish_dispatch_scalars(1.0 / 120.0, *epoch.dt(), *epoch.warp());

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel
            .dispatch(&cache.lanes()[0], false, true)
            .await
            .unwrap();
        let plain = conn.read();
        accel.dispatch(&cache.lanes()[0], true, true).await.unwrap();
        let jac = conn.read();

        // Non-degenerate, or the comparison proves nothing about the force law.
        assert!(
            plain[0].force().split().iter().any(|c| c.abs() > 1.0),
            "test state produced no force: {:?}",
            plain[0].force().split()
        );
        for end in 0..2 {
            for (label, p, j) in [
                ("force", plain[end].force(), jac[end].force()),
                ("torque", plain[end].torque(), jac[end].torque()),
            ] {
                for k in 0..3 {
                    assert!(
                        p[k] == j[k],
                        "end {end} {label}[{k}]: plain {} vs jacobian {} (ulp diff {})",
                        p[k],
                        j[k],
                        (p[k].to_bits() as i64 - j[k].to_bits() as i64).abs()
                    );
                }
            }
        }
    }

    /// The gather must equal `external + Σ` incident wrenches, computed here.
    /// The CPU implementation is gone, so the oracle is this test's own
    /// arithmetic, not the old code path.
    #[tokio::test]
    async fn gpu_gather_sums_incident_wrenches() {
        let world = Arc::new(World::builder().usual::<f32>());
        let w = |f: [f32; 3], t: [f32; 3]| Wrench::new(&Vector3::from(f), &Vector3::from(t));
        let (out, ext) = {
            let mut map = world.write::<Wrench<f32>>();
            (
                map.add(Wrench::zero()),
                map.add(w([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])),
            )
        };
        let pairs: Vec<_> = {
            let mut map = world.write::<[Wrench<f32>; 2]>();
            (0..5)
                .map(|i| {
                    let a = i as f32;
                    map.add([w([a, a, a], [a, a, a]), w([-a, -a, -a], [-a, -a, -a])])
                })
                .collect()
        };
        let terms: Vec<GatherTerm<f32>> = pairs
            .iter()
            .map(|k| GatherTerm {
                key: k.clone(),
                slot: 0,
            })
            .collect();
        let baked = rows::gather::bake(&world, &out, &ext, &terms);

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.gather_rows(&baked).await.unwrap();

        // external + Σ_{i<5} i  on every component.
        let got = out.read();
        assert!(
            (got.force()[0] - (1.0 + 10.0)).abs() < 1e-5,
            "{:?}",
            got.force()
        );
        assert!(
            (got.force()[2] - (3.0 + 10.0)).abs() < 1e-5,
            "{:?}",
            got.force()
        );
        assert!(
            (got.torque()[2] - (6.0 + 10.0)).abs() < 1e-5,
            "{:?}",
            got.torque()
        );
    }

    /// A reduction longer than one row must equal the single-round result. This
    /// is the only cover for `accumulate`: round 0 seeds from `external`, later
    /// rounds seed from the output they are extending.
    #[tokio::test]
    async fn gpu_gather_multi_round_matches_single_round() {
        let world = Arc::new(World::builder().usual::<f32>());
        let n = rows::gather::MAX_TERMS + 7;
        let one = Wrench::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO);
        let (out, ext) = {
            let mut map = world.write::<Wrench<f32>>();
            (map.add(Wrench::zero()), map.add(Wrench::zero()))
        };
        let pairs: Vec<_> = {
            let mut map = world.write::<[Wrench<f32>; 2]>();
            (0..n).map(|_| map.add([one, Wrench::zero()])).collect()
        };
        let terms: Vec<GatherTerm<f32>> = pairs
            .iter()
            .map(|k| GatherTerm {
                key: k.clone(),
                slot: 0,
            })
            .collect();
        let baked = rows::gather::bake(&world, &out, &ext, &terms);
        assert!(baked.len() > 1, "test must actually span rounds");

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.gather_rows(&baked).await.unwrap();

        assert!((out.read().force()[0] - n as f32).abs() < 1e-3);
    }

    /// A large flush followed by a small one: the small dispatch must not touch
    /// the slots the large one named. The batch table is reused across flushes,
    /// so rows past `count` still hold the previous flush's indices; the kernel's
    /// `idx >= count` guard is what stops the tail workgroup following them.
    #[tokio::test]
    async fn gpu_gather_short_flush_after_long_ignores_stale_rows() {
        let world = Arc::new(World::builder().usual::<f32>());
        let one = Wrench::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO);
        let accel = Accelerator::<f32>::builder(world.clone()).build();

        let mut outs = Vec::new();
        let mut rows_all = Vec::new();
        // The row holds raw indices, so the slot keys must outlive it:
        // a dropped `ext` would return to the free-list and be handed to `out` on the next
        // iteration, chaining independent rows together within one batch.
        let mut alive = Vec::new();
        for _ in 0..500 {
            let (out, ext) = {
                let mut map = world.write::<Wrench<f32>>();
                (map.add(Wrench::zero()), map.add(Wrench::zero()))
            };
            let pair = {
                let mut map = world.write::<[Wrench<f32>; 2]>();
                map.add([one, Wrench::zero()])
            };
            let terms = vec![GatherTerm { key: pair, slot: 0 }];
            rows_all.extend(rows::gather::bake(&world, &out, &ext, &terms));
            outs.push(out);
            alive.push((ext, terms));
        }
        let futs = rows_all
            .iter()
            .map(|r| accel.gather_rows(std::slice::from_ref(r)));
        for r in join_all(futs).await {
            r.unwrap();
        }

        // Second, deliberately tiny flush against ONE fresh body with no terms.
        let (out2, ext2) = {
            let mut map = world.write::<Wrench<f32>>();
            (map.add(Wrench::zero()), map.add(Wrench::zero()))
        };
        let row2 = rows::gather::bake(&world, &out2, &ext2, &[]);
        accel.gather_rows(&row2).await.unwrap();

        for o in &outs {
            assert!(
                (o.read().force()[0] - 1.0).abs() < 1e-5,
                "a stale row re-ran into an earlier output"
            );
        }
        assert!(out2.read().force()[0].abs() < 1e-5);
    }

    /// `dv = x · rhs` through the accelerator must equal the same product
    /// computed here — the same shape of check as the gemm test.
    #[tokio::test]
    async fn gpu_block_matvec_matches_direct_product() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = 2;
        let x: Vec<_> = {
            let mut map = world.write::<Block<f32>>();
            (0..m * m)
                .map(|t| {
                    let mut b = zero_block::<f32>();
                    for (r, row) in b.iter_mut().enumerate() {
                        for (c, cell) in row.iter_mut().enumerate() {
                            *cell = 0.1 * (t * 36 + r * 6 + c) as f32;
                        }
                    }
                    map.add(b)
                })
                .collect()
        };
        let rhs: Vec<_> = {
            let mut map = world.write::<Wrench<f32>>();
            (0..m)
                .map(|j| {
                    let a = (j + 1) as f32;
                    map.add(Wrench::new(
                        &Vector3::from([a, 2.0 * a, 3.0 * a]),
                        &Vector3::from([4.0 * a, 5.0 * a, 6.0 * a]),
                    ))
                })
                .collect()
        };
        let dv: Vec<_> = {
            let mut map = world.write::<Twist<f32>>();
            (0..m).map(|_| map.add(Twist::zero())).collect()
        };
        let baked = rows::matvec::bake(&world, &x, &rhs, &dv, m);

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.block_matvec(&baked).await.unwrap();

        for i in 0..m {
            let mut want = [0.0f32; 6];
            for j in 0..m {
                let b = x[i * m + j].read();
                let w = rhs[j].read();
                let (f, t) = (w.force(), w.torque());
                let v = [f[0], f[1], f[2], t[0], t[1], t[2]];
                for r in 0..6 {
                    for c in 0..6 {
                        want[r] += b[r][c] * v[c];
                    }
                }
            }
            let got = dv[i].read();
            let (l, a) = (got.linear(), got.angular());
            let g = [l[0], l[1], l[2], a[0], a[1], a[2]];
            for r in 0..6 {
                assert!(
                    (g[r] - want[r]).abs() < 1e-3,
                    "dv[{i}][{r}] = {} want {}",
                    g[r],
                    want[r]
                );
            }
        }
    }

    /// One connection's Jacobian must land in the matrix with the midpoint
    /// factors applied and the mass block added exactly once, on the diagonal.
    #[tokio::test]
    async fn gpu_assemble_applies_factors_and_adds_mass_once() {
        use crate::integrator::implicit::cache::BlockTerm;
        let world = Arc::new(World::builder().usual::<f32>());
        let half = 0.25f32;
        let jac = {
            let mut map = world.write::<[[Wrench<f32>; 24]; 2]>();
            let mut block = [[Wrench::zero(); 24]; 2];
            // Pose column 0 of end A: unit force along x.
            block[0][0] = Wrench::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO);
            // Velocity column 0 of end A lives at axis 12.
            block[0][12] = Wrench::new(&Vector3::from([0.0, 1.0, 0.0]), &Vector3::ZERO);
            map.add(block)
        };
        let (mass, out) = {
            let mut map = world.write::<Block<f32>>();
            let mut mb = zero_block::<f32>();
            mb[2][2] = 7.0;
            (map.add(mb), map.add(zero_block::<f32>()))
        };
        let terms = vec![BlockTerm {
            key: jac.clone(),
            row_end: 0,
            col_end: 0,
        }];
        // `bake_cell` gives ONE cell's rounds; `assemble` wants rounds-major, so
        // each of them is a round of its own with a single row in it.
        let (pf, vf) = {
            let mut map = world.write::<f32>();
            (map.add(-(half * half)), map.add(-half))
        };
        let baked: Vec<Arc<[WorldKey<row::Row>]>> =
            rows::assemble::bake_cell(&world, &out, &mass, true, &pf, &vf, &terms)
                .into_iter()
                .map(|r| Arc::from(vec![r]))
                .collect();

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.assemble(&baked).await.unwrap();

        let got = out.read();
        assert!((got[0][0] - -(half * half)).abs() < 1e-6, "pose factor");
        assert!((got[1][0] - -half).abs() < 1e-6, "velocity factor");
        assert!((got[2][2] - 7.0).abs() < 1e-6, "mass added once");
    }

    /// A single free body at rest: the mass block must be the inertia in world
    /// coordinates, the residual zero, and the self-scale exactly the floor.
    #[tokio::test]
    async fn gpu_body_post_at_rest_has_zero_residual_and_floor_scale() {
        use crate::{Inertia, RigidBody};
        use rows::body_post::{BodyRow, bake};
        let world = Arc::new(World::builder().usual::<f32>());
        let inertia = Inertia::diagonal(world.clone(), 2.0f32, [3.0, 4.0, 5.0]);
        let body = RigidBody::new(world.clone(), inertia);
        body.midpoint_pose.write(body.pose.read());
        body.solve_vel.write(Twist::zero());

        let (mass_out, rhs_out, scale_out) = {
            let mut bm = world.write::<Block<f32>>();
            let m = bm.add(zero_block::<f32>());
            drop(bm);
            let mut wm = world.write::<Wrench<f32>>();
            (m, wm.add(Wrench::zero()), wm.add(Wrench::zero()))
        };
        let (half_slot, floor2_slot) = {
            let mut map = world.write::<f32>();
            (map.add(0.5f32), map.add(1e-18f32))
        };
        let baked = bake(
            &world,
            &[BodyRow {
                midpoint_pose: body.midpoint_pose.clone(),
                solve_vel: body.solve_vel.clone(),
                inertia: body.inertia.keys(),
                snap_mom: body.world_momentum(),
                total_wrench: body.total_wrench.clone(),
                mass_out: mass_out.clone(),
                rhs_out: rhs_out.clone(),
                scale_out: scale_out.clone(),
            }],
            &half_slot,
            &floor2_slot,
        );

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.body_post(&baked).await.unwrap();

        let mb = mass_out.read();
        assert!((mb[0][0] - 2.0).abs() < 1e-4, "mass on the linear diagonal");
        assert!(
            (mb[3][3] - 3.0).abs() < 1e-4,
            "inertia on the angular diagonal"
        );
        let rhs = rhs_out.read();
        assert!(rhs.force()[0].abs() < 1e-5 && rhs.torque()[2].abs() < 1e-5);
        let sc = scale_out.read();
        assert!(
            (sc.force()[0] - 1e-18).abs() < 1e-24,
            "scale falls back to the floor"
        );
    }

    #[test]
    fn builds_and_drops() {
        // Compiles the kernels in the worker (backend-lua) and shuts down cleanly.
        let world = Arc::new(World::builder().usual::<f32>());
        let _accel = Accelerator::<f32>::builder(world.clone()).build();
    }

    /// The GPU kernel declares `float[]`, so binding the world's scalar map is
    /// only sound for `T = f32`. Same erased-scalar guard as the Lua backend's
    /// `t_to_f64`: a loud panic, not a silently wrong binding.
    #[test]
    #[should_panic(expected = "gpu backend: only f32")]
    fn gpu_path_rejects_non_f32_world() {
        let world = Arc::new(World::builder().usual::<f64>());
        let _accel = Accelerator::<f64>::builder(world).build();
    }

    /// A flush that fails must answer EVERY job it carried with
    /// `EvalError::Backend`, leave their outputs unwritten, and leave the worker
    /// serving: the batch after it, on the same accelerator, succeeds.
    #[tokio::test]
    async fn failed_flush_reports_backend_to_every_job_and_worker_survives() {
        let world = Arc::new(World::builder().usual::<f32>());
        const N: usize = 64;
        let mut map = world.write::<f32>();
        let sums: Vec<SumCase> = (0..N)
            .map(|i| {
                let i = i as f32 + 1.0;
                (map.add(i), map.add(i), map.add(0.0), i + i)
            })
            .collect();
        drop(map);

        let fault = Arc::new(FaultInjector::default());
        fault.fail.store(1, std::sync::atomic::Ordering::SeqCst);
        let accel = Accelerator::<f32>::builder(world.clone())
            .inject_faults(fault.clone())
            .build();

        // All N go out together, so the failed flush carries more than one job.
        let results = join_all(sums.iter().map(|s| accel.simple_sum(&s.0, &s.1, &s.2))).await;

        let failed = fault.failed_jobs.lock().unwrap().clone();
        assert_eq!(failed.len(), 1, "exactly one flush was set to fail");
        let errs: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
        assert_eq!(
            errs.len(),
            failed[0],
            "every job of the failed flush, and only those, must see the error"
        );
        assert!(failed[0] >= 1);
        for e in errs {
            match e {
                Err(EvalError::Backend { shader, .. }) => assert_eq!(*shader, "injected"),
                other => panic!("expected EvalError::Backend, got {other:?}"),
            }
        }
        for (s, r) in sums.iter().zip(&results) {
            match r {
                Ok(()) => assert_eq!(s.2.read(), s.3),
                // Failed before submission: the kernel never ran.
                Err(_) => assert_eq!(s.2.read(), 0.0),
            }
        }

        // The same accelerator serves the next batch.
        let again: Vec<_> = join_all(sums.iter().map(|s| accel.simple_sum(&s.0, &s.1, &s.2))).await;
        for (s, r) in sums.iter().zip(again) {
            r.expect("the batch after a failed one must succeed");
            assert_eq!(s.2.read(), s.3);
        }
        assert_eq!(fault.failed_jobs.lock().unwrap().len(), 1);
    }

    /// Several failed flushes in a row, with a batch-size hint of one row so each
    /// carries exactly one job: each job gets its own error, then the worker
    /// recovers.
    #[tokio::test]
    async fn consecutive_failed_flushes_each_report_their_own_job() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mut map = world.write::<f32>();
        let s: SumCase = (map.add(2.0), map.add(3.0), map.add(0.0), 5.0);
        drop(map);

        let fault = Arc::new(FaultInjector::default());
        fault.fail.store(3, std::sync::atomic::Ordering::SeqCst);
        let accel = Accelerator::<f32>::builder(world.clone())
            .batch_size(1)
            .inject_faults(fault.clone())
            .build();

        for _ in 0..3 {
            assert!(matches!(
                accel.simple_sum(&s.0, &s.1, &s.2).await,
                Err(EvalError::Backend { .. })
            ));
        }
        assert_eq!(*fault.failed_jobs.lock().unwrap(), vec![1, 1, 1]);
        assert_eq!(s.2.read(), 0.0);
        accel.simple_sum(&s.0, &s.1, &s.2).await.unwrap();
        assert_eq!(s.2.read(), s.3);
    }

    #[tokio::test]
    async fn gpu_simple_sum() {
        let world = Arc::new(World::builder().usual::<f32>());

        let mut map = world.write::<f32>();
        let sums: Vec<SumCase> = (0..6000)
            .map(|i| {
                (
                    map.add(i as f32),
                    map.add(i as f32),
                    map.add(0 as f32),
                    (i + i) as f32,
                )
            })
            .collect();
        drop(map);
        let accel = Accelerator::<f32>::builder(world.clone()).build();
        let futs = sums
            .iter()
            .map(|chunk| accel.simple_sum(&chunk.0, &chunk.1, &chunk.2));
        for r in join_all(futs).await {
            r.unwrap();
        }
        eprintln!("Batch finished");
        for s in sums {
            assert_eq!(s.2.read(), s.3);
        }
    }
}
