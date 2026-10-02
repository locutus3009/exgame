// SPDX-License-Identifier: MIT

use super::row::Row;
use super::{MessageInput, MessageKind, MessagePayload, Pending, PipelineBindPoint};
use crate::EvalError;
use aristotle::World;
use bytemuck::Pod;
use clifford::pga3::{Motor, Twist, Wrench};
use futures::channel::oneshot;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    pipeline::{ComputePipeline, Pipeline},
};

/// Fatal slots each generated kernel writes per invocation, one `usize` per
/// kernel, emitted by `build.rs` from the traces themselves. These are the counts
/// `check` scans; none is entered by hand.
mod fatal_counts {
    include!(concat!(env!("OUT_DIR"), "/fatal_counts.rs"));
}

/// Declared length of a shader's `Fatals.fatal[]`, read off the struct vulkano
/// reflects from the compiled SPIR-V. Used for the one hand-written shader,
/// `simple_sum.glsl`, which has no trace to count and only ever clears its slot.
const fn fatal_width<F>() -> usize {
    std::mem::size_of::<F>() / std::mem::size_of::<f32>()
}

pub(super) mod shader_simple_sum {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/accelerator/simple_sum.glsl"
    }
}

pub(super) mod critically_damped_warped_plain {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "CriticallyDampedWarped_plain.glsl"
    }
}

pub(super) mod critically_damped_warped_jacobian {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "CriticallyDampedWarped_jacobian.glsl"
    }
}

pub(super) mod perpendicular_damper_warped_plain {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "PerpendicularDamperWarped_plain.glsl"
    }
}

pub(super) mod perpendicular_damper_warped_jacobian {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "PerpendicularDamperWarped_jacobian.glsl"
    }
}

pub(super) mod simple_spring_damper_plain {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "SimpleSpringDamper_plain.glsl"
    }
}

pub(super) mod simple_spring_damper_jacobian {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "SimpleSpringDamper_jacobian.glsl"
    }
}

pub(super) mod torsional_damper_warped_plain {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "TorsionalDamperWarped_plain.glsl"
    }
}

pub(super) mod torsional_damper_warped_jacobian {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "TorsionalDamperWarped_jacobian.glsl"
    }
}

pub(super) mod pre_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "Pre.glsl"
    }
}

pub(super) mod gather_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "Gather.glsl"
    }
}

pub(super) mod block_matvec_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BlockMatVec.glsl"
    }
}

pub(super) mod gemm_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "Gemm.glsl"
    }
}

pub(super) mod assemble_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "AssembleBlock.glsl"
    }
}

pub(super) mod block_reduce_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BlockReduce.glsl"
    }
}

pub(super) mod block_copy_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BlockCopy.glsl"
    }
}

pub(super) mod body_post_diagonal_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BodyPostDiagonal.glsl"
    }
}

pub(super) mod body_post_gathered_diagonal_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BodyPostGatheredDiagonal.glsl"
    }
}

pub(super) mod body_post_gathered_full_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BodyPostGatheredFull.glsl"
    }
}

pub(super) mod body_post_full_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "OUT_DIR",
        path: "BodyPostFull.glsl"
    }
}

pub(crate) fn t_to_f32<T: Pod + 'static>(x: T) -> f32 {
    let id = TypeId::of::<T>();
    if id == TypeId::of::<f32>() {
        unsafe { *(&x as *const T as *const f32) }
    } else {
        panic!("lua backend: only f32, got {}", std::any::type_name::<T>());
    }
}

/// A shader's descriptor set together with the generation of every world
/// storage it binds, in `storages` order, as read when the set was written.
///
/// World storage is reallocated when it grows, and a set keeps the buffer it was
/// written with alive — so a set over a grown map would silently compute on the
/// old copy. `setup` compares these against `World::generation` and rewrites
/// the set first; `check` compares them again, to catch a growth that slipped in
/// after the set was bound.
struct Bound {
    set: Arc<DescriptorSet>,
    generations: Vec<u64>,
}

/// One OUTPUT column of a row layout: the row word that names the slot, and the
/// world storage that slot lives in.
#[derive(Clone, Copy)]
pub(super) struct Col {
    word: usize,
    id: fn() -> TypeId,
    name: fn() -> &'static str,
}

/// Where a row-driven kind's OUTPUT slots sit in its row — the write half of
/// the stage's word layout (`rows/`), next to the binding list in the table
/// below. Read slots are not listed: two invocations may read one slot, and one
/// invocation may read and write its own (Gather seeds `out` from itself); only
/// two WRITERS of one slot in one batch break the storage's invariant.
///
/// A row carries `count` records (`None`: exactly one) of `stride` words from
/// word `base`, and each record writes the slots its `cols` name.
#[derive(Clone, Copy)]
pub(super) struct Writes {
    count: Option<usize>,
    base: usize,
    stride: usize,
    cols: &'static [Col],
}

impl Writes {
    /// Every `(storage, slot)` the row writes; `Err` if the row's record count
    /// runs past its end, which no baker produces.
    fn each(&self, row: &Row, mut f: impl FnMut(&Col, u32)) -> Result<(), String> {
        let n = self.count.map_or(1, |w| row[w] as usize);
        for k in 0..n {
            let at = self.base + k * self.stride;
            for c in self.cols {
                let slot = row.get(at + c.word).ok_or_else(|| {
                    format!("flush refused: a row declares {n} records, more than it holds")
                })?;
                f(c, *slot);
            }
        }
        Ok(())
    }
}

/// A row with one record: `writes!(word => Type, …)`. With `per count, base,
/// stride;` in front, the row carries `row[count]` records.
macro_rules! writes {
    (per $count:expr, $base:expr, $stride:expr; $($word:expr => $ty:ty),* $(,)?) => {
        Writes {
            count: Some($count),
            base: $base,
            stride: $stride,
            cols: &[$(Col {
                word: $word,
                id: TypeId::of::<$ty>,
                name: std::any::type_name::<$ty>,
            }),*],
        }
    };
    ($($word:expr => $ty:ty),* $(,)?) => {
        Writes {
            count: None,
            base: 0,
            stride: 0,
            cols: &[$(Col {
                word: $word,
                id: TypeId::of::<$ty>,
                name: std::any::type_name::<$ty>,
            }),*],
        }
    };
}

/// Slots of one storage claimed in the current flush, as a bitset over raw slot
/// indices. `touched` lists the words that went non-zero, so clearing costs what
/// the flush wrote rather than the storage's capacity.
#[derive(Default)]
struct Claimed {
    bits: Vec<u64>,
    touched: Vec<usize>,
}

impl Claimed {
    /// Claim `slot`; `false` if it was already claimed.
    fn claim(&mut self, slot: u32) -> bool {
        let (w, b) = (slot as usize / 64, 1u64 << (slot % 64));
        if w >= self.bits.len() {
            self.bits.resize(w + 1, 0);
        }
        let word = &mut self.bits[w];
        if *word & b != 0 {
            return false;
        }
        if *word == 0 {
            self.touched.push(w);
        }
        *word |= b;
        true
    }

    fn clear(&mut self) {
        for w in self.touched.drain(..) {
            self.bits[w] = 0;
        }
    }
}

#[derive(Default)]
struct LedgerState {
    /// Kinds set up in the current flush whose `check` has not run yet. Every
    /// non-empty batch gets exactly one `setup` and, once the submission is
    /// done, one `check`; a `setup` that finds this at zero opens a new flush.
    open: usize,
    /// Per output storage, the slots claimed so far in this flush.
    claimed: Vec<(TypeId, Claimed)>,
    /// Why this flush was refused, once it has been.
    refused: Option<String>,
}

/// The one-writer-per-slot check (ACCELERATOR.md Part III), shared by every
/// kind of one accelerator because a flush spans kinds.
///
/// Each `setup` claims the output slots of every row it is about to dispatch.
/// A slot claimed twice in one flush — by two rows of one kind or by two kinds —
/// REFUSES the flush: the kind that found the collision and every kind set up
/// after it dispatch nothing, and `check` fails every message of the flush with
/// `EvalError::Backend`. Kinds already recorded before the collision was seen
/// did run (`Shaders::dispatch` records each kind as it sets it up, so a later
/// kind cannot withdraw an earlier one), but they ran without the colliding
/// writer, and their callers are failed all the same: the flush as a whole
/// broke the invariant.
#[derive(Default)]
pub(super) struct Ledger(Mutex<LedgerState>);

impl Ledger {
    /// Open the kind's part of the flush and claim its output slots. `Err`
    /// carries the refusal; the caller then dispatches nothing.
    fn claim(&self, kind: &'static str, writes: &Writes, batch: &[Pending]) -> Result<(), String> {
        let mut st = self.0.lock().unwrap();
        if st.open == 0 {
            st.claimed.iter_mut().for_each(|(_, c)| c.clear());
            st.refused = None;
        }
        st.open += 1;
        if let Some(why) = &st.refused {
            return Err(why.clone());
        }
        let LedgerState {
            claimed, refused, ..
        } = &mut *st;
        let mut clash = None;
        'rows: for (MessageInput { payload, .. }, _) in batch {
            let MessagePayload::Rows { rows } = payload;
            for row in rows.iter() {
                let walked = writes.each(&row.read(), |c, slot| {
                    if clash.is_some() {
                        return;
                    }
                    let id = (c.id)();
                    let at = match claimed.iter().position(|(t, _)| *t == id) {
                        Some(at) => at,
                        None => {
                            claimed.push((id, Claimed::default()));
                            claimed.len() - 1
                        }
                    };
                    if !claimed[at].1.claim(slot) {
                        clash = Some(format!(
                            "flush refused: slot {slot} of world storage {} has a second \
                             writer in this batch (found by {kind}); no two invocations of \
                             one flush may write the same slot",
                            (c.name)()
                        ));
                    }
                });
                if let Err(why) = walked {
                    clash.get_or_insert(why);
                }
                if clash.is_some() {
                    break 'rows;
                }
            }
        }
        match clash {
            Some(why) => {
                *refused = Some(why.clone());
                Err(why)
            }
            None => Ok(()),
        }
    }

    /// Close the kind's part of the flush; the refusal, if the flush was refused.
    fn close(&self) -> Option<String> {
        let mut st = self.0.lock().unwrap();
        st.open -= 1;
        st.refused.clone()
    }
}

pub(super) trait ShaderSetup: Sync + Send {
    fn storages(&self) -> &Vec<TypeId>;
    /// Fill the batch table and bind the pipeline; return how many INVOCATIONS the
    /// batch needs. That is not `batch.len()` for the row-driven stages: one
    /// message there carries a whole round, so it occupies a contiguous span of
    /// the table. Rounds from different mechanisms simply land next to each other
    /// and go out in the same dispatch.
    fn setup(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        batch: &[Pending],
    ) -> Result<u32, EvalError>;
    /// Report each message's outcome and RETIRE it, handing the drained inputs
    /// back rather than dropping them.
    ///
    /// Fatal semantics are MARK-AND-CONTINUE, per message. A kernel never aborts:
    /// every invocation runs to the end and writes all of its outputs, and a
    /// tripped division guard only sets that invocation's fatal slot. The host
    /// then fails exactly the messages whose span holds a set slot, with
    /// `EvalError::Backend`; every other message in the same dispatch succeeds.
    /// The outputs a failed message wrote are still in its slots and are garbage
    /// (an `inf` or `NaN` from the unguarded division) — the caller must not
    /// consume them. Nothing is rolled back, and nothing else is affected: the
    /// slots are per invocation and outputs are disjoint per message.
    ///
    /// The caller owns when they die, and that is load-bearing: a message holds
    /// `WorldKey`s, and dropping the last handle to one reaches into
    /// `World::write` for its storage — the very lock the flush holds while the
    /// device is working. Dropping them here would deadlock the worker against
    /// itself, and only a message that OWNS its rows outright would show it:
    /// everything the solver submits is baked once and kept alive by the cache,
    /// so the refcount never reaches zero. `simple_sum` is the one that does.
    fn check(&self, batch: &mut Vec<Pending>) -> Vec<MessageInput>;
}

/// Generate a plain connection shader: its `struct`, the shared setup skeleton
/// (`backend` / `setup_dispatch` / `new`), and the `setup` that fills the
/// per-connection incidence rows. `$module` is the `vulkano_shaders::shader!`
/// module, `$variant` the `MessageInput` arm the batch carries, and the trailing
/// `[$ty, …]` the world-storage types for descriptor bindings from 2 upward —
/// bindings 0 and 1 are ALWAYS `fatals` and `incidence` by convention. Every
/// joint kernel shares the same incidence layout, so only `SimpleSum` needs its
/// own `setup` fill (invoked via the `@simple` arm).
macro_rules! plain_shader {
    // Shared skeleton: struct + backend + setup_dispatch + new. Bindings 0/1 are
    // fatals/incidence; the rest are `w.read::<$ty>()` maps, in order, from 2 up.
    (@common $name:ident, $module:ident, $n_fatals:expr, [$($ty:ty),* $(,)?]) => {
        pub(super) struct $name {
	    storages: Vec<TypeId>,
	    n_fatals: usize,
            incidence: Subbuffer<[$module::Incidence]>,
            #[allow(dead_code)]
            fatals: Subbuffer<[$module::Fatals]>,
            /// The world the bound storages live in, to rebind them after growth.
            world: Arc<World>,
            bound: Mutex<Bound>,
            pipeline: Arc<ComputePipeline>,
            /// The output slots of a row, for the one-writer check.
            writes: Writes,
            ledger: Arc<Ledger>,
        }

        // The host scans `n_fatals` slots of the reflected `Fatals`, so the count
        // must fit the array the shader declares. `build.rs` sizes both from the
        // same trace; this holds them together at compile time.
        const _: () = assert!(
            $n_fatals <= fatal_width::<$module::Fatals>(),
            "fatal count exceeds the shader's Fatals array"
        );

        impl $name {
            fn backend(e: impl std::fmt::Display) -> EvalError {
                EvalError::Backend {
                    shader: std::any::type_name::<Self>(),
                    msg: e.to_string(),
                }
            }

            fn setup_dispatch(
                &self,
                builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                count: u32,
            ) -> Result<(), EvalError> {
                builder
                    .bind_pipeline_compute(self.pipeline.clone())
                    .map_err(Self::backend)?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.pipeline.layout().clone(),
                        0,
                        self.current_set(),
                    )
                    .map_err(Self::backend)?
                    .push_constants(
                        self.pipeline.layout().clone(),
                        0,
                        $module::Params { count },
                    )
                    .map_err(Self::backend)?;
                Ok(())
            }

            /// The descriptor set over the CURRENT buffer of every bound
            /// storage, rewritten first if any of them has grown since it was
            /// written. Called while recording, before the dispatch takes its
            /// guards, so the read guards `bind` takes cannot meet its own.
            fn current_set(&self) -> Arc<DescriptorSet> {
                let mut bound = self.bound.lock().unwrap();
                if self.stale(&bound).is_some() {
                    *bound = Self::bind(
                        &self.world,
                        &self.pipeline,
                        &self.fatals,
                        &self.incidence,
                    )
                    .0;
                }
                bound.set.clone()
            }

            /// The first bound storage whose generation moved since `bound`
            /// was written, if any.
            fn stale(&self, bound: &Bound) -> Option<TypeId> {
                self.storages
                    .iter()
                    .zip(&bound.generations)
                    .find(|(t, g)| self.world.generation(**t) != **g)
                    .map(|(t, _)| *t)
            }

            /// Write a descriptor set over the current buffers. Each buffer and
            /// its generation are read under the same read guard, so they agree.
            fn bind(
                w: &World,
                pipeline: &Arc<ComputePipeline>,
                fatals: &Subbuffer<[$module::Fatals]>,
                incidence: &Subbuffer<[$module::Incidence]>,
            ) -> (Bound, Vec<TypeId>) {
                let ds_allocator = w.gpu().ds_allocator().clone();
                // Bindings 0/1 are fatals/incidence by convention, the $ty list
                // fills 2.. — but shaderc STRIPS any buffer a kernel never touches
                // (e.g. `fatals` when the trace has no fatal leaf), so the pipeline
                // layout may lack it. Write only bindings the reflected layout
                // actually declares; a write to a stripped binding is a hard error.
                let layout = pipeline.layout().set_layouts()[0].clone();
                let has = |b: u32| layout.bindings().contains_key(&b);
                let mut writes = Vec::new();
                if has(0) {
                    writes.push(WriteDescriptorSet::buffer(0, fatals.clone()));
                }
                if has(1) {
                    writes.push(WriteDescriptorSet::buffer(1, incidence.clone()));
                }
                let mut binding = 2u32;
                let mut storages = Vec::new();
                let mut generations = Vec::new();
                $(
                    if has(binding) {
                        let view = w.read::<$ty>();
                        writes.push(WriteDescriptorSet::buffer(
                            binding,
                            view.get_map().clone(),
                        ));
                        generations.push(view.generation());
                        storages.push(TypeId::of::<$ty>());
                    }
                    binding += 1;
                )*
                let _b = binding;
                let set = DescriptorSet::new(ds_allocator, layout, writes, []).unwrap();
                (Bound { set, generations }, storages)
            }

            pub(super) fn new(w: &Arc<World>, writes: Writes, ledger: Arc<Ledger>) -> Self {
                let g = w.gpu();
                let shader_batch_size = g.gpu_in_flight();
                let shader = $module::load(g.device().clone()).unwrap();
                let pipeline = g.build_pipeline(&shader, "main");
                let incidence = g.allocate_buffer::<$module::Incidence>(shader_batch_size);
                let fatals = g.allocate_buffer::<$module::Fatals>(shader_batch_size);
                let (bound, storages) = Self::bind(w, &pipeline, &fatals, &incidence);
                // An output the kernel does not bind is a layout table out of
                // step with the kernel: the check would guard the wrong storage.
                for c in writes.cols {
                    assert!(
                        storages.contains(&(c.id)()),
                        "{}: output storage {} is not bound",
                        std::any::type_name::<Self>(),
                        (c.name)()
                    );
                }
                $name {
		    storages,
		    n_fatals: $n_fatals,
                    incidence,
                    fatals,
                    world: w.clone(),
                    bound: Mutex::new(bound),
                    pipeline,
                    writes,
                    ledger,
                }
            }
        }
    };




    // ROW-driven stage: one message carries a whole ROUND, and that round occupies
    // a CONTIGUOUS SPAN of the batch table. The row itself, baked on topology
    // change, holds all the indices, so the step-time cost is one `Arc` deref and
    // one store per invocation.
    //
    // The span is what keeps concurrent mechanisms transparent to one another:
    // their rounds are separate messages that land one after the other in the same
    // table and go out in the SAME dispatch, exactly as separate rows used to.
    // What changes is that a round can no longer be torn across two flushes, and
    // that a round of `m²` cells costs one message instead of `m²` — which is the
    // term that made the step grow quadratically with body count.
    (@row $name:ident, $module:ident, $n_fatals:expr, [$($ty:ty),* $(,)?]) => {
        plain_shader!(@common $name, $module, $n_fatals, [$($ty),*]);

        impl ShaderSetup for $name {
	    fn storages(&self) -> &Vec<TypeId> { &self.storages }

            fn setup(
                &self,
                builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                batch: &[(
                    MessageInput,
                    Option<oneshot::Sender<Result<(), EvalError>>>,
                )],
            ) -> Result<u32, EvalError> {
                // Before anything is recorded: a refused kind binds nothing and
                // dispatches nothing, and `check` fails its messages.
                if self
                    .ledger
                    .claim(std::any::type_name::<Self>(), &self.writes, batch)
                    .is_err()
                {
                    return Ok(0);
                }
                let total: usize = batch
                    .iter()
                    .map(|(MessageInput { payload, .. }, _)| {
                        let MessagePayload::Rows { rows } = payload;
                        rows.len()
                    })
                    .sum();
                self.setup_dispatch(builder, total as u32)?;

                let mut incidence = self.incidence.write().map_err(Self::backend)?;

                let mut at = 0usize;
                for (MessageInput { kind: _, payload }, _) in batch.iter() {
                    let MessagePayload::Rows { rows } = payload;
                    for row in rows.iter() {
                        incidence[at] = $module::Incidence {
                            row: row.raw_index() as u32,
                        };
                        at += 1;
                    }
                }

                Ok(total as u32)
            }

            fn check(
                &self,
                batch: &mut Vec<(
                    MessageInput,
                    Option<oneshot::Sender<Result<(), EvalError>>>,
                )>,
            ) -> Vec<MessageInput> {
                let refused = self.ledger.close();
                let mut retired = Vec::with_capacity(batch.len());
                // A refused flush ran nothing of this kind (or ran it beside a
                // refused one): every message fails, and the fatal table holds a
                // previous flush's marks, so it is not read.
                if let Some(why) = refused {
                    for (input, respond_to) in batch.drain(..) {
                        if let Some(respond_to) = respond_to {
                            let _b = respond_to.send(Err(EvalError::Backend {
                                shader: std::any::type_name::<Self>(),
                                msg: why.clone(),
                            }));
                        }
                        retired.push(input);
                    }
                    return retired;
                }
                let fatals = self.fatals.read().unwrap();
                // A storage that grew after `setup` bound it was reallocated
                // between recording and submit, so the kernel ran on the old copy:
                // its inputs may be stale and its outputs did not reach the live
                // buffer. Fail the batch rather than report garbage as success.
                // Growth after the fence is reported too — harmless, but it
                // cannot be told apart from here.
                let grown = {
                    let bound = self.bound.lock().unwrap();
                    self.stale(&bound)
                };
                // Fatals stay per INVOCATION, so a message owns the span it filled
                // and its outcome is the outcome of any row inside that span. Only
                // the first `n_fatals` slots are the trace's; a slot past them is
                // padding the kernel never writes.
                let mut at = 0usize;
                for (input, respond_to) in batch.drain(..) {
                    let MessagePayload::Rows { rows } = &input.payload;
                    let len = rows.len();
                    let fired = fatals[at..at + len]
                        .iter()
                        .enumerate()
                        .find_map(|(row, f)| {
                            let set: Vec<usize> = f.fatal[..self.n_fatals]
                                .iter()
                                .enumerate()
                                .filter(|(_, x)| **x != 0.0)
                                .map(|(slot, _)| slot)
                                .collect();
                            (!set.is_empty()).then_some((row, set))
                        });
                    at += len;
                    if let Some(respond_to) = respond_to {
                        let _b = respond_to.send(match (grown, fired) {
                            (Some(t), _) => Err(EvalError::Backend {
                                shader: std::any::type_name::<Self>(),
                                msg: format!(
                                    "world storage {t:?} grew while this batch was in flight; \
                                     the kernel ran on the old buffer"
                                ),
                            }),
                            (None, Some((row, slots))) => Err(EvalError::Backend {
                                shader: std::any::type_name::<Self>(),
                                msg: format!(
                                    "fatal slot(s) {slots:?} set at row {row} of this message"
                                ),
                            }),
                            (None, None) => Ok(()),
                        });
                    }
                    retired.push(input);
                }
                retired
            }
        }
    };

}

macro_rules! define_shaders {
    // Entry: rolling list of all your shader entries.
    // $w:expr — the device/context reference you pass to `new()`.
    (
        $w:expr;
        $(
            @ $mode:ident
            $name:ident,
            $fname:ident,
            $eval:ident,
	    $n_fatals:expr,
            [ $($ty:ty),* ],
            $writes:expr
        );* $(;)?
    ) => {
        {
            // 1. Emit all plain_shader! definitions (local to this block).
            $(
                plain_shader!(@$mode $name, $fname, $n_fatals, [ $($ty),* ]);
            )*

            // 2. Build the runtime HashMap.
            let mut __shaders: HashMap<MessageKind, Box<dyn ShaderSetup>> = HashMap::new();
            let __ledger = Arc::new(Ledger::default());
            $(
                define_shaders!(@insert $mode, $name, $eval, __shaders, $w, $writes, __ledger);
            )*
            __shaders
        }
    };

    // -----------------------------------------------------------------
    // Internal helpers that dispatch based on @simple / @plain / @jacobian.
    // -----------------------------------------------------------------





    (@insert row, $name:ident, $eval:ident, $shaders:ident, $w:expr, $writes:expr, $ledger:ident) => {
        $shaders.insert(
            MessageKind::$eval,
            Box::new($name::new($w, $writes, $ledger.clone())),
        );
    };
}

pub(super) fn all_shaders(w: &Arc<World>) -> HashMap<MessageKind, Box<dyn ShaderSetup>> {
    define_shaders!(
        &w;   // <-- pass your context reference here once
        @row     SimpleSum, shader_simple_sum, EvalSimpleSum, fatal_width::<shader_simple_sum::Fatals>(), [[u32; 128], f32], writes!(2 => f32);
        @row     CriticallyDampedWarpedPlain, critically_damped_warped_plain, EvalCriticallyDampedWarpedPlain, fatal_counts::CRITICALLY_DAMPED_WARPED_PLAIN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]], writes!(4 => [Wrench<f32>; 2]);
        @row     CriticallyDampedWarpedJacobian, critically_damped_warped_jacobian, EvalCriticallyDampedWarpedJacobian, fatal_counts::CRITICALLY_DAMPED_WARPED_JACOBIAN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]], writes!(4 => [[Wrench<f32>; 24]; 2], 5 => [Wrench<f32>; 2]);
        @row     PerpendicularDamperWarpedPlain, perpendicular_damper_warped_plain, EvalPerpendicularDamperWarpedPlain, fatal_counts::PERPENDICULAR_DAMPER_WARPED_PLAIN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]], writes!(4 => [Wrench<f32>; 2]);
        @row     PerpendicularDamperWarpedJacobian, perpendicular_damper_warped_jacobian, EvalPerpendicularDamperWarpedJacobian, fatal_counts::PERPENDICULAR_DAMPER_WARPED_JACOBIAN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]], writes!(4 => [[Wrench<f32>; 24]; 2], 5 => [Wrench<f32>; 2]);
        @row     SimpleSpringDamperPlain, simple_spring_damper_plain, EvalSimpleSpringDamperPlain, fatal_counts::SIMPLE_SPRING_DAMPER_PLAIN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]], writes!(4 => [Wrench<f32>; 2]);
        @row     SimpleSpringDamperJacobian, simple_spring_damper_jacobian, EvalSimpleSpringDamperJacobian, fatal_counts::SIMPLE_SPRING_DAMPER_JACOBIAN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]], writes!(4 => [[Wrench<f32>; 24]; 2], 5 => [Wrench<f32>; 2]);
        @row     TorsionalDamperWarpedPlain, torsional_damper_warped_plain, EvalTorsionalDamperWarpedPlain, fatal_counts::TORSIONAL_DAMPER_WARPED_PLAIN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]], writes!(4 => [Wrench<f32>; 2]);
        @row     TorsionalDamperWarpedJacobian, torsional_damper_warped_jacobian, EvalTorsionalDamperWarpedJacobian, fatal_counts::TORSIONAL_DAMPER_WARPED_JACOBIAN, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]], writes!(4 => [[Wrench<f32>; 24]; 2], 5 => [Wrench<f32>; 2]);
        @row Pre, pre_shader, Pre, fatal_counts::PRE, [[u32; 128], f32, Twist<f32>, Motor<f32>], writes!(2 => Motor<f32>, 3 => Twist<f32>);
        // ROW-driven stages: binding 2 is ALWAYS the row storage, the rest follow.
        @row Gather, gather_shader, Gather, fatal_counts::GATHER, [[u32; 128], Wrench<f32>, [Wrench<f32>; 2]], writes!(0 => Wrench<f32>);
        @row BlockMatVec, block_matvec_shader, BlockMatVec, fatal_counts::BLOCK_MAT_VEC, [[u32; 128], [[f32; 6]; 6], Wrench<f32>, Twist<f32>], writes!(0 => Twist<f32>);
        @row Gemm, gemm_shader, Gemm, fatal_counts::GEMM, [[u32; 128], [[f32; 6]; 6]], writes!(0 => [[f32; 6]; 6]);
        @row AssembleBlock, assemble_shader, AssembleBlock, fatal_counts::ASSEMBLE_BLOCK, [[u32; 128], f32, [[f32; 6]; 6], [[Wrench<f32>; 24]; 2]], writes!(0 => [[f32; 6]; 6]);
        @row BlockReduce, block_reduce_shader, BlockReduce, fatal_counts::BLOCK_REDUCE, [[u32; 128], [[f32; 6]; 6], f32], writes!(0 => f32);
        @row BlockCopy, block_copy_shader, BlockCopy, fatal_counts::BLOCK_COPY, [[u32; 128], [[f32; 6]; 6]], writes!(0 => [[f32; 6]; 6]);
        @row BodyPostDiagonal, body_post_diagonal_shader, BodyPostDiagonal, fatal_counts::BODY_POST_DIAGONAL, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [f32; 3], f32], writes!(per 0, 3, 9; 6 => [[f32; 6]; 6], 7 => Wrench<f32>, 8 => Wrench<f32>);
        @row BodyPostFull, body_post_full_shader, BodyPostFull, fatal_counts::BODY_POST_FULL, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [[f32; 3]; 3], f32], writes!(per 0, 3, 9; 6 => [[f32; 6]; 6], 7 => Wrench<f32>, 8 => Wrench<f32>);
        @row BodyPostGatheredDiagonal, body_post_gathered_diagonal_shader, BodyPostGatheredDiagonal, fatal_counts::BODY_POST_GATHERED_DIAGONAL, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [f32; 3], f32, [Wrench<f32>; 2]], writes!(8 => [[f32; 6]; 6], 9 => Wrench<f32>, 10 => Wrench<f32>);
        @row BodyPostGatheredFull, body_post_gathered_full_shader, BodyPostGatheredFull, fatal_counts::BODY_POST_GATHERED_FULL, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [[f32; 3]; 3], f32, [Wrench<f32>; 2]], writes!(8 => [[f32; 6]; 6], 9 => Wrench<f32>, 10 => Wrench<f32>)
    )
}

#[cfg(test)]
mod tests {
    use super::super::row::ROW;
    use super::super::rows::fixed::{JointRow, bake_joints};
    use super::super::{
        MessageInput, MessageKind, MessagePayload, Pending, PendingByKind, Shaders,
    };
    use super::*;
    use aristotle::{Epoch, WorldId, WorldKey};
    use clifford::pga3::Dynamics;
    use joints::{AxialSpringDamper, JointEdge, SimpleSpringDamper};
    use peano::prelude::*;

    const DT: f32 = 1.0 / 60.0;
    const WARP: f32 = 1.0;
    /// Sentinel the fatal connection's value slot starts at, so the test can see
    /// that the kernel overwrote it rather than skipping the store.
    const UNTOUCHED: f32 = 123.0;

    /// One `SimpleSpringDamper` connection between two bodies at the identity
    /// pose, at rest, with anchor `b` on body B (anchor `a` is the origin). With
    /// `b` at the origin and no softening the anchors coincide, `√(d² + ε²)` is
    /// exactly zero, and the kernel's division guard trips.
    fn connection(
        world: &Arc<World>,
        b: [f32; 3],
        softening: f32,
    ) -> (JointEdge<f32>, JointRow<f32>) {
        let joint = AxialSpringDamper::builder(world.clone(), SimpleSpringDamper)
            .b(Vector3::from(b))
            .rest(0.5)
            .stiffness(10.0)
            .damping(1.0)
            .softening(softening)
            .build();
        let edge = JointEdge::new(WorldId::get(), WorldId::get(), joint);
        let vels = {
            let mut map = world.write::<Twist<f32>>();
            [map.add(Twist::zero()), map.add(Twist::zero())]
        };
        let poses = {
            let mut map = world.write::<Motor<f32>>();
            [map.add(Motor::identity()), map.add(Motor::identity())]
        };
        let row = JointRow {
            vels,
            poses,
            conn: edge.wrench_key(),
            block: edge.jacobian_key(),
            params: edge.params_keys(),
        };
        (edge, row)
    }

    /// Bake `rows` into one message of `kind`, returning it with the receiver
    /// its outcome is reported on.
    fn message(
        world: &Arc<World>,
        kind: MessageKind,
        rows: &[JointRow<f32>],
        dt: &WorldKey<f32>,
        warp: &WorldKey<f32>,
        jacobian: bool,
    ) -> (Pending, oneshot::Receiver<Result<(), EvalError>>) {
        let mut rounds = bake_joints(world, rows, dt, warp, jacobian);
        assert_eq!(rounds.len(), 1, "connection rows are one round");
        let (tx, rx) = oneshot::channel();
        let input = MessageInput {
            kind,
            payload: MessagePayload::Rows {
                rows: rounds.pop().unwrap(),
            },
        };
        ((input, Some(tx)), rx)
    }

    /// Drive one connection kernel into its division guard and check the
    /// mark-and-continue contract: three messages go out in ONE dispatch — two
    /// healthy rows, the degenerate row, one healthy row — and only the middle
    /// one fails, with `EvalError::Backend`. Its outputs are still written; its
    /// batch-mates on either side succeed and match the CPU force law.
    fn fatal_fails_only_its_own_message(kind: MessageKind, jacobian: bool) {
        let world = Arc::new(World::builder().usual::<f32>());
        let g = world.gpu();
        let shaders = Shaders {
            shaders: all_shaders(&world),
            cmd_allocator: g.cmd_allocator().clone(),
            gpu: g.clone(),
            world: world.clone(),
        };
        let (dt, warp) = {
            let mut map = world.write::<f32>();
            (map.add(DT), map.add(WARP))
        };

        let (h0, r0) = connection(&world, [1.0, 0.0, 0.0], 0.0);
        let (h1, r1) = connection(&world, [0.0, 2.0, 0.0], 0.0);
        let (bad, rb) = connection(&world, [0.0, 0.0, 0.0], 0.0);
        let (h2, r2) = connection(&world, [0.0, 0.0, -1.5], 0.0);
        bad.wrench_key().write(
            [Wrench::new(
                &Vector3::from([UNTOUCHED; 3]),
                &Vector3::from([UNTOUCHED; 3]),
            ); 2],
        );

        let (m0, rx0) = message(&world, kind, &[r0, r1], &dt, &warp, jacobian);
        let (mb, rxb) = message(&world, kind, &[rb], &dt, &warp, jacobian);
        let (m2, rx2) = message(&world, kind, &[r2], &dt, &warp, jacobian);
        let mut batches: PendingByKind = HashMap::new();
        batches.insert(kind, vec![m0, mb, m2]);
        shaders.dispatch(&mut batches).unwrap();

        let outcome = |mut rx: oneshot::Receiver<Result<(), EvalError>>| {
            rx.try_recv()
                .expect("sender dropped without an outcome")
                .expect("check reported no outcome")
        };
        let (o0, ob, o2) = (outcome(rx0), outcome(rxb), outcome(rx2));
        assert!(
            o0.is_ok(),
            "healthy message before the fatal one failed: {o0:?}"
        );
        assert!(
            o2.is_ok(),
            "healthy message after the fatal one failed: {o2:?}"
        );
        match ob {
            Err(EvalError::Backend { msg, .. }) => {
                assert!(
                    msg.contains("row 0"),
                    "fatal reported at the wrong row: {msg}"
                );
            }
            other => panic!("degenerate connection did not fail with Backend: {other:?}"),
        }

        // Mark-and-continue: the failed message's outputs were still stored.
        let written = bad.wrench_key().read();
        assert!(
            written
                .iter()
                .flat_map(|w| w.force().split().into_iter().chain(w.torque().split()))
                .all(|c| c != UNTOUCHED),
            "fatal invocation skipped its stores: {written:?}"
        );

        // The healthy connections computed the force law, not garbage.
        let epoch = Epoch::standalone(DT, WARP);
        let poses = vector![Motor::identity(), Motor::identity()];
        let vels = vector![Twist::zero(), Twist::zero()];
        for edge in [&h0, &h1, &h2] {
            let want = edge.eval::<f32>(&poses, &vels, &epoch);
            let got = edge.wrench_key().read();
            for end in 0..2 {
                let (w, g) = (want[end], got[end]);
                let pairs = w
                    .force()
                    .split()
                    .into_iter()
                    .zip(g.force().split())
                    .chain(w.torque().split().into_iter().zip(g.torque().split()));
                for (w, g) in pairs {
                    assert!(
                        g.is_finite() && (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                        "end {end}: cpu {w} vs gpu {g}"
                    );
                }
            }
            assert!(
                got[0].force().split().iter().any(|c| c.abs() > 1.0),
                "healthy connection produced no force: {got:?}"
            );
        }
    }

    #[test]
    fn plain_kernel_fatal_fails_only_its_own_message() {
        fatal_fails_only_its_own_message(MessageKind::EvalSimpleSpringDamperPlain, false);
    }

    #[test]
    fn jacobian_kernel_fatal_fails_only_its_own_message() {
        fatal_fails_only_its_own_message(MessageKind::EvalSimpleSpringDamperJacobian, true);
    }

    // ── One writer per slot (AR-0102) ────────────────────────────────────────

    /// Sentinel the output slots start at, so a test can see that nothing ran.
    const FRESH: f32 = -7.0;

    fn bare_shaders(world: &Arc<World>) -> Shaders {
        let g = world.gpu();
        Shaders {
            shaders: all_shaders(world),
            cmd_allocator: g.cmd_allocator().clone(),
            gpu: g.clone(),
            world: world.clone(),
        }
    }

    /// One message of `kind` over raw rows.
    fn raw_message(
        world: &Arc<World>,
        kind: MessageKind,
        rows: &[Row],
    ) -> (Pending, oneshot::Receiver<Result<(), EvalError>>) {
        let keys: Vec<WorldKey<Row>> = {
            let mut map = world.write::<Row>();
            rows.iter().map(|r| map.add(*r)).collect()
        };
        let (tx, rx) = oneshot::channel();
        let input = MessageInput {
            kind,
            payload: MessagePayload::Rows {
                rows: Arc::from(keys),
            },
        };
        ((input, Some(tx)), rx)
    }

    /// A `simple_sum` row: `out = a + b`.
    fn sum_row(a: &WorldKey<f32>, b: &WorldKey<f32>, out: &WorldKey<f32>) -> Row {
        let mut r: Row = [0; ROW];
        r[0] = a.raw_index() as u32;
        r[1] = b.raw_index() as u32;
        r[2] = out.raw_index() as u32;
        r
    }

    fn outcome(mut rx: oneshot::Receiver<Result<(), EvalError>>) -> Result<(), EvalError> {
        rx.try_recv()
            .expect("sender dropped without an outcome")
            .expect("check reported no outcome")
    }

    fn assert_refused(o: Result<(), EvalError>) {
        match o {
            Err(EvalError::Backend { msg, .. }) => assert!(
                msg.contains("second writer"),
                "refused for the wrong reason: {msg}"
            ),
            other => panic!("a colliding flush was not refused: {other:?}"),
        }
    }

    fn scalars(world: &Arc<World>, values: &[f32]) -> Vec<WorldKey<f32>> {
        let mut map = world.write::<f32>();
        values.iter().map(|v| map.add(*v)).collect()
    }

    /// Two messages of one kind write the same slot, a third writes its own:
    /// the whole flush is refused, every caller gets `EvalError::Backend`, and
    /// no output slot moved — not even the innocent one — because nothing ran.
    #[test]
    fn two_writers_of_one_slot_refuse_the_flush() {
        let world = Arc::new(World::builder().usual::<f32>());
        let shaders = bare_shaders(&world);
        let k = scalars(&world, &[1.0, 2.0, 3.0, 4.0, FRESH, FRESH]);
        let (shared, own) = (&k[4], &k[5]);
        let kind = MessageKind::EvalSimpleSum;
        let (m0, rx0) = raw_message(&world, kind, &[sum_row(&k[0], &k[1], shared)]);
        let (m1, rx1) = raw_message(&world, kind, &[sum_row(&k[2], &k[3], shared)]);
        let (m2, rx2) = raw_message(&world, kind, &[sum_row(&k[0], &k[3], own)]);
        let mut batches: PendingByKind = HashMap::new();
        batches.insert(kind, vec![m0, m1, m2]);
        shaders.dispatch(&mut batches).unwrap();

        for rx in [rx0, rx1, rx2] {
            assert_refused(outcome(rx));
        }
        assert_eq!(shared.read(), FRESH, "the contested slot was written");
        assert_eq!(own.read(), FRESH, "a refused flush still dispatched");
    }

    /// The collision inside ONE message: the same row twice in a round.
    #[test]
    fn a_row_repeated_within_a_round_is_refused() {
        let world = Arc::new(World::builder().usual::<f32>());
        let shaders = bare_shaders(&world);
        let k = scalars(&world, &[1.0, 2.0, FRESH]);
        let row = sum_row(&k[0], &k[1], &k[2]);
        let kind = MessageKind::EvalSimpleSum;
        let (m, rx) = raw_message(&world, kind, &[row, row]);
        let mut batches: PendingByKind = HashMap::new();
        batches.insert(kind, vec![m]);
        shaders.dispatch(&mut batches).unwrap();

        assert_refused(outcome(rx));
        assert_eq!(k[2].read(), FRESH);
    }

    /// Two KINDS write one slot in one flush — `simple_sum`'s `out` and a
    /// `BlockReduce` partial over the same scalar storage. Both callers are
    /// refused, and the slot holds at most one writer's value.
    #[test]
    fn two_writers_across_kinds_refuse_the_flush() {
        let world = Arc::new(World::builder().usual::<f32>());
        let shaders = bare_shaders(&world);
        let k = scalars(&world, &[1.0, 2.0, FRESH]);
        let (m0, rx0) = raw_message(
            &world,
            MessageKind::EvalSimpleSum,
            &[sum_row(&k[0], &k[1], &k[2])],
        );
        // An empty reduction: `out` ← 0.
        let mut reduce: Row = [0; ROW];
        reduce[0] = k[2].raw_index() as u32;
        let (m1, rx1) = raw_message(&world, MessageKind::BlockReduce, &[reduce]);
        let mut batches: PendingByKind = HashMap::new();
        batches.insert(MessageKind::EvalSimpleSum, vec![m0]);
        batches.insert(MessageKind::BlockReduce, vec![m1]);
        shaders.dispatch(&mut batches).unwrap();

        assert_refused(outcome(rx0));
        assert_refused(outcome(rx1));
        let v = k[2].read();
        assert!([FRESH, 3.0, 0.0].contains(&v), "slot holds {v}");
    }

    /// Negative case: the check is per FLUSH. The same slot written by one row
    /// in each of two consecutive flushes is two writers in sequence, not a
    /// collision — and a refused flush does not poison the next one.
    #[test]
    fn one_writer_per_flush_is_not_a_collision() {
        let world = Arc::new(World::builder().usual::<f32>());
        let shaders = bare_shaders(&world);
        let k = scalars(&world, &[1.0, 2.0, 5.0, FRESH]);
        let kind = MessageKind::EvalSimpleSum;

        // A refused flush first.
        let row = sum_row(&k[0], &k[1], &k[3]);
        let (m, rx) = raw_message(&world, kind, &[row, row]);
        let mut batches: PendingByKind = HashMap::new();
        batches.insert(kind, vec![m]);
        shaders.dispatch(&mut batches).unwrap();
        assert_refused(outcome(rx));

        for (a, want) in [(&k[0], 3.0), (&k[2], 7.0)] {
            let (m, rx) = raw_message(&world, kind, &[sum_row(a, &k[1], &k[3])]);
            batches.insert(kind, vec![m]);
            shaders.dispatch(&mut batches).unwrap();
            outcome(rx).unwrap();
            assert_eq!(k[3].read(), want);
        }
    }

    /// A kernel whose storages grew after its descriptor set was written reads
    /// and writes the GROWN buffers: `simple_sum` over scalar slots and rows
    /// allocated past the initial capacity, after one dispatch bound the
    /// original ones.
    #[tokio::test]
    async fn simple_sum_reaches_slots_past_initial_capacity() {
        use super::super::{Accelerator, row::Row};
        use futures::future::join_all;

        const INITIAL: usize = 16;
        let world = Arc::new(World::builder().capacity(INITIAL).usual::<f32>());
        let accel = Accelerator::<f32>::builder(world.clone()).build();

        // One dispatch on the initial buffers, so the set is bound before growth.
        let (a, b, out) = {
            let mut map = world.write::<f32>();
            (map.add(1.0), map.add(2.0), map.add(0.0))
        };
        accel.simple_sum(&a, &b, &out).await.unwrap();
        assert_eq!(out.read(), 3.0);
        let scalars = world.generation(TypeId::of::<f32>());
        let rows = world.generation(TypeId::of::<Row>());

        let n = INITIAL * 8;
        let sums: Vec<(WorldKey<f32>, WorldKey<f32>, WorldKey<f32>)> = {
            let mut map = world.write::<f32>();
            (0..n)
                .map(|i| (map.add(i as f32), map.add(0.5 * i as f32), map.add(-1.0)))
                .collect()
        };
        // Grow the row storage up front too, then free the slots: `simple_sum`
        // bakes its row inline, and a reallocation while an earlier batch is in
        // flight is the case `check` reports as an error rather than this one.
        let spare: Vec<WorldKey<Row>> = {
            let mut map = world.write::<Row>();
            (0..n + 1).map(|_| map.add([0; 128])).collect()
        };
        drop(spare);
        assert!(world.generation(TypeId::of::<f32>()) > scalars);
        assert!(world.generation(TypeId::of::<Row>()) > rows);
        assert!(sums.last().unwrap().2.raw_index() >= INITIAL * 4);

        let futs = sums.iter().map(|s| accel.simple_sum(&s.0, &s.1, &s.2));
        for r in join_all(futs).await {
            r.unwrap();
        }
        for (i, s) in sums.iter().enumerate() {
            assert_eq!(s.2.read(), 1.5 * i as f32, "slot {}", s.2.raw_index());
        }
        // The value written before growth was carried over.
        assert_eq!(out.read(), 3.0);
    }
}
