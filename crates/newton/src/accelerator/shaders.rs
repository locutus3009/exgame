// SPDX-License-Identifier: MIT

use super::{MessageInput, MessageKind, MessagePayload, Pending, PipelineBindPoint};
use crate::EvalError;
use aristotle::World;
use bytemuck::Pod;
use clifford::pga3::{Motor, Twist, Wrench};
use futures::channel::oneshot;
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    pipeline::{ComputePipeline, Pipeline},
};

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
        #[derive(Clone)]
        pub(super) struct $name {
	    storages: Vec<TypeId>,
	    n_fatals: usize,
            incidence: Subbuffer<[$module::Incidence]>,
            #[allow(dead_code)]
            fatals: Subbuffer<[$module::Fatals]>,
            set: Arc<DescriptorSet>,
            pipeline: Arc<ComputePipeline>,
        }

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
                        self.set.clone(),
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

            pub(super) fn new(w: &World) -> Self {
                let g = w.gpu();
                let shader_batch_size = g.gpu_in_flight();
                let ds_allocator = g.ds_allocator();
                let shader = $module::load(g.device().clone()).unwrap();
                let pipeline = g.build_pipeline(&shader, "main");
                let incidence = g.allocate_buffer::<$module::Incidence>(shader_batch_size);
                let fatals = g.allocate_buffer::<$module::Fatals>(shader_batch_size);
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
                $(
                    if has(binding) {
                        writes.push(WriteDescriptorSet::buffer(
                            binding,
                            w.read::<$ty>().get_map().clone(),
                        ));
			storages.push(TypeId::of::<$ty>());
                    }
                    binding += 1;
                )*
                let _b = binding;
                let set = DescriptorSet::new(ds_allocator.clone(), layout, writes, []).unwrap();
                $name {
		    storages,
		    n_fatals: $n_fatals,
                    incidence,
                    fatals,
                    set,
                    pipeline,
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
                let fatals = self.fatals.read().unwrap();
                let mut retired = Vec::with_capacity(batch.len());
                // Fatals stay per INVOCATION, so a message owns the span it filled
                // and its outcome is the outcome of any row inside that span.
                let mut at = 0usize;
                for (input, respond_to) in batch.drain(..) {
                    let MessagePayload::Rows { rows } = &input.payload;
                    let len = rows.len();
                    let fired = fatals[at..at + len]
                        .iter()
                        .find(|f| f.fatal[0..self.n_fatals].iter().any(|x| *x != 0.0))
                        .map(|f| f.fatal);
                    at += len;
                    if let Some(respond_to) = respond_to {
                        let _b = respond_to.send(match fired {
                            Some(f) => Err(EvalError::Backend {
                                shader: std::any::type_name::<Self>(),
                                msg: format!("Fatal error during execution: {f:?}"),
                            }),
                            None => Ok(()),
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
            [ $($ty:ty),* ]
        );* $(;)?
    ) => {
        {
            // 1. Emit all plain_shader! definitions (local to this block).
            $(
                plain_shader!(@$mode $name, $fname, $n_fatals, [ $($ty),* ]);
            )*

            // 2. Build the runtime HashMap.
            let mut __shaders: HashMap<MessageKind, Box<dyn ShaderSetup>> = HashMap::new();
            $(
                define_shaders!(@insert $mode, $name, $eval, __shaders, $w);
            )*
            __shaders
        }
    };

    // -----------------------------------------------------------------
    // Internal helpers that dispatch based on @simple / @plain / @jacobian.
    // -----------------------------------------------------------------





    (@insert row, $name:ident, $eval:ident, $shaders:ident, $w:expr) => {
        $shaders.insert(MessageKind::$eval, Box::new($name::new($w)));
    };
}

pub(super) fn all_shaders(w: &World) -> HashMap<MessageKind, Box<dyn ShaderSetup>> {
    define_shaders!(
        &w;   // <-- pass your context reference here once
        @row     SimpleSum, shader_simple_sum, EvalSimpleSum, 1, [[u32; 128], f32];
        @row     CriticallyDampedWarpedPlain, critically_damped_warped_plain, EvalCriticallyDampedWarpedPlain, 1, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]];
        @row     CriticallyDampedWarpedJacobian, critically_damped_warped_jacobian, EvalCriticallyDampedWarpedJacobian, 8, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]];
        @row     PerpendicularDamperWarpedPlain, perpendicular_damper_warped_plain, EvalPerpendicularDamperWarpedPlain, 2, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]];
        @row     PerpendicularDamperWarpedJacobian, perpendicular_damper_warped_jacobian, EvalPerpendicularDamperWarpedJacobian, 8, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]];
        @row     SimpleSpringDamperPlain, simple_spring_damper_plain, EvalSimpleSpringDamperPlain, 1, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]];
        @row     SimpleSpringDamperJacobian, simple_spring_damper_jacobian, EvalSimpleSpringDamperJacobian, 6, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]];
        @row     TorsionalDamperWarpedPlain, torsional_damper_warped_plain, EvalTorsionalDamperWarpedPlain, 0, [[u32; 128], f32, Twist<f32>, Motor<f32>, [Wrench<f32>; 2]];
        @row     TorsionalDamperWarpedJacobian, torsional_damper_warped_jacobian, EvalTorsionalDamperWarpedJacobian, 2, [[u32; 128], f32, Twist<f32>, Motor<f32>, [[Wrench<f32>; 24]; 2], [Wrench<f32>; 2]];
        @row Pre, pre_shader, Pre, 0, [[u32; 128], f32, Twist<f32>, Motor<f32>];
        // ROW-driven stages: binding 2 is ALWAYS the row storage, the rest follow.
        @row Gather, gather_shader, Gather, 0, [[u32; 128], Wrench<f32>, [Wrench<f32>; 2]];
        @row BlockMatVec, block_matvec_shader, BlockMatVec, 0, [[u32; 128], [[f32; 6]; 6], Wrench<f32>, Twist<f32>];
        @row Gemm, gemm_shader, Gemm, 0, [[u32; 128], [[f32; 6]; 6]];
        @row AssembleBlock, assemble_shader, AssembleBlock, 0, [[u32; 128], f32, [[f32; 6]; 6], [[Wrench<f32>; 24]; 2]];
        @row BlockReduce, block_reduce_shader, BlockReduce, 1, [[u32; 128], [[f32; 6]; 6], f32];
        @row BlockCopy, block_copy_shader, BlockCopy, 1, [[u32; 128], [[f32; 6]; 6]];
        @row BodyPostDiagonal, body_post_diagonal_shader, BodyPostDiagonal, 20, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [f32; 3], f32];
        @row BodyPostFull, body_post_full_shader, BodyPostFull, 20, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [[f32; 3]; 3], f32];
        @row BodyPostGatheredDiagonal, body_post_gathered_diagonal_shader, BodyPostGatheredDiagonal, 20, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [f32; 3], f32, [Wrench<f32>; 2]];
        @row BodyPostGatheredFull, body_post_gathered_full_shader, BodyPostGatheredFull, 20, [[u32; 128], Motor<f32>, Twist<f32>, Wrench<f32>, [[f32; 6]; 6], [[f32; 3]; 3], f32, [Wrench<f32>; 2]]
    )
}
