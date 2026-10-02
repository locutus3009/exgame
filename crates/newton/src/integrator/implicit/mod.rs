// SPDX-License-Identifier: MIT

use crate::Component;
use crate::integrator::explicit::{ExplicitEuler, LieEuler, SymplecticEuler};
use crate::{Accelerator, EvalError};
use aristotle::{Epoch, WorldId, WorldKey};
use bytemuck::Pod;
use clifford::{Lift, pga3::Wrench};
use indexmap::IndexMap;
use joints::JointEdge;
use peano::prelude::*;
use std::sync::Arc;

pub(crate) mod block;
mod bridge;
pub(crate) mod cache;
mod newton;

pub use newton::Newton;

// ============================================================================
// IMPLICIT INTEGRATOR — coupled-bodies step
// ============================================================================

/// Steps every coupled body of a `Mechanism` over `dt`. Receives the full
/// set of bodies and the matching pre-aggregated wrenches (world frame),
/// computed in the field-accumulate phase. Solves the bodies as a coupled
/// implicit step (`Newton`) or bridges a per-body explicit scheme
/// (`ExplicitEuler` / `SymplecticEuler` / `LieEuler`) over the same set.
///
/// A closed `enum` rather than a trait object: the scheme set is fixed, and
/// an inherent `async fn step_all` lets the mechanism `.await` the step
/// directly — no `Pin<Box<dyn Future>>` boxing at the dispatch boundary.
///
/// `bodies` and `wrenches` are length-aligned: `wrenches[id]` is the
/// accumulated wrench for `bodies[id]`. Order is determined by
/// `Mechanism::step` (currently slot-map iteration order). Variants that
/// need identity-based lookup build their own index from the map keys.
/// The coupled-step integrator. Each variant carries a clone of the ONE
/// `Accelerator` — the sole GPU access point, created EXPLICITLY by the app
/// (`Arc::new(Accelerator::builder().build())`) and cloned into every integrator
/// at construction. Different mechanisms with different schemes all share that
/// one accelerator by holding clones of the same `Arc`.
pub enum ImplicitIntegrator<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> {
    ExplicitEuler(Arc<Accelerator<T>>),
    SymplecticEuler(Arc<Accelerator<T>>),
    LieEuler(Arc<Accelerator<T>>),
    Newton(Newton<T>),
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> Clone for ImplicitIntegrator<T> {
    /// Cloning shares the same accelerator (Arc clone) — the point of the "one GPU"
    /// design. Clone a mechanism's integrator to give a sibling the same GPU.
    fn clone(&self) -> Self {
        match self {
            Self::ExplicitEuler(a) => Self::ExplicitEuler(a.clone()),
            Self::SymplecticEuler(a) => Self::SymplecticEuler(a.clone()),
            Self::LieEuler(a) => Self::LieEuler(a.clone()),
            Self::Newton(n) => Self::Newton(n.clone()),
        }
    }
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> std::fmt::Debug
    for ImplicitIntegrator<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExplicitEuler(_) => write!(f, "ExplicitEuler"),
            Self::SymplecticEuler(_) => write!(f, "SymplecticEuler"),
            Self::LieEuler(_) => write!(f, "LieEuler"),
            Self::Newton(n) => write!(f, "{n:?}"),
        }
    }
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> ImplicitIntegrator<T> {
    /// This integrator's accelerator. Clone it to build a sibling integrator
    /// (e.g. on `Mechanism::split`) sharing the same GPU point.
    pub fn accelerator(&self) -> Arc<Accelerator<T>> {
        match self {
            Self::ExplicitEuler(a) | Self::SymplecticEuler(a) | Self::LieEuler(a) => a.clone(),
            Self::Newton(n) => n.accelerator(),
        }
    }

    /// Step all `bodies` of a mechanism over `dt`. The world → body pullback
    /// (explicit schemes) and the coupled solve (`Newton`) live behind this
    /// one entry point; `Mechanism::step` just hands over one island's maps.
    /// The accelerator is the integrator's own (cloned at construction) — the
    /// mechanism no longer supplies it.
    pub(crate) async fn step_all<S: Ring>(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        joints: &IndexMap<WorldId, JointEdge<T>>,
        wrenches: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        cache: &mut cache::NewtonCache<T>,
        epoch: &Epoch<T>,
    ) -> Result<(), EvalError> {
        match self {
            Self::ExplicitEuler(a) => {
                bridge::explicit_step_all(a, &ExplicitEuler, bodies, joints, wrenches, cache, epoch)
                    .await
            }
            Self::SymplecticEuler(a) => {
                bridge::explicit_step_all(
                    a,
                    &SymplecticEuler,
                    bodies,
                    joints,
                    wrenches,
                    cache,
                    epoch,
                )
                .await
            }
            Self::LieEuler(a) => {
                bridge::explicit_step_all(a, &LieEuler, bodies, joints, wrenches, cache, epoch)
                    .await
            }
            // The explicit schemes ignore the cache: they solve nothing, so
            // there is nothing to persist.
            Self::Newton(newton) => {
                newton
                    .step_all(bodies, joints, wrenches, cache, epoch)
                    .await
            }
        }
    }
}
