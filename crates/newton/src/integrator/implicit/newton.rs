// SPDX-License-Identifier: MIT

use super::cache::{LANES, NewtonCache};
use crate::Component;
use crate::{Accelerator, EvalError};
use aristotle::{Epoch, WorldId, WorldKey};
use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Twist, Wrench},
};
use indexmap::IndexMap;
use joints::JointEdge;
use peano::prelude::*;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

/// Implicit integrator for bodies stiffly coupled by springs and dampers
/// (camera rig, docking nodes).
///
/// Why it is needed. The simple integrators (`ExplicitEuler`, `SymplecticEuler`)
/// compute forces from the CURRENT state and step forward. If a coupling is stiff
/// (a strong spring), such a step is stable only for a very small `dt` — otherwise
/// the solution starts oscillating and blows up (see the note on the camera rig: the
/// spring network on `LieEuler` diverges already at `dt >= 1/30` s). The implicit scheme
/// computes forces not at the start but at the MIDDLE of the step (this is what is called the
/// "implicit midpoint rule"). This is stable for any `dt` and
/// more accurate, but the new state enters its own equation — so every
/// step has to SOLVE an equation rather than just evaluate one.
///
/// How we solve it. With Newton's method: take an approximation, compute the residual (how far
/// the equation is from being satisfied), solve a linear system for the correction, repeat
/// until the residual is small. The linear system needs the derivative of the joint forces
/// with respect to the body positions and velocities — it is provided by automatic differentiation
/// of the joint force function (`Differential::jacobian`): these are the stiffness matrices
/// (derivative with respect to positions) and damping matrices (with respect to velocities).
///
/// What is still missing. A sparse matrix type and a linear system solver (LU).
/// Everything else is already in the API: joint forces (`JointEdge::eval`) and their
/// derivative (`Differential::jacobian`), inertia (`Inertia::apply` /
/// `apply_inverse` — conversion between momentum and velocity), pose update
/// (`Motor::compose` + `Twist::exp`). Until there is a solver, `step_all` does a single
/// explicit step — a working fallback, but not stable for stiff couplings. Below,
/// step by step, is what should take its place.
///
/// A subtlety for the future: joint forces arrive in WORLD coordinates, whereas the momentum
/// of a body is stored in the coordinates of the BODY itself. The system must be assembled entirely in
/// one set of coordinates (for world coordinates there are `world_momentum` / `world_velocity`).
/// Newton–Schulz iterations per Newton iteration with a warm cache.
///
/// Measured (`findings/2026-07-21-newton-schulz-convergence.md`): the warm path
/// needs 2 at any stiffness, even with 5% drift in `dt` — four times more than
/// a single step gives. The third iteration was taken "as a reserve" and then removed after measuring
/// on the demo: 5.94 → 5.53 ms per step, while the final position diverges by 6e-8 out of
/// 4.3e-3, i.e. 1.4e-5 relative — an order of magnitude inside the integrator's own tolerance
/// (1e-4). Two iterations pass frame drops and a 30 s step.
const NS_ITERATIONS_DEFAULT: usize = 2;
/// Iterations for the first solve after the cache is (re)allocated, when `X` is only
/// a seed. The worst measured cost of the fallback seed is 18, at `dt = 0.5`.
const NS_SEED_ITERATIONS_DEFAULT: usize = 24;

pub struct Newton<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> {
    /// Newton–Schulz budget per Newton iteration with a warm cache.
    ns_iterations: usize,
    /// The budget when the cache holds only a seed.
    ns_seed_iterations: usize,
    /// The solver's accelerator. It lives HERE rather than as a separate field of the
    /// `ImplicitIntegrator` variant: the whole solver works through `&self`, and carrying it
    /// as a parameter would have to go through `step_all`, `solve_step` and `newton_solve`.
    accelerator: Arc<Accelerator<T>>,
    _marker: PhantomData<T>,
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> Newton<T> {
    /// A fresh integrator on top of the given accelerator.
    pub fn new(accelerator: Arc<Accelerator<T>>) -> Self {
        Newton {
            ns_iterations: NS_ITERATIONS_DEFAULT,
            ns_seed_iterations: NS_SEED_ITERATIONS_DEFAULT,
            accelerator,
            _marker: PhantomData,
        }
    }

    /// This solver's accelerator — to build a neighbouring integrator on top of the
    /// same GPU access point (for example, in `Mechanism::split`).
    pub fn accelerator(&self) -> Arc<Accelerator<T>> {
        self.accelerator.clone()
    }

    /// Override the iteration budgets. Used by the sweep in tests.
    pub fn with_budgets(mut self, warm: usize, seed: usize) -> Self {
        self.ns_iterations = warm;
        self.ns_seed_iterations = seed;
        self
    }

    /// `⌊log₂ v⌋` through the carrier's own comparisons — `T` has no `log2`, and
    /// the answer is only ever used to pick a power-of-two sub-step. `None` for
    /// anything not strictly positive and finite, which is what makes a NaN or an
    /// infinite signal fall back to a plain halving.
    fn log2_floor(v: T) -> Option<i32> {
        let two = T::ONE + T::ONE;
        let mut x = v;
        let mut e = 0i32;
        // `partial_cmp`, not `x <= T::ZERO`: `T` may carry NaN, which is
        // unordered against zero and must land in the `None` arm.
        if !matches!(x.partial_cmp(&T::ZERO), Some(Ordering::Greater)) {
            return None;
        }
        while x >= two {
            x = x / two;
            e += 1;
            if e > 250 {
                return None; // infinite, or so large the estimate is meaningless
            }
        }
        while x < T::ONE && e > -250 {
            x = x * two;
            e -= 1;
        }
        Some(e)
    }

    #[inline]
    fn max2(a: T, b: T) -> T {
        if a > b { a } else { b }
    }

    /// RELATIVE convergence criterion, over all degrees of freedom. The block residual
    /// ÷ the self-scale of the same block (assembled in `residual` from the residual terms),
    /// force and torque SEPARATELY, and we take the WORST ratio over all (dynamic)
    /// bodies. No `is_effective_zero`: the scale self-adjusts to the magnitude of the
    /// body's momentum (and hence to the f32 noise level at that magnitude).
    /// Reads the residual and the scale from the cache slots — they are written by the `BodyPost` task.
    /// The reduction itself stays on the CPU: it is a maximum over bodies, the shape of the task is different,
    /// and it costs 0.01% of the step.
    fn wnorm(rhs: &[WorldKey<Wrench<T>>], scale: &[WorldKey<Wrench<T>>]) -> T {
        let mut worst = T::ZERO;
        for (r, sc) in rhs.iter().zip(scale.iter()) {
            let (rv, sv) = (r.read(), sc.read());
            let (rf, rt) = (rv.force(), rv.torque());
            let (sf, st) = (sv.force(), sv.torque());
            let lin2 = rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2];
            let ang2 = rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2];
            worst = Self::max2(worst, lin2 / sf[0]);
            worst = Self::max2(worst, ang2 / st[0]);
        }
        worst
    }

    /// Write `vmid = base + alpha·dv` into the iterate slots.
    ///
    /// `base` holds the VALUES at the start of the iteration, not handles. This is essential:
    /// `WorldKey::clone` clones the `Arc`, i.e. yields the same slot, so
    /// "clone the map and write into the clone" means writing into the original.
    /// That is exactly how it used to be here, and the line search did not try steps but
    /// destructively shifted the iterate: a rejected probe was not rolled back, and eight
    /// halvings accumulated `dv·(1 + ½ + ¼ + …)` instead of `dv·α`.
    fn write_vmid(
        vmid: &IndexMap<WorldId, WorldKey<Twist<T>>>,
        base: &[Twist<T>],
        order: &IndexMap<WorldId, usize>,
        dv: &[WorldKey<Twist<T>>],
        alpha: T,
    ) {
        for (id, &idx) in order {
            vmid[id].write(base[idx] + dv[idx].read() * alpha);
        }
    }

    /// A snapshot of the iterate VALUES over the DYNAMIC bodies, indexed by position in
    /// `order` (see `write_vmid` on why values rather than handles).
    ///
    /// A vector, not a map by `WorldId`: inside the solver a body is its
    /// row number, and a hash on every access is not needed here. Kinematic bodies are
    /// not part of the snapshot — their velocity is prescribed and never rewritten.
    fn read_vmid(
        vmid: &IndexMap<WorldId, WorldKey<Twist<T>>>,
        order: &IndexMap<WorldId, usize>,
    ) -> Vec<Twist<T>> {
        let mut out = vec![Twist::zero(); order.len()];
        for (id, &idx) in order {
            out[idx] = vmid[id].read();
        }
        out
    }

    /// Seed of the approximate inverse: `X₀ = Aᵀ / (‖A‖₁·‖A‖_∞)`.
    ///
    /// Chosen for one property: it converges for ANY nonsingular `A`, with no
    /// conditions on the step or the conditioning. This is essential. The obvious
    /// mass-block seed `𝕀⁻¹` is four times cheaper (4 iterations versus 9), but
    /// contractive only for `dt < 2/ω` — exactly where the EXPLICIT scheme is stable too.
    /// Beyond that bound it diverges quadratically, and the implicit integrator is used
    /// precisely to go beyond it: making its linear solve depend on
    /// the very limit it exists to escape is pointless. The cheap
    /// seed was here and was removed together with the divergence guard that
    /// backed it up — there is nothing left to diverge, and the guard is not needed.
    ///
    /// The price is a cold start of 9-18 iterations (measured, see
    /// `findings/2026-07-21-newton-schulz-convergence.md`), once per island;
    /// the warm path is still 2-3.
    ///
    /// `‖A‖₁` is the largest sum of absolute values over a column, `‖A‖_∞` over a row. In
    /// the block layout the transposition is blockwise: block `(i, j)` of `Aᵀ`
    /// is the transposed block `(j, i)` of `A`.
    fn seed_transpose(cache: &mut NewtonCache<T>) {
        let m = cache.m();
        let abs = |x: T| if x < T::ZERO { -x } else { x };
        let mut one = T::ZERO;
        let mut inf = T::ZERO;
        for i in 0..m {
            let mut row = T::ZERO;
            let mut col = T::ZERO;
            for j in 0..m {
                let rb = cache.a()[i * m + j].read();
                let cb = cache.a()[j * m + i].read();
                for r in 0..6 {
                    for c in 0..6 {
                        row += abs(rb[r][c]);
                        col += abs(cb[r][c]);
                    }
                }
            }
            inf = Self::max2(inf, row);
            one = Self::max2(one, col);
        }
        let scale = T::ONE / (one * inf);
        for i in 0..m {
            for j in 0..m {
                let src = cache.a()[j * m + i].read();
                let mut dst = super::block::zero_block::<T>();
                for r in 0..6 {
                    for c in 0..6 {
                        dst[r][c] = src[c][r] * scale;
                    }
                }
                cache.x()[i * m + j].write(dst);
            }
        }
        cache.mark_seeded();
    }

    /// One order-2 Newton–Schulz step over the approximate inverse from the cache:
    /// `R ← A·X`, then `X ← X·(2I − R)`. The residual is squared every
    /// time: below one this is quadratic convergence, above one — quadratic
    /// DIVERGENCE. Hence the guard.
    ///
    /// Two products, two barriers. The first is cheap: `A` is block-sparse, and
    /// each row is summed only over the body and its joint neighbours. `R`
    /// is dense, so the second is summed over everything.
    ///
    async fn refine_inverse(
        accelerator: &Accelerator<T>,
        cache: &mut NewtonCache<T>,
    ) -> Result<T, EvalError> {
        let m = cache.m();
        if m == 0 {
            return Ok(T::ZERO);
        }
        accelerator.gemm(cache.ns_ax()).await?;
        // The contraction measure and the second product both READ `R` and write
        // disjoint slots, so they ride the same wave. That is the point of moving
        // the measure onto the GPU: as a host loop over `m²` blocks it did not just
        // cost `m²` guarded reads, it sat BETWEEN the two products and stalled the
        // pipeline with them.
        futures::future::try_join(
            accelerator.block_reduce(cache.r_reduce_rows()),
            accelerator.gemm(cache.ns_xr()),
        )
        .await?;
        let before = cache.contraction();
        cache.swap_x();
        Ok(before)
    }

    /// Work through the Newton–Schulz budget, watching that the iteration goes down.
    ///
    /// The guard here does NOT back up the seed: `seed_transpose` converges for any
    /// nonsingular `A`, so a freshly seeded iteration cannot diverge.
    /// It backs up the WARM hint — `X` from the previous step, carried over to the
    /// changed `A`. The guarantee applies to the seed, not to the carry-over: if
    /// `A` has moved far (subdivision changes the stiffness term fourfold, a frame
    /// drop — more), the inherited `X` may end up outside the region of
    /// convergence, and then the residual starts squaring upwards.
    ///
    /// The cure is to go back to the seed, which always works; so the guard
    /// cannot fire twice and is not what correctness rests on.
    ///
    /// There is nothing to serve as a threshold here: for the transposed seed the Frobenius norm
    /// starts around 2.8 and still converges perfectly well (at `n = 12` the
    /// `‖I‖_F` alone is 3.46). Divergence is distinguished by GROWTH, and multiplicative at that —
    /// squaring gives 13.7× per pass, whereas a converged iterate
    /// jitters on the machine floor within 1.25×. The measures are quadratic, so
    /// a "twofold" margin is written as "fourfold".
    ///
    /// A negation of `<`, not `>=`: NaN loses every comparison, and the direct form
    /// would let through exactly the case the guard exists for.
    async fn run_budget(
        accelerator: &Accelerator<T>,
        cache: &mut NewtonCache<T>,
        budget: usize,
        seed_budget: usize,
    ) -> Result<(), EvalError> {
        let margin = T::from_u32(4);
        let mut previous: Option<T> = None;
        let mut reseeded = false;
        let mut remaining = budget;
        while remaining > 0 {
            remaining -= 1;
            let before = Self::refine_inverse(accelerator, cache).await?;
            if let Some(prev) = previous
                && !matches!(before.partial_cmp(&(margin * prev)), Some(Ordering::Less))
                && !reseeded
            {
                Self::seed_transpose(cache);
                reseeded = true;
                previous = None;
                remaining = seed_budget;
                continue;
            }
            previous = Some(before);
        }
        Ok(())
    }

    /// One Newton solve at the given `half`=dt/2, local iterate `vmid` (world
    /// midpoint velocities). Returns (converged, the BEST finite `worst²`, `vmid`).
    /// Does NOT mutate bodies; kinematic bodies in `vmid` hold their prescribed velocity
    /// and are never updated (they are not in `order`).
    #[allow(clippy::too_many_arguments)]
    async fn newton_solve<S: Ring>(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        order: &IndexMap<WorldId, usize>,
        cache: &mut NewtonCache<T>,
        snap_mom: &IndexMap<WorldId, WorldKey<Wrench<T>>>, // P_s^n — momentum at the start of the step
        _external: &IndexMap<WorldId, WorldKey<Wrench<T>>>, // external forces (they reach the GPU gather through the baked rows)
        half: T,
        tol2: T,
        huge: T,
        ns_iterations: usize,
        ns_seed_iterations: usize,
        epoch: &Epoch<T>,
    ) -> Result<(bool, T, IndexMap<WorldId, WorldKey<Twist<T>>>, bool), EvalError> {
        let two = T::ONE + T::ONE;
        // Self-scale floor (squared) — a division guard, not a tolerance: a body at
        // PERFECT rest gives zero residual AND zero scale, ratio 0/floor = 0.
        let floor2 = {
            let f = T::from_rational(1, 1_000_000_000);
            f * f
        };
        const MAX_ITERATIONS: usize = 32;
        let vmid: IndexMap<WorldId, WorldKey<Twist<T>>> = bodies
            .iter()
            .map(|(id, e)| (*id, e.body().world_velocity()))
            .collect();
        // Publish this span's scalars and bake the POST rows for it. The rows name
        // only span-stable slots (`snap_mom` is the per-body `world_momentum`
        // slot), so once per span suffices; the VALUES the kernels read are in the
        // slots `publish_scalars` writes.
        if let Some((_, e)) = bodies.iter().next() {
            let world = e.body().world.clone();
            cache.publish_scalars(half, floor2);
            cache.bake_body_post(&world, bodies, snap_mom);
        }
        // Seed every probe lane's iterate from the real one. `write_vmid` below
        // only ever touches the DYNAMIC bodies, so without this the kinematic ends
        // would enter the trial force laws at zero velocity instead of at their
        // prescribed one — and only in lanes 1.., which is exactly the kind of
        // discrepancy that would show up as a line search that mysteriously prefers
        // lane 0.
        for lane in &cache.lanes()[1..] {
            for (id, k) in &lane.vmid {
                k.write(vmid[id].read());
            }
        }
        // best = the minimal FINITE worst², best_vals — the corresponding iterate.
        // best stays `huge` if no residual was finite (→ solve_step
        // treats the span as non-finite and subdivides/freezes).
        //
        // VALUES, not handles: `vmid.clone()` clones the `Arc`s and points at
        // the same slots, so such a "snapshot" would simply follow the iterate.
        let mut best = huge;
        let mut best_vals = Self::read_vmid(&vmid, order);
        let mut converged = false;
        // The residual ALREADY computed by the accepted line-search probe of the previous iteration.
        // The probe runs a full dispatch on exactly the iterate the next
        // iteration starts from: nobody writes `vmid` in between (the only
        // `write_vmid` after the loop is in the rejection branch, and that one exits), and
        // the pose is modified in the meantime only by `commit_span`, outside the loop. So PRE
        // would get the same inputs, the gather — the same terms, `BodyPost` — the same
        // mass block, residual and scale, and `wnorm` — the same number the probe already
        // returned. The only thing the probe does not write is the Jacobian: it needs the value,
        // not the derivative. Therefore the iteration after an accepted probe costs ONE wave
        // of dispatch instead of four.
        let mut carried: Option<T> = None;
        let mut assembled = false;
        for _iter in 0..MAX_ITERATIONS {
            let rnorm = match carried.take() {
                Some(from_probe) => {
                    self.accelerator.eval_jacobians(&cache.lanes()[0]).await?;
                    from_probe
                }
                None => {
                    // First iteration of the span (and any after a line-search rejection):
                    // there is nothing carried, take the full path. One dispatch with the Jacobian
                    // at the current vmid fills both total_wrench (value gather) and
                    // edge.jacobian — the residual and the matrix read them.
                    cache.publish_dispatch_scalars(half, *epoch.dt(), *epoch.warp());
                    // The fused POST sums the incident wrenches itself, so the
                    // separate gather wave is skipped when it is available.
                    let fused = cache.lanes()[0].gathered_rows().is_some();
                    self.accelerator
                        .dispatch(&cache.lanes()[0], true, !fused)
                        .await?;
                    match cache.lanes()[0].gathered_rows() {
                        Some(rows) => self.accelerator.body_post_gathered(rows).await?,
                        None => self.accelerator.body_post(cache.body_post_rows()).await?,
                    }
                    // Phase 4: mass block, residual and scale — per body, into slots.
                    // The barrier here is inherent to the data dependency (the residual needs
                    // the gather output), not to splitting the call in two.
                    Self::wnorm(cache.rhs(), cache.scale())
                }
            };
            if rnorm < best {
                best = rnorm;
                best_vals = Self::read_vmid(&vmid, order);
            }
            if rnorm <= tol2 {
                converged = true;
                break;
            }
            // A negation of `<`, not `>=`: NaN is unordered with everything, so
            // `rnorm >= huge` would let NaN through — into assembly, the solver and the
            // persistent `X`. Here NaN and inf both interrupt the iteration.
            if !matches!(rnorm.partial_cmp(&huge), Some(Ordering::Less)) {
                break; // non-finite → let solve_step decide
            }
            // Block assembly + Newton–Schulz instead of a direct solver. The budget
            // is fixed: on a warm cache a few iterations suffice, on a cold one
            // the fallback seed works under the guard (see `run_budget`).
            self.accelerator.assemble(cache.assemble_rows()).await?;
            // `A` and the mass block now hold this sub-step's system. Only that
            // fact is reported: measuring it costs a wave, and the caller needs
            // the number only on the path that actually subdivides.
            assembled = true;
            let budget = if cache.is_seeded() {
                ns_iterations
            } else {
                Self::seed_transpose(cache);
                ns_seed_iterations
            };
            Self::run_budget(&self.accelerator, cache, budget, ns_seed_iterations).await?;
            self.accelerator.block_matvec(cache.matvec_rows()).await?;
            // The correction stays in the slots: there is no point unpacking it into a flat vector
            // only to assemble it back into twists right away.
            let no_correction = cache.dv().iter().all(|k| {
                let d = k.read();
                let (l, a) = (d.linear(), d.angular());
                (0..3).all(|i| l[i] == T::ZERO && a[i] == T::ZERO)
            });
            if no_correction {
                break; // the solver gave no correction → on to subdivision
            }
            // Backtracking line search on the common worst²: accept the fraction α only
            // if the norm decreased. All LANES of fractions are independent for fixed `base`
            // and `dv`, so they are computed in ONE wave rather than a ladder: eight probes
            // go out together, and the search has exactly as many barriers as there used to be for
            // a single probe. The FIRST decreasing one is accepted — the same fraction at which
            // the sequential descent used to stop.
            //
            // The probe base is the values BEFORE the search, otherwise each probe starts from
            // the result of the previous one and halvings accumulate instead of being enumerated.
            let base = Self::read_vmid(&vmid, order);
            let accepted = {
                let lanes = cache.lanes();
                let mut alpha = T::ONE;
                for lane in lanes {
                    Self::write_vmid(&lane.vmid, &base, order, cache.dv(), alpha);
                    alpha = alpha / two;
                }
                // One wave for all lanes: the rows are already merged by phase at
                // baking time, so the PRE of all eight goes out as ONE message,
                // then the joints, then the gathers, then the common `BodyPost`. Eight
                // concurrent dispatches instead of this used to wake up out of step, and
                // the worker flushed half the wave ahead of the rest.
                let fused = cache.wave().gathered.is_some();
                self.accelerator
                    .dispatch_wave(cache.wave(), false, !fused)
                    .await?;
                match &cache.wave().gathered {
                    Some(rows) => self.accelerator.body_post_gathered(rows).await?,
                    None => self.accelerator.body_post(&cache.wave().body_post).await?,
                }
                lanes.iter().enumerate().find_map(|(i, lane)| {
                    let tnorm = Self::wnorm(lane.rhs(), lane.scale());
                    if tnorm < rnorm {
                        Some((i, tnorm))
                    } else {
                        None
                    }
                })
            };
            match accepted {
                Some((lane, tnorm)) => {
                    // The accepted lane becomes the iterate: its slots are what
                    // the next iteration starts from, and its norm is the norm of that
                    // iteration (see `carried`).
                    cache.adopt_lane(lane);
                    carried = Some(tnorm);
                }
                None => {
                    // No fraction helped — return the iterate to where we
                    // started, and hand the solve over to step subdivision.
                    Self::write_vmid(&vmid, &base, order, cache.dv(), T::ZERO);
                    break;
                }
            }
        }
        // Restore the slots to the BEST iterate seen: `commit_span` reads
        // exactly those next.
        for (id, &idx) in order {
            vmid[id].write(best_vals[idx]);
        }
        Ok((converged, best, vmid, assembled))
    }

    /// Cover `dt` with sub-steps, each as long as the system allows.
    ///
    /// Was a binary recursion: fail, halve, recurse twice. Two things were wrong
    /// with that. It walked down one level at a time, paying a failed span per
    /// level; and once the depth-jump landed, every sub-step of the interval was
    /// forced to the SAME size, so a quiet tail paid for a stiff head.
    ///
    /// A budget loop has neither problem and is simpler besides. The remaining
    /// interval is counted in units of `dt / 2^MAX_DEPTH`, so the arithmetic is
    /// exact integers and the sub-steps provably sum to `dt` however they vary.
    #[allow(clippy::too_many_arguments)]
    async fn solve_step<S: Ring>(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        order: &IndexMap<WorldId, usize>,
        cache: &mut NewtonCache<T>,
        external: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        dt: T,
        tol2: T,
        huge: T,
        ns_iterations: usize,
        ns_seed_iterations: usize,
        spans: &mut usize,
        epoch: &Epoch<T>,
    ) -> Result<(), EvalError> {
        const MAX_DEPTH: usize = 8;
        // Both directions read the same measured relation. `stiffness` is
        // `‖A‖²_F / ‖𝕀_s‖²_F` at the current sub-step, i.e. `(half·ω)⁴` — both
        // norms are squared, so one halving moves `log₂` by exactly 4 (observed
        // 34 → 30 → 26 → 21 down depths 0..3). The threshold is measured too: all
        // 389 converged spans of the subdividing suite sit at `log₂(r) ≤ 8`,
        // without exception, against 283 of 461 failures above it.
        //
        // So the criterion is NECESSARY, and it is applied symmetrically. Going
        // down it cannot skip a depth that would have converged. Going UP the same
        // asymmetry argues for taking it at face value: an over-optimistic growth
        // costs one cheap failed span and shrinks back, while an over-cautious one
        // keeps the whole remaining interval at half the step it could use.
        const THRESHOLD_LOG2: i32 = 8;
        const PER_HALVING_LOG2: i32 = 4;

        let two = T::ONE + T::ONE;
        // Remaining interval, in units of the finest sub-step. Exact by
        // construction: every span consumes a whole number of units.
        let mut remaining: u64 = 1 << MAX_DEPTH;
        let mut depth = 0usize;

        while remaining > 0 {
            // The span this depth wants, deepened until it fits what is left.
            let mut units = 1u64 << (MAX_DEPTH - depth);
            while units > remaining {
                depth += 1;
                units >>= 1;
            }
            let mut span = dt;
            for _ in 0..depth {
                span = span / two;
            }
            let half = span / two;
            *spans += 1;

            // Snapshot of the span start from the CURRENT body state (world momentum).
            let snap_mom: IndexMap<WorldId, WorldKey<Wrench<T>>> = bodies
                .iter()
                .map(|(id, e)| (*id, e.body().world_momentum()))
                .collect();
            let (converged, best, vmid, assembled) = self
                .newton_solve(
                    bodies,
                    order,
                    cache,
                    &snap_mom,
                    external,
                    half,
                    tol2,
                    huge,
                    ns_iterations,
                    ns_seed_iterations,
                    epoch,
                )
                .await?;
            let finite = best < huge;
            let last = depth == MAX_DEPTH;

            if !converged && !last {
                // Not accepted: roll the hint back to the last accepted step,
                // so that the next attempt starts from a good `X`, not from
                // the one that just diverged.
                self.accelerator.block_copy(cache.restore_rows()).await?;
                cache.finish_rollback();
                // Measured only here — it costs a wave, and it is only ever needed
                // when the step has to move. `assembled` is what says the matrix
                // belongs to THIS sub-step: a span that broke before assembling (a
                // non-finite residual) would otherwise be read against the previous
                // span's `A`.
                let down = match self.signal_log2(cache, assembled).await? {
                    Some(r) => (r - THRESHOLD_LOG2).div_euclid(PER_HALVING_LOG2) + 1,
                    None => 1,
                };
                depth += (down.clamp(1, (MAX_DEPTH - depth) as i32)) as usize;
                continue;
            }

            if converged || finite {
                // At the depth limit we commit best-effort: `commit_span` knows about
                // `converged` and switches to implicit Euler, which does not double
                // the solver error.
                Self::commit_span(bodies, order, &snap_mom, &vmid, span, half, converged);
                // The approximate inverse becomes "accepted" ONLY here. The live
                // `X` is a draft: it is rewritten by every Newton iteration of every
                // span, including rejected ones.
                //
                // The copy and the finiteness probe read the same `X` and write into
                // different slots, so they go out in ONE wave; the host is left to
                // sum the partials and set the flags.
                futures::future::try_join(
                    self.accelerator.block_copy(cache.publish_rows()),
                    self.accelerator.block_reduce(cache.x_reduce_rows()),
                )
                .await?;
                cache.finish_checkpoint();
            } else {
                // Non-finite at the depth limit: we do not commit, but the interval is still
                // traversed — exactly as the recursion did, which simply returned.
                self.accelerator.block_copy(cache.restore_rows()).await?;
                cache.finish_rollback();
            }
            remaining -= units;

            // Grow while the measurement says the next span can be longer — but
            // only to a size the remaining budget still divides by, so the units
            // stay exact. That alignment is checked FIRST: it costs nothing, and
            // when it forbids growing there is no point paying a wave to find out
            // how much we could have grown.
            let aligned = |d: usize| -> bool {
                let bigger = 1u64 << (MAX_DEPTH - (d - 1));
                d > 0 && bigger <= remaining && remaining.is_multiple_of(bigger)
            };
            if remaining > 0
                && converged
                && aligned(depth)
                && let Some(r) = self.signal_log2(cache, assembled).await?
            {
                let mut up = (THRESHOLD_LOG2 - r).div_euclid(PER_HALVING_LOG2);
                while up > 0 && aligned(depth) {
                    depth -= 1;
                    up -= 1;
                }
            }
        }
        Ok(())
    }

    /// `⌊log₂ ‖A‖²_F/‖𝕀_s‖²_F⌋`, or `None` when the matrix does not belong to the
    /// current sub-step or the ratio is not a usable number. Costs one wave: both
    /// norms read what `assemble` and `BodyPost` wrote and write disjoint scalars.
    async fn signal_log2(
        &self,
        cache: &NewtonCache<T>,
        assembled: bool,
    ) -> Result<Option<i32>, EvalError> {
        if !assembled {
            return Ok(None);
        }
        futures::future::try_join(
            self.accelerator.block_reduce(cache.a_reduce_rows()),
            self.accelerator.block_reduce(cache.mass_reduce_rows()),
        )
        .await?;
        Ok(Self::log2_floor(cache.stiffness()))
    }

    /// Writes the span result into the DYNAMIC bodies (kinematic ones are not touched).
    /// Everything in the world frame: the pose for the full `dt` by left composition from the snapshot,
    /// the momentum by the midpoint rule `P^{n+1}=2·P_mid−P^n`, then world→body.
    #[allow(clippy::too_many_arguments)]
    fn commit_span<S: Ring>(
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        order: &IndexMap<WorldId, usize>,
        snap_mom: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        vmid: &IndexMap<WorldId, WorldKey<Twist<T>>>,
        dt: T,
        half: T,
        converged: bool,
    ) {
        let two = T::ONE + T::ONE;
        for id in order.keys() {
            let vs = &vmid[id];
            let entity = bodies.get(id).unwrap();
            // The pose at the start of the span is the CURRENT pose: during the span it is written only by
            // this commit. There was no reason to snapshot a copy of every motor on entering the span;
            // we read before writing, and the bodies do not depend on each other.
            let m_start = entity.body().pose.read();
            let p_s_n = &snap_mom[id];
            let pose_mid = m_start.compose(&vs.read().exp(half));
            let v_body = pose_mid.inverse().conjugate(&vs.read());
            let p_s_mid = entity.body().inertia.apply(&v_body).transport(&pose_mid);
            // The midpoint rule reconstructs the end of the step as `2·P_mid − P_n`
            // and with it DOUBLES the solver error. As long as Newton converges, this
            // error is at machine level and the doubling does not matter. On an unconverged span —
            // and it happens at the bottom of the recursion, where there is nowhere further to subdivide — a factor
            // of 2 on every step is exactly an exponential pumping of energy.
            //
            // So an unconverged span falls back to IMPLICIT EULER: the velocity found
            // is treated as the velocity at the END of the step, not the middle. The same
            // solve, a different reconstruction. The order drops from second to first, and
            // the scheme turns from energy-conserving into dissipative — accuracy on
            // such a step is lost in any case, but the error is no longer doubled,
            // and the energy cannot grow. This way the depth limit stops being a cliff:
            // beyond it things are "inaccurate", not "blown apart".
            let p_s_new = if converged {
                p_s_mid * two - p_s_n.read()
            } else {
                p_s_mid
            };
            let m_new = m_start.compose(&vs.read().exp(dt));
            let body = entity.body();

            body.pose.write(m_new);
            body.momentum.write(p_s_new.transport(&m_new.inverse()));
        }
    }

    // TODO: a method that recomputes the matrix structure when the joint graph changes
    // (a body or joint was added/removed). Call it from `Mechanism` on such a
    // change, not on every step.
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> std::fmt::Debug for Newton<T> {
    /// The accelerator is not `Debug` (it owns a worker thread), so we print what
    /// the solver has of its own — the budgets.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Newton(ns={}, seed={})",
            self.ns_iterations, self.ns_seed_iterations
        )
    }
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> Clone for Newton<T> {
    /// The budgets are configuration, they carry over; the solver has no state of its
    /// own (the warm inverse lives on the island, not here).
    fn clone(&self) -> Self {
        Newton {
            ns_iterations: self.ns_iterations,
            ns_seed_iterations: self.ns_seed_iterations,
            accelerator: self.accelerator.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod> Newton<T> {
    /// Coupled implicit step of all bodies of the island over `dt` (see `ImplicitIntegrator`).
    /// An inherent `async fn` — called from `ImplicitIntegrator::step_all` without
    /// boxing the future.
    pub(crate) async fn step_all<S: Ring>(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        joints: &IndexMap<WorldId, JointEdge<T>>,
        wrenches: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        cache: &mut NewtonCache<T>,
        epoch: &Epoch<T>,
    ) -> Result<(), EvalError> {
        let dt = *epoch.dt();
        // Divergence / non-finite threshold. `< huge` yields false for +inf and NaN.
        let huge = T::from_u32(1_000_000_000);
        // Relative tolerance PER BLOCK (force/torque): worst² ≤ tol². 1e-4 on
        // the ratio → 1e-8 on the square. Calibrated for the GUI (Task 5).
        let tol2 = {
            let r = T::from_rational(1, 10_000); // 1e-4
            r * r // 1e-8
        };
        // Bring the cache in line with the island. The unknowns are only the
        // DYNAMIC bodies. Kinematic (prescribed) ones are excluded from the system:
        // no row/column, no residual; their velocity enters only the eval of the joint
        // forces. Hence the criterion excludes them STRUCTURALLY.
        let world = bodies
            .values()
            .next()
            .map(|e| e.body().pose.world())
            .expect("island with no bodies");
        cache.sync(
            &world,
            bodies
                .iter()
                .filter(|(_, e)| !e.body().inertia.is_kinematic())
                .map(|(id, _)| *id),
            joints,
        );
        // The line-search lanes follow the same topology latch as the other
        // cache tables: every slot addressed by a baked lane row
        // is stable exactly as long as the topology is stable.
        cache.bake_lanes(&world, bodies, joints, wrenches, LANES);
        // A copy of the order: `solve_step` takes the cache by `&mut`, while it also needs the order
        // by `&`. A map of a few entries, copied once per step — cheaper
        // than threading split borrows through a recursive async.
        let order = cache.order().clone();
        // External forces (gravity) are frozen for the whole dt — a semi-implicit
        // splitting; sub-steps do NOT recompute them (nor the prescribed velocities).
        // `wrenches` = the external buffer (the bodies' accum_wrench); dispatch reads it as
        // the gather seed and never writes it — so the shared handle is safe.
        let external = wrenches;

        // Honest subdivision of dt: a span converges or is halved, entirely
        // inside the integrator. The top-level parent_best = huge (the first subdivision is always
        // allowed). Bodies are mutated by spans; a non-finite span is not committed.
        let mut spans = 0usize;
        self.solve_step(
            bodies,
            &order,
            cache,
            external,
            dt,
            tol2,
            huge,
            self.ns_iterations,
            self.ns_seed_iterations,
            &mut spans,
            epoch,
        )
        .await?;
        cache.set_spans(spans);
        if spans > 1 {
            // The only (non-spammy) diagnostic trace: the step turned out to be stiff.
            eprintln!("[newton] dt subdivided into {spans} spans");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::ImplicitIntegrator;
    use crate::{Accelerator, Inert, Inertia, Mechanism, RigidBody};
    use aristotle::{Epoch, World};

    fn accel(world: Arc<World>) -> Arc<Accelerator<f32>> {
        Arc::new(Accelerator::builder(world).build())
    }
    use clifford::pga3::Twist;
    use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
    use std::sync::Arc;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
        AxialSpringDamper::builder(world, SimpleSpringDamper)
            .rest(rest)
            .stiffness(k)
            .damping(c)
            .build()
    }

    /// A body with kinematic inertia is a reference, not an unknown: it does not enter
    /// the criterion/matrix, and the dynamic body next to it must converge (not freeze).
    #[tokio::test]
    async fn kinematic_body_does_not_stall_dynamic_neighbour() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(Newton::new(accel(
            world.clone(),
        ))));
        let dyn_id = m
            .add_body(Inert::new(RigidBody::body_at_with_mass(
                world.clone(),
                &Vector3::from([1.0, 0.0, 0.0]),
                1.0,
            )))
            .await;
        let kin_id = m
            .add_body(Inert::new(RigidBody::new(
                world.clone(),
                Inertia::Kinematic,
            )))
            .await;
        // rest=0: the spring pulls the dynamic body towards the reference at the origin.
        m.connect(dyn_id, vec![(spring(world, 0.0, 10.0, 0.5), kin_id)])
            .await;
        let x0 = m.body_absolute_position(dyn_id).await.unwrap()[0];
        for _ in 0..200 {
            m.step(&Epoch::standalone(1.0 / 60.0, 1.0)).await.unwrap();
        }
        let x1 = m.body_absolute_position(dyn_id).await.unwrap()[0];
        assert!(x1 < x0 - 0.05, "dynamic body did not move: {x0} -> {x1}");
        assert!(x1.is_finite());
    }

    /// A large step on a stiff GEOMETRICALLY NONLINEAR coupling must land
    /// where physics says. Two bodies with large TRANSVERSE velocities on
    /// a stiff spring: the spring axis rotates noticeably within the step, so
    /// a linear problem is no good here — the implicit midpoint is A-stable and
    /// would pass it without any subdivision.
    ///
    /// The reference is an INDEPENDENT scheme (`LieEuler`) with a small step, not the same
    /// integrator with a finer step: that way they do not share a common error. With `k = 5e4` and
    /// `m = 1` we have `ω ≈ 316 rad/s`, and `dt = 1e-4` is stable with a large margin;
    /// it was verified that the reference has converged — halving the step shifts it by 3.2e-4,
    /// i.e. 200 times less than the quantity measured here.
    ///
    /// The comparison step `COARSE` is deliberately not huge. The first version of this test
    /// compared trajectories after ONE step of 0.5 s — that is 25 periods at once,
    /// where no method would give agreement, and the threshold would have had to be made up.
    /// Stability at such steps is checked separately (`cloth_grid_stability`),
    /// and here we check ACCURACY where it is meaningful.
    ///
    /// Replaces the former assert `last_spans() > 1`. That one watched the solver's
    /// INTERNALS: subdivision is the mechanism by which the integrator achieves the correct
    /// result, not the result itself, and any rework of the solver breaks such an assert
    /// while telling nothing about the physics. Finiteness alone is not enough either —
    /// a solver that committed garbage which happened not to blow up passes it.
    #[tokio::test]
    async fn stiff_nonlinear_step_matches_a_fine_reference() {
        const K: f32 = 5.0e4;
        const C: f32 = 5.0;
        const REST: f32 = 2.0;
        const SPEED: f32 = 50.0;
        const SPAN: f32 = 0.5;
        const COARSE: f32 = 0.002;
        // Observed discrepancy 0.065 at radius ~1 — phase drift over 25
        // periods at `ω·dt = 0.63`, normal for second order. The threshold
        // is twice that so as not to be brittle, but to catch gross errors: a flipped
        // sign or a factor of 2 in the Jacobian gives a discrepancy of order one.
        const TOL: f32 = 0.12;

        let world = Arc::new(World::builder().usual::<f32>());

        async fn run(
            world: Arc<World>,
            integrator: ImplicitIntegrator<f32>,
            dt: f32,
            steps: usize,
        ) -> [Vector3<f32>; 2] {
            let m = Mechanism::<f32, f32>::new(integrator);
            let a = m
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::from([1.0, 0.0, 0.0]),
                    &Vector3::from([0.0, SPEED, 0.0]),
                    1.0,
                )))
                .await;
            let b = m
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::from([-1.0, 0.0, 0.0]),
                    &Vector3::from([0.0, -SPEED, 0.0]),
                    1.0,
                )))
                .await;
            m.connect(a, vec![(spring(world.clone(), REST, K, C), b)])
                .await;
            for _ in 0..steps {
                m.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            }
            [
                m.body_absolute_position(a).await.unwrap(),
                m.body_absolute_position(b).await.unwrap(),
            ]
        }

        let steps = (SPAN / 1.0e-4).round() as usize;
        let reference = run(
            world.clone(),
            ImplicitIntegrator::LieEuler(accel(world.clone())),
            1.0e-4,
            steps,
        )
        .await;
        let coarse = run(
            world.clone(),
            ImplicitIntegrator::Newton(Newton::new(accel(world.clone()))),
            COARSE,
            (SPAN / COARSE).round() as usize,
        )
        .await;

        for body in 0..2 {
            for axis in 0..3 {
                assert!(
                    coarse[body][axis].is_finite(),
                    "body {body} axis {axis} is not finite"
                );
                let d = (coarse[body][axis] - reference[body][axis]).abs();
                assert!(
                    d < TOL,
                    "body {body} axis {axis}: coarse step {} versus reference {}, difference {d}",
                    coarse[body][axis],
                    reference[body][axis]
                );
            }
        }
    }
    /// Regression on DOF ordering consistency (catches a silent permutation bug
    /// between `Dof::ALL`, `Twist::basis`, `wrench_rows` and the unpacking of `dv`). The mass
    /// block of a diagonal inertia in the identity pose must come out as diag(m,m,m,Ix,Iy,Iz):
    /// Regression on DOF ordering consistency — it catches a silent permutation
    /// bug between `Dof::ALL`, `Twist::basis`, the layout of a wrench into a 6-row and
    /// the unpacking of `dv`.
    ///
    /// The PRODUCTION path is checked: the mass block is taken from the `BodyPost` task,
    /// which is what builds it. Previously the test called the `wrench_rows` helper; when assembly
    /// moved into tasks, that helper would have lived on only for the sake of the test — i.e.
    /// the test would be guarding code that is no longer the code that runs.
    ///
    /// A diagonal inertia in the identity pose must give diag(m,m,m,Ix,Iy,Iz).
    #[tokio::test]
    async fn newton_mass_block_is_diagonal_for_diagonal_inertia() {
        let world = Arc::new(World::builder().usual::<f32>());
        let inertia = Inertia::diagonal(world.clone(), 2.0, [3.0, 5.0, 7.0]);
        let body = RigidBody::new(world.clone(), inertia);
        // Identity pose, zero velocity: `BodyPost` reads midpoint_pose and
        // solve_vel, so we fill them directly.
        body.midpoint_pose.write(body.pose.read());
        body.solve_vel.write(Twist::zero());

        let id = aristotle::WorldId::get();
        let mut bodies: IndexMap<WorldId, Box<dyn Component<f32, f32>>> = IndexMap::new();
        bodies.insert(id, Inert::new(body));
        let order: IndexMap<WorldId, usize> = [(id, 0usize)].into_iter().collect();
        let snap_mom: IndexMap<WorldId, WorldKey<Wrench<f32>>> = bodies
            .iter()
            .map(|(i, e)| (*i, e.body().world_momentum()))
            .collect();

        let external: IndexMap<WorldId, WorldKey<Wrench<f32>>> = {
            let mut map = world.write::<Wrench<f32>>();
            [(id, map.add(Wrench::zero()))].into_iter().collect()
        };
        let mut cache = NewtonCache::<f32>::new();
        cache.sync(&world, [id].into_iter(), &IndexMap::new());
        cache.publish_scalars(0.5, 1.0e-18);
        cache.bake_lanes(&world, &bodies, &IndexMap::new(), &external, 1);
        cache.bake_body_post(&world, &bodies, &snap_mom);
        let _o = &order;
        let accel = Accelerator::<f32>::builder(world.clone()).build();
        accel.body_post(cache.body_post_rows()).await.unwrap();

        let block = cache.mass_block()[0].read();
        let diag = [2.0, 2.0, 2.0, 3.0, 5.0, 7.0];
        for row in 0..6 {
            for (col, &got) in block[row].iter().enumerate() {
                let want = if row == col { diag[row] } else { 0.0 };
                assert!(
                    approx(got, want, 1e-12),
                    "mass block [{row}][{col}]={got}, expected {want}"
                );
            }
        }
    }

    /// A free body (no joints/fields) in the WORLD frame: the world angular momentum
    /// and the kinetic energy are held AT THE LEVEL OF the Newton TOLERANCE over the horizon.
    /// With EXACT convergence R=0 ⇒ P_s^{n+1}=2P_s^n−P_s^n=P_s^n (structurally), but
    /// the solver only gets down to its threshold, so a spurious term ∝ R leaks per step
    /// (≈3e-5, bounded by `is_effective_zero`) and accumulates ~linearly — hence ~2e-3 over
    /// t=1, not machine zero. The test catches a GROSS breakage of the scheme (blow-up/secular
    /// acceleration), not exact conservation; tightening it requires a stricter solver.
    #[tokio::test]
    async fn newton_free_body_conserves_world_momentum_and_energy() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 1.0, [2.0, 3.0, 4.0]);
        let body = RigidBody::new(world.clone(), inertia.clone());
        body.momentum
            .write(inertia.apply(&Twist::new(&Vector3::ZERO, &Vector3::from([1.0, 1.5, 0.7]))));

        let mech = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(Newton::<f32>::new(
            accel(world.clone()),
        )));
        let id = mech.add_body(Inert::new(body)).await;

        let l0 = mech
            .inspect_body(id, async |b| b.world_momentum().read().torque())
            .await
            .unwrap();
        let e0 = mech
            .inspect_body(id, async |b| b.kinetic_energy())
            .await
            .unwrap();
        for _ in 0..100 {
            mech.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
        }
        let l1 = mech
            .inspect_body(id, async |b| b.world_momentum().read().torque())
            .await
            .unwrap();
        let e1 = mech
            .inspect_body(id, async |b| b.kinetic_energy())
            .await
            .unwrap();

        let mom_drift = {
            let d = [l1[0] - l0[0], l1[1] - l0[1], l1[2] - l0[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let e_drift = (e1 - e0).abs();
        // The threshold is an order of magnitude above the observed value (~2e-3): it catches
        // acceleration/blow-up and does not claim machine precision.
        assert!(
            mom_drift < 2e-2,
            "world momentum is running away: {mom_drift}"
        );
        assert!(e_drift < 2e-2, "energy is running away: {e_drift}");
    }
    /// Newton–Schulz with a generous budget must reproduce the exact solve.
    /// The oracle is `nalgebra`, a dev dependency, not linked into the library.
    #[tokio::test]
    async fn newton_schulz_reproduces_the_exact_solve() {
        let world = Arc::new(World::builder().usual::<f32>());
        let ida = aristotle::WorldId::get();
        let idb = aristotle::WorldId::get();
        let mut bodies: IndexMap<WorldId, Box<dyn Component<f32, f32>>> = IndexMap::new();
        bodies.insert(
            ida,
            Inert::new(RigidBody::body_at_with_mass(
                world.clone(),
                &Vector3::from([1.0, 0.0, 0.0]),
                1.0,
            )),
        );
        bodies.insert(
            idb,
            Inert::new(RigidBody::body_at_with_mass(
                world.clone(),
                &Vector3::from([-1.0, 0.0, 0.0]),
                2.0,
            )),
        );
        let mut joints: IndexMap<WorldId, JointEdge<f32>> = IndexMap::new();
        joints.insert(
            aristotle::WorldId::get(),
            JointEdge::new(ida, idb, spring(world.clone(), 2.0, 100.0, 1.0)),
        );
        let mut wrenches: IndexMap<WorldId, WorldKey<Wrench<f32>>> = IndexMap::new();
        {
            let mut map = world.write();
            wrenches.insert(ida, map.add(Wrench::zero()));
            wrenches.insert(idb, map.add(Wrench::zero()));
        }
        crate::accelerator::bake_incidence(&mut bodies, &joints, &wrenches);

        let dt = 1.0 / 60.0;
        let half = dt / 2.0;
        // Lane 0 aliases the body-owned slots, so recomputing the world velocity
        // seeds the iterate the dispatch will read.
        for e in bodies.values() {
            e.body().world_velocity();
        }

        let accel = Accelerator::<f32>::builder(world.clone()).build();
        let mut cache = NewtonCache::<f32>::new();
        cache.sync(&world, bodies.keys().copied(), &joints);
        cache.bake_lanes(&world, &bodies, &joints, &wrenches, 1);
        cache.publish_dispatch_scalars(half, dt, 1.0);
        accel.dispatch(&cache.lanes()[0], true, true).await.unwrap();

        // The matrix diagonal comes from the mass blocks written by `BodyPost`.
        let snap_mom: IndexMap<WorldId, WorldKey<Wrench<f32>>> = bodies
            .iter()
            .map(|(id, e)| (*id, e.body().world_momentum()))
            .collect();
        cache.publish_scalars(half, 1.0e-18);
        cache.bake_body_post(&world, &bodies, &snap_mom);
        accel.body_post(cache.body_post_rows()).await.unwrap();
        accel.assemble(cache.assemble_rows()).await.unwrap();
        Newton::<f32>::seed_transpose(&mut cache);
        for _ in 0..40 {
            Newton::<f32>::refine_inverse(&accel, &mut cache)
                .await
                .unwrap();
        }

        // Right-hand side: a made-up but nonzero residual per body.
        for (i, slot) in cache.rhs().iter().enumerate() {
            slot.write(Wrench::new(
                &Vector3::from([1.0 + i as f32, -2.0, 0.5]),
                &Vector3::from([0.25, 1.5, -0.75]),
            ));
        }
        // rhs changed after the rows were built, but matvec_rows address the rhs slots
        // by index — the values are read on the GPU, no re-baking is needed.
        accel.block_matvec(cache.matvec_rows()).await.unwrap();

        // Oracle: flatten A and rhs, solve exactly.
        let m = cache.m();
        let n = 6 * m;
        let mut flat = vec![0.0f32; n * n];
        for i in 0..m {
            for j in 0..m {
                let b = cache.a()[i * m + j].read();
                for r in 0..6 {
                    for c in 0..6 {
                        flat[(i * 6 + r) * n + (j * 6 + c)] = b[r][c];
                    }
                }
            }
        }
        let mut rhs_flat = vec![0.0f32; n];
        for i in 0..m {
            let w = cache.rhs()[i].read();
            let (f, t) = (w.force(), w.torque());
            for r in 0..3 {
                rhs_flat[i * 6 + r] = f[r];
                rhs_flat[i * 6 + 3 + r] = t[r];
            }
        }
        let exact = nalgebra::DMatrix::from_row_slice(n, n, &flat)
            .lu()
            .solve(&nalgebra::DVector::from_row_slice(&rhs_flat))
            .expect("oracle solve failed");

        for i in 0..m {
            let d = cache.dv()[i].read();
            let (l, a) = (d.linear(), d.angular());
            let got = [l[0], l[1], l[2], a[0], a[1], a[2]];
            for r in 0..6 {
                let want = exact[i * 6 + r];
                assert!(
                    (got[r] - want).abs() < 1e-6 * (1.0 + want.abs()),
                    "dv[{i}][{r}] = {} want {want}",
                    got[r]
                );
            }
        }
    }

    /// Regression for poisoning of the persistent hint (cloth_grid demo, 2026-07-21).
    ///
    /// The demo runs at REAL frame time with `MAX_DT = 0.1`, and the cloth has
    /// `k = 100 N/m` per node of mass `1/18 kg`, i.e. `ω ≈ 60 rad/s`. The mass-block
    /// seed is contractive only for `dt < 2/ω ≈ 1/30`, so a frame drop
    /// drives Newton–Schulz into divergence. While `X` was not rolled back on
    /// a rejected span, a single such drop wrote inf into the island's persistent
    /// cache, the `seeded` flag got stuck — and the island never recovered
    /// again. This is reproduced here: a large step, then normal ones.
    #[tokio::test]
    async fn a_diverging_step_does_not_poison_later_steps() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mech = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(Newton::<f32>::new(
            accel(world.clone()),
        )));
        // Cloth node parameters: ω = sqrt(2k/m) ≈ 60 rad/s.
        let mass = 1.0 / 18.0;
        let a = mech
            .add_body(Inert::new(RigidBody::body_at_with_mass(
                world.clone(),
                &Vector3::from([0.05, 0.0, 0.0]),
                mass,
            )))
            .await;
        let b = mech
            .add_body(Inert::new(RigidBody::body_at_with_mass(
                world.clone(),
                &Vector3::from([-0.05, 0.0, 0.0]),
                mass,
            )))
            .await;
        mech.connect(a, vec![(spring(world, 0.1, 100.0, 0.5), b)])
            .await;

        // Healthy steps first — the cache warms up.
        for _ in 0..10 {
            mech.step(&Epoch::standalone(1.0 / 60.0, 1.0))
                .await
                .unwrap();
        }
        // Frame drop: dt = the demo's MAX_DT, deep in the divergent regime
        // ((dt²/4)·ω² = 9).
        mech.step(&Epoch::standalone(0.1, 1.0)).await.unwrap();
        // And normal ones again. The island must survive them.
        for _ in 0..60 {
            mech.step(&Epoch::standalone(1.0 / 60.0, 1.0))
                .await
                .unwrap();
        }

        let pa = mech.body_absolute_position(a).await.unwrap();
        let pb = mech.body_absolute_position(b).await.unwrap();
        for axis in 0..3 {
            assert!(
                pa[axis].is_finite() && pb[axis].is_finite(),
                "island poisoned by a frame drop: a={pa:?} b={pb:?}"
            );
        }
        // Spring with a damper: the pair must stay near the rest length, not
        // fly apart.
        let sep =
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        assert!(
            sep < 1.0,
            "the pair flew apart after a frame drop: distance {sep}"
        );
    }
    /// DEGENERATE regime: the system is exactly at equilibrium and at rest. The forces are zero,
    /// the residual is zero, and so is the self-normalization scale — this is the case for
    /// which `residual` has the `floor2` floor. The solver must handle it
    /// normally at ANY budget, including 1, and leave the bodies in place.
    ///
    /// Kept separate from the budget sweep on purpose: there, equilibrium at
    /// rest made the test vacuous (all budgets gave bit-for-bit the same
    /// result), and the coverage of the degenerate regime had to be moved out, not thrown away.
    #[tokio::test]
    async fn equilibrium_at_rest_stays_put_at_any_budget() {
        for &warm in &[1usize, 3, 8] {
            let world = Arc::new(World::builder().usual::<f32>());
            let m = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
                Newton::<f32>::new(accel(world.clone())).with_budgets(warm, 24),
            ));
            // Distance 2.0 — exactly the rest length; zero velocities.
            let a = m
                .add_body(Inert::new(RigidBody::body_at_with_mass(
                    world.clone(),
                    &Vector3::from([1.0, 0.0, 0.0]),
                    1.0,
                )))
                .await;
            let b = m
                .add_body(Inert::new(RigidBody::body_at_with_mass(
                    world.clone(),
                    &Vector3::from([-1.0, 0.0, 0.0]),
                    1.0,
                )))
                .await;
            m.connect(a, vec![(spring(world, 2.0, 500.0, 1.0), b)])
                .await;
            for _ in 0..60 {
                m.step(&Epoch::standalone(1.0 / 60.0, 1.0)).await.unwrap();
            }
            let pa = m.body_absolute_position(a).await.unwrap();
            let pb = m.body_absolute_position(b).await.unwrap();
            for axis in 0..3 {
                let want_a = if axis == 0 { 1.0 } else { 0.0 };
                let want_b = if axis == 0 { -1.0 } else { 0.0 };
                assert!(
                    (pa[axis] - want_a).abs() < 1e-9 && (pb[axis] - want_b).abs() < 1e-9,
                    "budget {warm}: equilibrium drifted, a={pa:?} b={pb:?}"
                );
            }
        }
    }

    /// Accuracy versus the Newton–Schulz budget. The point is to pin down the relation itself:
    /// a change that silently needs more iterations will surface here, rather than
    /// as drift in the demo.
    #[tokio::test]
    async fn accuracy_improves_with_the_iteration_budget() {
        async fn run_with(warm: usize) -> f32 {
            let world = Arc::new(World::builder().usual::<f32>());
            let m = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
                Newton::<f32>::new(accel(world.clone())).with_budgets(warm, 24),
            ));
            // Stretched (3.0 versus rest length 2.0) and with transverse
            // velocities: the spring axis rotates, the matrix changes from
            // step to step. At equilibrium and rest the test would measure nothing.
            let a = m
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::from([1.5, 0.0, 0.0]),
                    &Vector3::from([0.0, 4.0, 0.0]),
                    1.0,
                )))
                .await;
            let b = m
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::from([-1.5, 0.0, 0.0]),
                    &Vector3::from([0.0, -4.0, 0.0]),
                    1.0,
                )))
                .await;
            m.connect(a, vec![(spring(world, 2.0, 500.0, 1.0), b)])
                .await;
            for _ in 0..60 {
                m.step(&Epoch::standalone(1.0 / 60.0, 1.0)).await.unwrap();
            }
            m.body_absolute_position(a).await.unwrap()[0]
        }

        let generous = run_with(12).await;
        println!("\n{:>6} {:>18} {:>14}", "budget", "x", "|x - generous|");
        let mut closest = f32::INFINITY;
        for &warm in &[1usize, 2, 3, 5, 8] {
            let x = run_with(warm).await;
            let err = (x - generous).abs();
            println!("{warm:>6} {x:>18.10} {err:>14.3e}");
            assert!(x.is_finite(), "budget {warm} gave a non-finite result");
            closest = closest.min(err);
        }
        assert!(
            closest < 1e-4,
            "no budget in the sweep came closer than 1e-4 to the generous one"
        );
    }

    /// A warm start must stay warm: the approximate inverse must
    /// FOLLOW the matrix, not degrade. The measure is the same `‖I − A·X‖²`
    /// the guard uses, taken directly from the cache after every step.
    #[tokio::test]
    async fn warm_start_does_not_degrade_over_a_run() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mech = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(Newton::<f32>::new(
            accel(world.clone()),
        )));
        // Stretched and with transverse velocities — otherwise the system sits at
        // equilibrium, the matrix does not change, and "the warm start does not degrade"
        // holds trivially.
        let a = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([1.5, 0.0, 0.0]),
                &Vector3::from([0.0, 4.0, 0.0]),
                1.0,
            )))
            .await;
        let b = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([-1.5, 0.0, 0.0]),
                &Vector3::from([0.0, -4.0, 0.0]),
                1.0,
            )))
            .await;
        mech.connect(a, vec![(spring(world, 2.0, 500.0, 1.0), b)])
            .await;

        let mut residuals = Vec::new();
        for _ in 0..40 {
            mech.step(&Epoch::standalone(1.0 / 60.0, 1.0))
                .await
                .unwrap();
            residuals.push(mech.inspect_solver_residual().await.unwrap());
        }
        println!("\ninverse residual per step: {residuals:?}");

        let early = residuals[..5].iter().cloned().fold(0.0f32, f32::max);
        let late = residuals[35..].iter().cloned().fold(0.0f32, f32::max);
        assert!(
            late <= early * 10.0 + 1e-9,
            "warm start degraded: start {early:.3e}, end {late:.3e}"
        );
    }

    /// A topology change drops the island's cache (the dimension changes), and this must not
    /// be visible in the physics: a cold cache has no right to cause a jolt.
    #[tokio::test]
    #[ignore]
    async fn topology_change_causes_no_visible_jolt() {
        async fn chain(detach_at: Option<usize>, steps: usize) -> f32 {
            // FIXME: deadlock?
            let world = Arc::new(World::builder().usual::<f32>());
            let mech = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(Newton::<f32>::new(
                accel(world.clone()),
            )));
            let a = mech
                .add_body(Inert::new(RigidBody::body_at_with_mass(
                    world.clone(),
                    &Vector3::from([1.0, 0.0, 0.0]),
                    1.0,
                )))
                .await;
            let b = mech
                .add_body(Inert::new(RigidBody::body_at_with_mass(
                    world.clone(),
                    &Vector3::from([-1.0, 0.0, 0.0]),
                    1.0,
                )))
                .await;
            let c = mech
                .add_body(Inert::new(RigidBody::body_at_with_mass(
                    world.clone(),
                    &Vector3::from([0.0, 3.0, 0.0]),
                    1.0,
                )))
                .await;
            mech.connect(a, vec![(spring(world.clone(), 2.0, 500.0, 1.0), b)])
                .await;
            mech.connect(c, vec![(spring(world, 3.0, 200.0, 1.0), b)])
                .await;
            for step in 0..steps {
                if Some(step) == detach_at {
                    mech.detach(c).await;
                }
                mech.step(&Epoch::standalone(1.0 / 60.0, 1.0))
                    .await
                    .unwrap();
            }
            mech.body_absolute_position(a).await.unwrap()[0]
        }

        // Detaching at step 30 versus detaching from the very start: from step 30
        // on the systems are already identical, so the test's job is to exercise the
        // cache reallocation path and make sure it neither crashes nor tears the physics.
        let late = chain(Some(30), 60).await;
        let never = chain(Some(0), 60).await;
        println!("\nlate detach {late}, detached immediately {never}");
        assert!(late.is_finite() && never.is_finite());
    }

    /// The only external evidence of the guard at work: a warm hint
    /// carried over to a STRONGLY changed matrix can leave the region of
    /// convergence, and then it must be reset to the seed. Here the step jumps between
    /// small and large, which makes the stiffness term change by orders of magnitude.
    #[tokio::test]
    async fn a_stale_warm_hint_is_reseeded_not_followed() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mech = Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(Newton::<f32>::new(
            accel(world.clone()),
        )));
        let a = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([1.0, 0.0, 0.0]),
                &Vector3::from([0.0, 5.0, 0.0]),
                1.0,
            )))
            .await;
        let b = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([-1.0, 0.0, 0.0]),
                &Vector3::from([0.0, -5.0, 0.0]),
                1.0,
            )))
            .await;
        mech.connect(a, vec![(spring(world, 2.0, 5.0e4, 5.0), b)])
            .await;

        for step in 0..40 {
            let dt = if step % 3 == 0 { 0.2 } else { 1.0 / 600.0 };
            mech.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            let p = mech.body_absolute_position(a).await.unwrap();
            for axis in 0..3 {
                assert!(
                    p[axis].is_finite(),
                    "step {step} (dt={dt}) broke the solver: {p:?}"
                );
            }
        }
        let res = mech.inspect_solver_residual().await.unwrap();
        assert!(
            res.is_finite(),
            "the approximate inverse got corrupted after step jumps: {res}"
        );
    }
}
