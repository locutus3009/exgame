// SPDX-License-Identifier: MIT

//! PROTOTYPE — implicit constrained dynamics step  `M·a = f_ext + Jᵀλ`.
//!
//! Not engine code. A bare f64 sketch, no clifford / no traits / no real
//! `ForceField`, written to let the *shape of the phase-5 solver* fall out
//! before any signature is touched (the discipline the repo confirmed thrice:
//! free-function `#[test]` first, architecture second).
//!
//! WHAT THIS MODELS. Planar point masses, some pairs joined by RIGID rods, some
//! coordinates pinned. One backward-Euler step solved implicitly:
//!
//!     v_{n+1} = v_n + dt·M⁻¹(f_ext + Jᵀ(q_{n+1})·λ)
//!     q_{n+1} = q_n + dt·v_{n+1}
//!     C(q_{n+1}) = 0                              (rigid constraints)
//!
//! Eliminating v gives the per-step root-find on (q_trial, λ):
//!
//!     r_dyn(q,λ) = M(q − q_n − dt·v_n) − dt²(f_ext + Jᵀ(q)λ) = 0     ⇔  M·a = f_ext + Jᵀλ
//!     r_con(q)   = C(q) = 0
//!
//! solved by a local (Gauss–)Newton on the saddle system
//!
//!     ┌ M      −dt²Jᵀ ┐ ┌ δq ┐     ┌ r_dyn ┐
//!     │ J       0     │ │ δλ │ = − │ r_con │
//!     └               ┘ └    ┘     └       ┘
//!
//! MAPPING TO THE ENGINE PHASES (placeholders flagged inline):
//!   * `f_ext`  ← phase 4 `ForceField::accumulate` + any PENALTY joint (soft
//!     constraint stays a wrench). Frozen at step start — the lag-1 gravity
//!     snapshot lives here and never re-enters the Newton loop, so the lag-1 /
//!     implicit tension dissolves: external forces are explicit, constraints
//!     implicit, and the two phases do not cross.
//!   * `graph`  ← the RIGID-joint topology. Today phase 3 turns a joint into a
//!     wrench (mechanism.rs:569-577); a rigid joint instead becomes an equation
//!     the solver evaluates on the TRIAL state — it dissolves into solver input,
//!     it does not move in time.
//!   * `solve_step` ← phase-5 object. NOTE it now needs `(bodies, f_ext, graph,
//!     dt)` — strictly more than today's `step_all(bodies, wrenches, epoch)`,
//!     which never sees topology. That is the real signature pressure; whether
//!     this stays `ImplicitIntegrator` or becomes a `Solver` (with per-body
//!     explicit schemes as the empty-graph degenerate case) is the open question
//!     this prototype exists to inform — NOT decided here.
//!
//! OPEN QUESTIONS this sketch is meant to expose (see asserts / comments):
//!   - empty graph MUST collapse to `for body { step }` at zero overhead — the
//!     dispatch is the first line of `solve_step`, and `fast_path` proves it.
//!   - the solver wants dense `0..n` row indices for J (here: `2*body+axis`) —
//!     this is where `IndexMap::get_index` would earn its keep over a slice.
//!   - it also wants per-pair access to two bodies at once (here trivial because
//!     the state is one flat `Vec<f64>`; in the engine that is the
//!     `get_disjoint_mut` vs `split_at_mut` question).
//!
//! DELIBERATE SIMPLIFICATIONS (prototype, not production):
//!   - Gauss–Newton: the geometric-stiffness term `−dt²·∂(Jᵀλ)/∂q` is dropped.
//!     It is O(dt²·λ) and does not affect final constraint satisfaction (the
//!     bottom block drives C→0 exactly); it only nudges convergence rate.
//!   - `f_ext` is semi-implicit (evaluated at step start). If a stiff solve
//!     moves bodies far within the step, f_ext is slightly stale w.r.t. the
//!     post-solve pose — the standard "external explicit, constraints implicit"
//!     trade, negligible for slowly-varying gravity/aero, a conscious choice
//!     for deep-contact resolution. Knowing it, not stumbling into it.

// ───────────────────────── minimal dense linear algebra ─────────────────────
// Gaussian elimination, partial pivot. Self-contained; the saddle system is
// small (n = 2·bodies + constraints). Prototype-grade, not a sparse solver.

fn solve_dense(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r * n + col].abs() > a[piv * n + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        assert!(d.abs() > 1e-14, "singular KKT (column {col})");
        for r in (col + 1)..n {
            let f = a[r * n + col] / d;
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in (r + 1)..n {
            s -= a[r * n + c] * x[c];
        }
        x[r] = s / a[r * n + r];
    }
    x
}

// ───────────────────────────── world state ─────────────────────────────────
// Flat planar state: q = [x0, y0, x1, y1, …], dof index = 2*body + axis.

#[derive(Clone)]
struct World {
    q: Vec<f64>,    // positions
    v: Vec<f64>,    // velocities
    mass: Vec<f64>, // per body (isotropic point mass)
}

impl World {
    fn n_dof(&self) -> usize {
        self.q.len()
    }
    fn mass_dof(&self, i: usize) -> f64 {
        self.mass[i / 2]
    }
    /// Mass-weighted centre of mass (2D).
    fn com(&self) -> [f64; 2] {
        let mtot: f64 = self.mass.iter().sum();
        let mut c = [0.0, 0.0];
        for b in 0..self.mass.len() {
            c[0] += self.mass[b] * self.q[2 * b];
            c[1] += self.mass[b] * self.q[2 * b + 1];
        }
        [c[0] / mtot, c[1] / mtot]
    }
    fn com_vel(&self) -> [f64; 2] {
        let mtot: f64 = self.mass.iter().sum();
        let mut c = [0.0, 0.0];
        for b in 0..self.mass.len() {
            c[0] += self.mass[b] * self.v[2 * b];
            c[1] += self.mass[b] * self.v[2 * b + 1];
        }
        [c[0] / mtot, c[1] / mtot]
    }
}

// ──────────────────── constraint graph  (PLACEHOLDER topology) ──────────────
// Each variant is ONE scalar constraint → one J row → one λ. This is the rigid
// joint topology the phase-5 solver would receive; today's `step_all` gets none
// of it. A pin = two `Coord` constraints.

enum Constraint {
    /// Pin one coordinate to a target value (anchor / ground joint).
    Coord {
        body: usize,
        axis: usize,
        target: f64,
    },
    /// Rigid rod: squared distance between two bodies fixed (smooth, no sqrt).
    Dist { a: usize, b: usize, len2: f64 },
}

impl Constraint {
    fn value(&self, q: &[f64]) -> f64 {
        match *self {
            Constraint::Coord { body, axis, target } => q[2 * body + axis] - target,
            Constraint::Dist { a, b, len2 } => {
                let dx = q[2 * a] - q[2 * b];
                let dy = q[2 * a + 1] - q[2 * b + 1];
                dx * dx + dy * dy - len2
            }
        }
    }

    /// One row of J = ∂C/∂q. (Dense `0..n_dof` indexing — the `get_index`
    /// access pattern a real solver needs from the body container.)
    fn jacobian_row(&self, q: &[f64], n_dof: usize) -> Vec<f64> {
        let mut r = vec![0.0; n_dof];
        match *self {
            Constraint::Coord { body, axis, .. } => r[2 * body + axis] = 1.0,
            Constraint::Dist { a, b, .. } => {
                let dx = q[2 * a] - q[2 * b];
                let dy = q[2 * a + 1] - q[2 * b + 1];
                r[2 * a] = 2.0 * dx;
                r[2 * a + 1] = 2.0 * dy;
                r[2 * b] = -2.0 * dx;
                r[2 * b + 1] = -2.0 * dy;
            }
        }
        r
    }
}

// ─────────────────── f_ext  (PLACEHOLDER for ForceField + penalty) ──────────
// In the engine this is phase 4: `ForceField::accumulate` (gravity from the
// lag-1 front snapshot) plus any PENALTY joint wrench. Frozen here, at step
// start, exactly as the engine freezes it. Uniform gravity stands in.

fn external_forces(w: &World, g: [f64; 2]) -> Vec<f64> {
    let mut f = vec![0.0; w.n_dof()];
    for b in 0..w.mass.len() {
        f[2 * b] = w.mass[b] * g[0];
        f[2 * b + 1] = w.mass[b] * g[1];
    }
    f
}

// ─────────────────── the implicit step  (PLACEHOLDER for phase-5) ───────────

struct StepReport {
    newton_iters: usize,
    fast_path: bool, // empty graph ⇒ collapsed to per-body, no solver
}

fn solve_step(w: &mut World, graph: &[Constraint], f_ext: &[f64], dt: f64) -> StepReport {
    let n = w.n_dof();
    let m = graph.len();
    let dt2 = dt * dt;
    let qn = w.q.clone();
    let vn = w.v.clone();

    // Unconstrained predictor — this IS the per-body semi-implicit (symplectic
    // Euler) update: q = q_n + dt·v_n + dt²·M⁻¹·f_ext.
    let mut q: Vec<f64> = (0..n)
        .map(|i| qn[i] + dt * vn[i] + dt2 * f_ext[i] / w.mass_dof(i))
        .collect();

    // DEGENERATE DISPATCH — first line. No rigid constraints (or penalty-only,
    // which already lives in f_ext) ⇒ the predictor is the answer. Zero solver
    // overhead. This is the path N-body gravity stays on forever.
    if m == 0 {
        for i in 0..n {
            w.v[i] = (q[i] - qn[i]) / dt;
        }
        w.q = q;
        return StepReport {
            newton_iters: 0,
            fast_path: true,
        };
    }

    // Coupled implicit solve on (q_trial, λ).
    let mut lambda = vec![0.0; m];
    let dim = n + m;
    let mut iters = 0usize;
    loop {
        // J evaluated at the TRIAL state — the implicit heart. No f_ext re-eval.
        let rows: Vec<Vec<f64>> = graph.iter().map(|c| c.jacobian_row(&q, n)).collect();

        // residual = [r_dyn ; r_con]
        let mut res = vec![0.0; dim];
        for i in 0..n {
            let mut jt_lambda = 0.0;
            for k in 0..m {
                jt_lambda += rows[k][i] * lambda[k];
            }
            res[i] = w.mass_dof(i) * (q[i] - qn[i] - dt * vn[i]) - dt2 * (f_ext[i] + jt_lambda);
        }
        for k in 0..m {
            res[n + k] = graph[k].value(&q);
        }
        let err = res.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if err < 1e-10 {
            break;
        }

        // KKT saddle matrix (Gauss–Newton: geometric-stiffness term dropped).
        let mut a = vec![0.0; dim * dim];
        for i in 0..n {
            a[i * dim + i] = w.mass_dof(i); // M
        }
        for k in 0..m {
            for i in 0..n {
                let j = rows[k][i];
                a[(n + k) * dim + i] = j; // J        (bottom-left)
                a[i * dim + (n + k)] = -dt2 * j; // −dt²Jᵀ (top-right)
            }
        }
        let rhs: Vec<f64> = res.iter().map(|x| -x).collect();
        let delta = solve_dense(a, rhs, dim);
        for i in 0..n {
            q[i] += delta[i];
        }
        for k in 0..m {
            lambda[k] += delta[n + k];
        }

        iters += 1;
        assert!(iters < 60, "Newton failed to converge");
    }

    for i in 0..n {
        w.v[i] = (q[i] - qn[i]) / dt;
    }
    w.q = q;
    StepReport {
        newton_iters: iters,
        fast_path: false,
    }
}

// ───────────────────────────────── tests ───────────────────────────────────

const G: [f64; 2] = [0.0, -9.81];

fn rod_len(w: &World, a: usize, b: usize) -> f64 {
    let dx = w.q[2 * a] - w.q[2 * b];
    let dy = w.q[2 * a + 1] - w.q[2 * b + 1];
    (dx * dx + dy * dy).sqrt()
}

/// RIGID CONSTRAINTS HOLD under rich motion. Pinned double pendulum (pin + two
/// rods): released horizontal, it swings and rotates — yet every rod stays at
/// its rest length to solver tolerance every step. Exercises the Newton solve
/// hard (non-trivial λ, real geometry change on the trial state).
#[test]
fn rigid_rods_stay_rigid_under_motion() {
    let mut w = World {
        q: vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0], // bodies 0,1,2 in a horizontal line
        v: vec![0.0; 6],
        mass: vec![1.0, 1.0, 1.0],
    };
    // body 0 pinned at origin + rigid rods 0–1 and 1–2, each length 1.
    let graph = [
        Constraint::Coord {
            body: 0,
            axis: 0,
            target: 0.0,
        },
        Constraint::Coord {
            body: 0,
            axis: 1,
            target: 0.0,
        },
        Constraint::Dist {
            a: 0,
            b: 1,
            len2: 1.0,
        },
        Constraint::Dist {
            a: 1,
            b: 2,
            len2: 1.0,
        },
    ];

    let dt = 0.005;
    let mut max_iters = 0;
    for _ in 0..400 {
        // t = 2 s
        let f_ext = external_forces(&w, G); // phase-4 placeholder, frozen
        let rep = solve_step(&mut w, &graph, &f_ext, dt);
        assert!(!rep.fast_path);
        max_iters = max_iters.max(rep.newton_iters);

        assert!((rod_len(&w, 0, 1) - 1.0).abs() < 1e-6, "rod 0-1 drifted");
        assert!((rod_len(&w, 1, 2) - 1.0).abs() < 1e-6, "rod 1-2 drifted");
        assert!(w.q[0].abs() < 1e-6 && w.q[1].abs() < 1e-6, "pin slipped");
    }
    // It actually moved (not a frozen trivial solution): tip fell well below start.
    assert!(
        w.q[5] < -0.5,
        "pendulum tip should have fallen, y = {}",
        w.q[5]
    );
    // Newton converged cheaply — the geometric term we dropped was indeed small.
    assert!(
        max_iters <= 6,
        "unexpectedly many Newton iters: {max_iters}"
    );
}

/// INTERNAL FORCES SUM TO ZERO ⇒ the centre of mass free-falls at exactly g,
/// no matter how violently the rods yank the bodies. This is the Jᵀ structure
/// (equal-and-opposite constraint forces) made into an invariant. Asymmetric
/// masses + an initial kick make λ genuinely non-zero, so it is a real test of
/// the sign convention, not a trivial uniform free-fall.
#[test]
fn com_freefalls_while_rods_act() {
    let mut w = World {
        q: vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        v: vec![0.0, 1.5, 0.0, 0.0, 0.0, 0.0], // kick body 0 upward only
        mass: vec![1.0, 2.0, 3.0],             // asymmetric
    };
    let graph = [
        Constraint::Dist {
            a: 0,
            b: 1,
            len2: 1.0,
        },
        Constraint::Dist {
            a: 1,
            b: 2,
            len2: 1.0,
        },
    ];

    let dt = 0.005;
    let mut com_v = w.com_vel();
    let mut com_p = w.com();
    for _ in 0..300 {
        // t = 1.5 s
        let f_ext = external_forces(&w, G);
        solve_step(&mut w, &graph, &f_ext, dt);

        // Expected CoM follows symplectic free-fall, independent of λ.
        com_v = [com_v[0] + dt * G[0], com_v[1] + dt * G[1]];
        com_p = [com_p[0] + dt * com_v[0], com_p[1] + dt * com_v[1]];

        let c = w.com();
        assert!(
            (c[0] - com_p[0]).abs() < 1e-9,
            "CoM x: {} vs {}",
            c[0],
            com_p[0]
        );
        assert!(
            (c[1] - com_p[1]).abs() < 1e-9,
            "CoM y: {} vs {}",
            c[1],
            com_p[1]
        );

        // Rods still rigid throughout.
        assert!((rod_len(&w, 0, 1) - 1.0).abs() < 1e-6);
        assert!((rod_len(&w, 1, 2) - 1.0).abs() < 1e-6);
    }
}

/// DEGENERATE DISPATCH. Empty graph ⇒ the "solver" is never entered: it
/// collapses to the per-body semi-implicit update at zero overhead, and the
/// result matches the closed form `v += dt·g ; q += dt·v` to 1e-12 — NOT
/// bit-for-bit: `(dt·dt)·g` vs `dt·(dt·g)` differ by ~1 ULP in op ordering.
/// This is the
/// invariant that keeps N-body gravity (no rigid constraints) on its current
/// cheap path even after phase 5 grows a solver.
#[test]
fn empty_graph_collapses_to_per_body() {
    let mut w = World {
        q: vec![3.0, -2.0],
        v: vec![1.0, 0.5],
        mass: vec![2.0],
    };
    let graph: [Constraint; 0] = [];
    let dt = 0.01;

    // Independent closed-form reference (symplectic Euler).
    let (mut rq, mut rv) = (w.q.clone(), w.v.clone());

    for _ in 0..50 {
        let f_ext = external_forces(&w, G); // could also carry penalty-joint wrenches
        let rep = solve_step(&mut w, &graph, &f_ext, dt);
        assert!(
            rep.fast_path,
            "empty graph must take the per-body fast path"
        );
        assert_eq!(
            rep.newton_iters, 0,
            "no Newton iteration with no constraints"
        );

        rv[0] += dt * G[0];
        rv[1] += dt * G[1];
        rq[0] += dt * rv[0];
        rq[1] += dt * rv[1];
        assert!((w.q[0] - rq[0]).abs() < 1e-12 && (w.q[1] - rq[1]).abs() < 1e-12);
        assert!((w.v[0] - rv[0]).abs() < 1e-12 && (w.v[1] - rv[1]).abs() < 1e-12);
    }
}
