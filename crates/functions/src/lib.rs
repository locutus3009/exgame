// SPDX-License-Identifier: MIT

use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Dof, Motor, Twist, Wrench},
};
use peano::prelude::*;

/// One 6×6 block of the implicit solver's system matrix, as the accelerator
/// stores it. Plain array, index `[row][col]` — no blade packing, so no probe
/// mapping applies to it anywhere.
pub type Block<T> = [[T; 6]; 6];

pub fn pre<T: Scalar + StandardPart + Lift<T> + Pod>(
    pose: Motor<T>,
    vel: Twist<T>,
    retraction: T,
) -> (Motor<T>, Twist<T>) {
    (pose.compose(&vel.exp(retraction)), vel)
}

/// One term of the wrench gather: add an incident connection's wrench to the
/// running total. Traced into the `Gather` kernel's loop body — there is no CPU
/// copy of this, the shader is the only consumer.
pub fn gather_term<T: Scalar + StandardPart + Lift<T> + Pod>(
    acc: Wrench<T>,
    w: Wrench<T>,
) -> Wrench<T> {
    acc + w
}

/// One term of the Newton correction `dv = Σ_j x[i*m + j] · rhs[j]`: apply one
/// block to one body's residual wrench and add it to the running twist.
pub fn matvec_term<T: Scalar + StandardPart + Lift<T> + Pod>(
    acc: Twist<T>,
    x: Block<T>,
    w: Wrench<T>,
) -> Twist<T> {
    let (f, t) = (w.force(), w.torque());
    let v = [f[0], f[1], f[2], t[0], t[1], t[2]];
    let (l, a) = (acc.linear(), acc.angular());
    let mut o = [l[0], l[1], l[2], a[0], a[1], a[2]];
    for (r, or) in o.iter_mut().enumerate() {
        let mut s = T::ZERO;
        for (c, vc) in v.iter().enumerate() {
            s += x[r][c] * *vc;
        }
        *or += s;
    }
    Twist::new(
        &Vector3::from([o[0], o[1], o[2]]),
        &Vector3::from([o[3], o[4], o[5]]),
    )
}

/// One term of a block matrix product: `acc + a · (sign · b + diag · 2 · I)`.
///
/// The `two_minus` fold of the Newton–Schulz step and the `k == j` diagonal test
/// are resolved on the host into those two coefficients, so the kernel carries no
/// branch and no knowledge of `m`, `i` or `j`. The block ring is NOT commutative
/// — `a` is the left factor, always.
pub fn gemm_term<T: Scalar + StandardPart + Lift<T> + Pod>(
    acc: Block<T>,
    a: Block<T>,
    b: Block<T>,
    sign: T,
    diag: T,
) -> Block<T> {
    let two = T::ONE + T::ONE;
    let mut right = [[T::ZERO; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            right[r][c] = sign * b[r][c];
        }
        right[r][r] += diag * two;
    }
    let mut out = acc;
    for r in 0..6 {
        for c in 0..6 {
            let mut s = T::ZERO;
            for (k, rk) in right.iter().enumerate() {
                s += a[r][k] * rk[c];
            }
            out[r][c] += s;
        }
    }
    out
}

/// One term of `Σ ‖diag·I − b‖²_F` over a list of blocks.
///
/// This is how far an approximate inverse is from being one: with `b` the blocks
/// of `R = A·X` and `diag` set on the diagonal cells, the sum is `‖I − A·X‖²`.
/// With `diag = 0` it is `‖X‖²`, which is finite exactly when every component of
/// `X` is — the check `checkpoint` needs before publishing a hint.
///
/// The identity is a MULTIPLIER rather than a branch, the same shape as
/// `gemm_term`'s `diag`: the host knows whether a cell is diagonal at bake time,
/// and the kernel must not learn about `i`, `j` or `m`.
///
/// The square root is not taken. Every consumer compares this against another
/// value of the same quantity, and `sqrt` would fork the decision trie.
pub fn reduce_sq_term<T: Scalar + StandardPart + Lift<T> + Pod>(acc: T, b: Block<T>, diag: T) -> T {
    let mut out = acc;
    for (r, row) in b.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            // `want` is a compile-time 0 or 1 per cell, so the off-diagonal terms
            // fold to `-v` and only the six diagonal ones keep `diag`.
            let e = if r == c { diag - v } else { -v };
            out += e * e;
        }
    }
    out
}

/// One block, unchanged.
///
/// There is no arithmetic here and that is the point: the solver's block copies
/// (`checkpoint` publishing the accepted approximate inverse, `rollback`
/// restoring it) were `m²` round trips through host storage, and they belong on
/// the device with everything else that touches those arrays. Routing even a copy
/// through the tracer keeps one rule with no exceptions — the shader is generated
/// from a function over `Scalar`, never hand-written — so the block's layout is
/// described in exactly one place.
pub fn copy_block<T: Scalar + StandardPart + Lift<T> + Pod>(b: Block<T>) -> Block<T> {
    b
}

/// One connection's contribution to one block of the system matrix.
///
/// The factors are the midpoint scheme's, not the kernel's: over half a step the
/// pose moves by `(dt/2)·V`, so a connection's velocity columns carry `−dt/2` and
/// its pose columns `−(dt/2)²`. They arrive as `pf` and `vf`, already formed by
/// the caller.
pub fn assemble_term<T: Scalar + StandardPart + Lift<T> + Pod>(
    acc: Block<T>,
    pose: [Wrench<T>; 6],
    vel: [Wrench<T>; 6],
    pf: T,
    vf: T,
) -> Block<T> {
    let mut out = acc;
    for col in 0..6 {
        let (pfc, ptc) = (pose[col].force(), pose[col].torque());
        let (vfc, vtc) = (vel[col].force(), vel[col].torque());
        let p = [pfc[0], pfc[1], pfc[2], ptc[0], ptc[1], ptc[2]];
        let v = [vfc[0], vfc[1], vfc[2], vtc[0], vtc[1], vtc[2]];
        for r in 0..6 {
            out[r][col] = out[r][col] + pf * p[r] + vf * v[r];
        }
    }
    out
}

/// POST — per body: the world mass block, the residual and its self-scale.
///
/// The mass block `𝕀_s = Ad*_M ∘ I ∘ Ad_M⁻¹` is formed here ONCE and used twice:
/// as the residual's `P_s = 𝕀_s·V_s` and as the matrix's diagonal block.
///
/// The inertia arrives already split — `mass` scalar, `angular` a 3×3 tensor. A
/// diagonal inertia is passed as a diagonal matrix: the caller picks the kernel
/// variant, so no branch reaches the shader.
///
/// The self-scale is the largest of the squared 3-norms of the residual's own
/// terms, force and torque separately (different units). `floor2` is a division
/// guard, not a tolerance: a body at pure rest has zero residual AND zero scale,
/// and `0/floor = 0`.
#[allow(clippy::too_many_arguments)]
pub fn body_post<T: Scalar + StandardPart + Lift<T> + Pod>(
    pose_mid: Motor<T>,
    solve_vel: Twist<T>,
    mass: T,
    angular: [[T; 3]; 3],
    snap_mom: Wrench<T>,
    total_wrench: Wrench<T>,
    half: T,
    floor2: T,
) -> (Block<T>, Wrench<T>, Wrench<T>) {
    let inv = pose_mid.inverse();

    // World mass block column by column: world basis twist → body → inertia →
    // back to the world.
    let mut block = [[T::ZERO; 6]; 6];
    for (col, dof) in Dof::ALL.iter().enumerate() {
        let ej_body = inv.conjugate(&Twist::basis(*dof));
        let v = ej_body.linear();
        let w = ej_body.angular();
        let l = [
            angular[0][0] * w[0] + angular[0][1] * w[1] + angular[0][2] * w[2],
            angular[1][0] * w[0] + angular[1][1] * w[1] + angular[1][2] * w[2],
            angular[2][0] * w[0] + angular[2][1] * w[1] + angular[2][2] * w[2],
        ];
        let column = Wrench::new(&v.scale(mass), &Vector3::from(l)).transport(&pose_mid);
        let f = column.force();
        let t = column.torque();
        let vals = [f[0], f[1], f[2], t[0], t[1], t[2]];
        for (r, val) in vals.iter().enumerate() {
            block[r][col] = *val;
        }
    }

    // P_s = 𝕀_s·V_s — matvec over the block just assembled.
    let (vl, va) = (solve_vel.linear(), solve_vel.angular());
    let v = [vl[0], vl[1], vl[2], va[0], va[1], va[2]];
    let mut p = [T::ZERO; 6];
    for (r, pr) in p.iter_mut().enumerate() {
        let mut acc = T::ZERO;
        for (c, vc) in v.iter().enumerate() {
            acc += block[r][c] * *vc;
        }
        *pr = acc;
    }

    let hw = total_wrench * half;
    let (nf, nt) = (snap_mom.force(), snap_mom.torque());
    let (hf, ht) = (hw.force(), hw.torque());
    let nn = [nf[0], nf[1], nf[2], nt[0], nt[1], nt[2]];
    let hh = [hf[0], hf[1], hf[2], ht[0], ht[1], ht[2]];

    // R = P_s^mid − P_s^n − (dt/2)·W;  rhs = −R.
    let mut res = [T::ZERO; 6];
    for r in 0..6 {
        res[r] = -(p[r] - nn[r] - hh[r]);
    }
    let rhs = Wrench::new(
        &Vector3::from([res[0], res[1], res[2]]),
        &Vector3::from([res[3], res[4], res[5]]),
    );

    // Block self-scale: the largest of the squared 3-norms of the residual terms.
    let sq = |x: &[T; 6], o: usize| x[o] * x[o] + x[o + 1] * x[o + 1] + x[o + 2] * x[o + 2];
    let lin = sq(&p, 0)
        .max_explicit(sq(&nn, 0))
        .max_explicit(sq(&hh, 0))
        .max_explicit(floor2);
    let ang = sq(&p, 3)
        .max_explicit(sq(&nn, 3))
        .max_explicit(sq(&hh, 3))
        .max_explicit(floor2);
    let scale = Wrench::new(
        &Vector3::from([lin, lin, lin]),
        &Vector3::from([ang, ang, ang]),
    );

    (block, rhs, scale)
}
