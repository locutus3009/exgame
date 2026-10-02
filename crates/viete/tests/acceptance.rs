// SPDX-License-Identifier: MIT

#![cfg(feature = "lua")]

use clifford::pga3::{Motor, Twist};
use peano::prelude::*;
use viete::{Sym, Tracer};

/// The computation under test, generic over the scalar carrier.
fn exp_coeffs<S>(inp: &[S]) -> Vec<S>
where
    S: Scalar + StandardPart + FromRational + Copy,
{
    let tw = Twist::new(
        &Vector3::from([inp[0], inp[1], inp[2]]),
        &Vector3::from([inp[3], inp[4], inp[5]]),
    );
    let m = Motor::exp(&tw);
    std::array::from_fn::<S, 16, _>(|i| m.as_mv().get(i)).to_vec()
}

fn assert_close(got: Vec<f64>, want: Vec<f64>) {
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs();
        let rel = d / want[i].abs().max(1.0);
        assert!(
            d < 1e-9 || rel < 1e-9,
            "coeff {i}: lua={} f64={} (abs {d:e})",
            got[i],
            want[i]
        );
    }
}

#[test]
fn lua_matches_f64_nontrivial() {
    let tracer = Tracer::builder().build();
    let traced = tracer.trace::<_>(6, 0, 16, &[], |inp, _| exp_coeffs::<Sym>(inp));

    // non-trivial: nonzero rotation AND translation -> u >> 0 -> closed-form arms
    let inputs = vec![0.3, -0.7, 1.1, 0.5, 0.9, -0.4];
    let got = traced.run_lua(&inputs, &[]).unwrap();
    let want = exp_coeffs::<f64>(&inputs);
    assert_close(got, want);
}

fn div_pair<S>(inp: &[S]) -> Vec<S>
where
    S: Scalar + StandardPart + FromRational + Copy,
{
    vec![inp[0] / inp[1]]
}

#[test]
fn div_guard_no_false_trap_on_small_divisor() {
    let tracer = Tracer::builder().build();
    let traced = tracer.trace::<_>(2, 0, 1, &[], |inp, _| div_pair::<Sym>(inp));
    let inputs = vec![1.0, 1e-11]; // tiny but non-zero divisor
    let got = traced.run_lua(&inputs, &[]).unwrap();
    let want = div_pair::<f64>(&inputs); // 1e11, NOT a trap
    let d = (got[0] - want[0]).abs();
    assert!(
        d / want[0].abs().max(1.0) < 1e-9,
        "got {} want {}",
        got[0],
        want[0]
    );
}

#[test]
fn div_guard_reports_fatal() {
    let tracer = Tracer::builder().build();
    let traced = tracer.trace::<_>(2, 0, 1, &[], |inp, _| div_pair::<Sym>(inp));
    // exact-zero divisor -> fatal slot fires -> Err (branchless, no Lua error/trap)
    assert!(
        traced.run_lua(&[1.0, 0.0], &[]).is_err(),
        "zero divisor -> fatal -> Err"
    );
    // nonzero divisor -> Ok
    assert_eq!(
        traced.run_lua(&[1.0, 2.0], &[]).unwrap(),
        [0.5],
        "nonzero divisor -> Ok"
    );
    let lua = traced.emit_lua();
    assert!(!lua.contains("error("), "no error() for div-by-zero");
    assert!(lua.contains("local eps"), "1e-9 is a named preamble local");
}

fn fms_chain<S>(inp: &[S]) -> Vec<S>
where
    S: Scalar + StandardPart + FromRational + Copy,
{
    let ab = inp[0] * inp[1];
    let cd = inp[2] * inp[3];
    // subtraction-heavy: `cd` is single-use (fnma contracts), `ab` is reused
    // (multi-use gate keeps it) -> mixes contracted and non-contracted paths
    // through the same Lua output. The point is oracle parity, not Fma count.
    vec![(ab - inp[2]) + (inp[3] - cd) - (ab + inp[3])]
}

#[test]
fn lua_matches_f64_fms_heavy() {
    let tracer = Tracer::builder().build();
    let traced = tracer.trace::<_>(4, 0, 1, &[], |inp, _| fms_chain::<Sym>(inp));
    let inputs = vec![0.3, -0.7, 1.1, 0.5];
    let got = traced.run_lua(&inputs, &[]).unwrap();
    let want = fms_chain::<f64>(&inputs);
    let d = (got[0] - want[0]).abs();
    assert!(
        d / want[0].abs().max(1.0) < 1e-12,
        "fms-heavy: lua={} f64={} (abs {d:e})",
        got[0],
        want[0]
    );
}

// Guard case: sin(x) feeding an is_effective_zero check where |x| >> eps. This
// is the path that an UNSOUND `Sin: abs>eps` propagation rule would wrongly prune
// (sin oscillates — abs(x)>eps does NOT give abs(sin x)>eps). With sound code
// both arms are emitted and lua == f64. The negative acceptance step relies on
// THIS test reddening when such a rule is injected.
fn sin_seam<S>(inp: &[S]) -> Vec<S>
where
    S: Scalar + StandardPart + EffectiveZero + FromRational + Copy,
{
    let x = inp[0];
    if x.is_effective_zero() {
        vec![S::from_rational(1, 1)]
    } else {
        let s = x.sin_explicit();
        if s.is_effective_zero() {
            vec![S::from_rational(0, 1)]
        } else {
            vec![S::from_rational(1, 1) / s]
        }
    }
}

#[test]
fn lua_matches_f64_sin_near_zero_seam() {
    let tracer = Tracer::builder().build();
    let traced = tracer.trace::<_>(1, 0, 1, &[], |i, _| sin_seam::<Sym>(i));
    // sin(pi) ~ 1.2e-16 < eps, yet |pi| >> eps: exercises the abs>eps -> sin path.
    let inputs = vec![std::f64::consts::PI];
    let got = traced.run_lua(&inputs, &[]).unwrap();
    let want = sin_seam::<f64>(&inputs);
    assert!(
        (got[0] - want[0]).abs() < 1e-9,
        "sin seam: lua={} f64={}",
        got[0],
        want[0]
    );
}

#[test]
fn lua_matches_f64_near_seam() {
    let tracer = Tracer::builder().build();
    let traced = tracer.trace::<_>(6, 0, 16, &[], |inp, _| exp_coeffs::<Sym>(inp));

    // near-seam: tiny rotation -> u ~ 0 -> Taylor arms
    let inputs = vec![0.2, 0.1, -0.3, 1e-7, -2e-7, 1e-7];
    let got = traced.run_lua(&inputs, &[]).unwrap();
    let want = exp_coeffs::<f64>(&inputs);
    assert_close(got, want);
}
