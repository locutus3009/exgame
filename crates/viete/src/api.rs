// SPDX-License-Identifier: MIT

use crate::engine::{Config, run_trace};
use crate::ir::{BlockId, FatalKind, InputFact, InputRef, Term, Tree};
use crate::sym::Sym;
use std::collections::HashMap;

/// Opaque tracer, configured via [`Tracer::builder`]. Reusable across traces.
pub struct Tracer {
    cfg: Config,
}

pub struct TracerBuilder {
    cfg: Config,
}

impl Tracer {
    pub fn builder() -> TracerBuilder {
        TracerBuilder {
            cfg: Config {
                lua_threshold: 1e-9,
                max_leaves: None,
                fn_name: "trace".into(),
                flatten: false,
                vectorize: false,
                vectorize_lanes: Vec::new(),
                fast_math: false,
            },
        }
    }

    /// Trace `f` (a pure `Fn` over symbolic scalars). `constraints` declares
    /// caller-guaranteed domain predicates on inputs/params (empty slice = none;
    /// see [`InputFact`] / [`InputRef`]). The emitted code does NOT check them —
    /// violating one is shader UB.
    pub fn trace<F>(
        &self,
        inputs: usize,
        params: usize,
        outputs: usize,
        constraints: &[(InputRef, InputFact)],
        f: F,
    ) -> Trace
    where
        F: Fn(&Vec<Sym>, &Vec<Sym>) -> Vec<Sym>,
    {
        let tree = run_trace(&self.cfg, constraints, &f, inputs, params, outputs);
        Trace {
            inputs,
            params,
            outputs,
            tree,
            lua_threshold: self.cfg.lua_threshold,
            fn_name: self.cfg.fn_name.clone(),
            input_facts: constraints.to_vec(),
        }
    }
}

impl TracerBuilder {
    pub fn lua_threshold(mut self, t: f64) -> Self {
        self.cfg.lua_threshold = t;
        self
    }
    pub fn max_leaves(mut self, n: Option<usize>) -> Self {
        self.cfg.max_leaves = n;
        self
    }
    /// Set the name of the emitted entry function (default `"trace"`).
    /// Backend-general: applies to every backend, not just Lua.
    pub fn fn_name(mut self, name: impl Into<String>) -> Self {
        self.cfg.fn_name = name.into();
        self
    }
    /// If-convert every representation-select branch into a single flat,
    /// divergence-free block (the GPU-friendly end state). Off by default, which
    /// keeps the branched decision trie. Both arms are computed unconditionally
    /// and merged by `select`; dead-arm fatals are masked by the path condition.
    pub fn flatten(mut self) -> Self {
        self.cfg.flatten = true;
        self
    }
    /// Run the SLP `vectorize` pass after flattening: collapse lane-parallel
    /// elementwise groups (the AD-gradient axis) into width-≤4 vector instrs.
    /// Off by default. Bit-for-bit (elementwise-only, no reduction reassoc).
    pub fn vectorize(mut self) -> Self {
        self.cfg.vectorize = true;
        self
    }
    /// Enable vectorization and declare caller-known lane-parallel output groups
    /// (indices into the output list), e.g. the per-component gradient lanes of
    /// an AD Jacobian. Each group is tiled to ≤4 and forced through the pass
    /// (AD zero-folding hides these from shape inference). Bit-for-bit.
    pub fn vectorize_lanes(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.cfg.vectorize = true;
        self.cfg.vectorize_lanes = groups;
        self
    }
    /// Run the algebraic `fast_math` pass before vectorize: cancellations like
    /// `(a/b)*b -> a`, `a/b*b/c -> a/c`, `(a*b)/b -> a`, `sqrt(a)*sqrt(a) -> a`.
    /// NOT bit-exact with the f64 carrier (changes rounding) — only
    /// algebraically equal, valid when the consumer accepts a tolerance.
    /// Operates on the flat block, so pairs with `flatten`.
    pub fn fast_math(mut self) -> Self {
        self.cfg.fast_math = true;
        self
    }
    pub fn build(self) -> Tracer {
        Tracer { cfg: self.cfg }
    }
}

/// Opaque trace result: the raw decision trie plus emit/run methods.
#[allow(dead_code)]
pub struct Trace {
    inputs: usize,
    params: usize,
    outputs: usize,
    tree: Tree,
    pub(crate) lua_threshold: f64,
    /// Name of the emitted entry function (backend-general).
    fn_name: String,
    /// Caller-guaranteed predicates (for the run_lua contract check + header).
    input_facts: Vec<(InputRef, InputFact)>,
}

impl Trace {
    pub fn tree(&self) -> &Tree {
        &self.tree
    }
    pub fn leaf_count(&self) -> usize {
        self.tree.leaf_count()
    }

    /// Emit the Lua source for this trace, prefixed with the input contracts.
    pub fn emit_lua(&self) -> String {
        let mut header = String::new();
        for &(r, fact) in &self.input_facts {
            let name = match r {
                InputRef::Input(k) => format!("inp[{}]", k + 1),
                InputRef::Param(j) => format!("param[{}]", j + 1),
            };
            let cond = match fact {
                InputFact::Positive => format!("{name} > 0"),
                InputFact::Negative => format!("{name} < 0"),
                InputFact::NonPositive => format!("{name} <= 0"),
                InputFact::NonNegative => format!("{name} >= 0"),
                InputFact::Nonzero => format!("{name} ~= 0"),
                InputFact::AbsGtEps => format!("abs({name}) > {:e}", self.lua_threshold),
                InputFact::AbsGeOne => format!("abs({name}) >= 1"),
            };
            header.push_str(&format!("-- requires (UB if violated): {cond}\n"));
        }
        header + &crate::lua::emit_lua(&self.tree, self.lua_threshold, &self.fn_name)
    }

    pub fn emit_glsl(
        &self,
        preamble: String,
        pre: String,
        footer: String,
        inputs_map: &HashMap<u32, String>,
        output_map: &HashMap<u32, (String, i8)>,
        fatal_map: &HashMap<u32, String>,
    ) -> String {
        let mut header = String::new();
        header.push_str("// This file has been automatically generated. Do not edit!\n\n");
        header.push_str("#version 460\n\n");
        header.push_str(&preamble);
        header
            + &crate::glsl::emit_glsl(
                &self.tree,
                self.lua_threshold,
                &self.fn_name,
                pre,
                footer,
                &crate::glsl::Boundary {
                    inputs: inputs_map,
                    outputs: output_map,
                    fatals: fatal_map,
                },
            )
    }

    /// Execute the emitted Lua on concrete inputs + params via an in-process Lua
    /// VM, returning the OUTPUTS outputs as f64 (no text round-trip). `params` is
    /// a runtime-checked slice (its length must equal `PARAMS`), so a joint's
    /// `Vec<f64>` feeds straight in.
    #[cfg(feature = "lua")]
    pub fn run_lua(&self, inputs: &[f64], params: &[f64]) -> Result<Vec<f64>, String> {
        assert_eq!(
            params.len(),
            self.params,
            "run_lua: param slice length {} != PARAMS {}",
            params.len(),
            self.params
        );
        // Contract check (harness-level; the emitted shader carries none). Each
        // declared fact MUST hold on the right array, else the dropped guards
        // make the result garbage — fail loudly instead.
        for &(r, fact) in &self.input_facts {
            let (name, x) = match r {
                InputRef::Input(k) => (format!("inp[{k}]"), inputs[k as usize]),
                InputRef::Param(j) => (format!("param[{j}]"), params[j as usize]),
            };
            match fact {
                InputFact::Positive => {
                    assert!(x > 0.0, "{name} violates Positive contract (got {x})");
                }
                InputFact::Negative => {
                    assert!(x < 0.0, "{name} violates Negative contract (got {x})");
                }
                InputFact::NonPositive => {
                    assert!(x <= 0.0, "{name} violates NonPositive contract (got {x})");
                }
                InputFact::NonNegative => {
                    assert!(x >= 0.0, "{name} violates NonNegative contract (got {x})");
                }
                InputFact::Nonzero => {
                    assert!(x != 0.0, "{name} violates Nonzero contract (got {x})");
                }
                InputFact::AbsGtEps => {
                    assert!(
                        x.abs() > self.lua_threshold,
                        "{name} violates AbsGtEps contract (|{x}| <= {:e})",
                        self.lua_threshold
                    );
                }
                InputFact::AbsGeOne => {
                    assert!(
                        x.abs() >= 1.0,
                        "{name} violates AbsGeOne contract (|{x}| < 1)"
                    );
                }
            }
        }
        let lua = mlua::Lua::new();
        lua.load(self.emit_lua()).exec().expect("lua load/exec");
        let f: mlua::Function = lua.globals().get(self.fn_name.as_str()).expect("entry fn");
        let inp = lua.create_table().expect("table");
        for (i, v) in inputs.iter().enumerate() {
            inp.set(i + 1, *v).expect("set input");
        }
        let param = lua.create_table().expect("table");
        for (j, v) in params.iter().enumerate() {
            param.set(j + 1, *v).expect("set param");
        }
        // A residual NonInvertible `error()` (only on invalid data) surfaces here.
        let (out, fatal): (mlua::Table, mlua::Table) = match f.call((inp, param)) {
            Ok(v) => v,
            Err(e) => return Err(format!("viete: shader trapped: {e}")),
        };
        // Any non-zero fatal slot -> the whole result is invalid (filter it out).
        let mut fired = Vec::new();
        for i in 1..=fatal.raw_len() {
            let v = fatal
                .get::<Option<f64>>(i)
                .expect("fatal slot")
                .unwrap_or(0.0);
            if v != 0.0 {
                fired.push(i - 1);
            }
        }
        if !fired.is_empty() {
            return Err(format!(
                "viete: division-by-zero at fatal site(s) {fired:?}"
            ));
        }
        Ok((0..self.outputs)
            .map(|i| out.get::<f64>(i + 1).expect("get output"))
            .collect())
    }
    /// Human-readable text dump of the decision trie.
    pub fn pretty(&self) -> String {
        self.tree.pretty()
    }

    /// Count of `Fatal` leaves whose kind is `Unexpected` (tracer-bug marker).
    pub fn unexpected_count(&self) -> usize {
        self.tree
            .fatal_leaves()
            .iter()
            .filter(|(_, k)| matches!(k, FatalKind::Unexpected(_)))
            .count()
    }
    pub fn fatal_leaves(&self) -> Vec<(BlockId, FatalKind)> {
        self.tree.fatal_leaves()
    }
    /// Does any leaf trap? (Used by structural tests.)
    pub fn has_fatal(&self) -> bool {
        self.tree
            .blocks
            .iter()
            .any(|b| matches!(b.term, Term::Fatal { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Term;
    use peano::prelude::*;

    #[cfg(feature = "lua")]
    #[test]
    fn select_is_branchless_kronecker_delta() {
        let tracer = Tracer::builder().build();
        // selector is the single param; lane 0. INPUTS=0, PARAMS=1, OUTPUTS=1.
        let traced: Trace = tracer.trace(0, 1, 1, &[], |_i, p| vec![Sym::select(p[0], 0)]);
        // straight-line: exactly one leaf == no decision-trie branch (the helper's
        // own `if` is module-scope, not a trie fork).
        assert_eq!(traced.leaf_count(), 1, "select must not fork the trie");
        let code = traced.emit_lua();
        assert!(
            code.contains("local function select("),
            "select is a module-scope helper (like fma):\n{code}"
        );
        // the 1/0 are pooled constant references, NOT inline literals in the call.
        assert!(
            code.contains("(1.0/1.0)") && code.contains("(0.0/1.0)"),
            "select's 1 and 0 are pooled constants:\n{code}"
        );
        assert!(
            !code.contains("and 1.0 or 0.0"),
            "no inline literal delta expression:\n{code}"
        );
        // delta semantics
        assert_eq!(
            traced.run_lua(&[], &[0.0]).unwrap(),
            [1.0],
            "selector == lane -> 1"
        );
        assert_eq!(
            traced.run_lua(&[], &[3.0]).unwrap(),
            [0.0],
            "selector != lane -> 0"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn select_distinct_lanes_form_one_hot() {
        let tracer = Tracer::builder().build();
        // two lanes off the same selector -> one-hot pair
        let traced: Trace = tracer.trace(0, 1, 2, &[], |_i, p| {
            vec![Sym::select(p[0], 0), Sym::select(p[0], 1)]
        });
        assert_eq!(traced.run_lua(&[], &[0.0]).unwrap(), [1.0, 0.0]);
        assert_eq!(traced.run_lua(&[], &[1.0]).unwrap(), [0.0, 1.0]);
        assert_eq!(traced.run_lua(&[], &[2.0]).unwrap(), [0.0, 0.0]);
    }

    #[test]
    fn straight_line_is_one_return_leaf() {
        let tracer = Tracer::builder().build();
        // no is_effective_zero anywhere -> exactly one path, one Return leaf
        let traced: Trace = tracer.trace(2, 0, 3, &[], |inp, _| {
            let a = inp[0];
            let b = inp[1];
            vec![a * b + a, a - b, b]
        });
        assert_eq!(traced.leaf_count(), 1);
        let root = &traced.tree().blocks[0];
        match &root.term {
            Term::Return { outputs, .. } => assert_eq!(outputs.len(), 3),
            other => panic!("expected Return, got {other:?}"),
        }
    }

    #[test]
    fn input_abs_gt_eps_suppresses_div_guard() {
        use crate::ir::{FatalKind, InputFact, InputRef};
        // inp[1] declared abs>eps -> inp[0]/inp[1] forks no div-by-zero guard.
        let tracer = Tracer::builder().build();
        let traced: Trace = tracer.trace(
            2,
            0,
            1,
            &[(InputRef::Input(1), InputFact::AbsGtEps)],
            |i, _| vec![i[0] / i[1]],
        );
        assert!(
            !traced.tree().blocks.iter().any(|b| matches!(
                b.term,
                Term::Fatal {
                    kind: FatalKind::DivByZero
                }
            )),
            "abs>eps input -> divisor nonzero -> no div guard"
        );
    }

    #[test]
    fn input_abs_gt_eps_determines_effective_zero() {
        use crate::ir::{InputFact, InputRef};
        // is_effective_zero(inp[0]) is determined false (no branch) when inp[0] abs>eps.
        let tracer = Tracer::builder().build();
        let traced: Trace = tracer.trace(
            1,
            0,
            1,
            &[(InputRef::Input(0), InputFact::AbsGtEps)],
            |i, _| {
                let _x = i[0].is_effective_zero();
                vec![i[0]]
            },
        );
        let branches = traced
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        assert_eq!(
            branches, 0,
            "abs>eps input determines the EffectiveZero check"
        );
    }

    #[test]
    fn abs_ge_one_is_multiplicative() {
        use crate::ir::{InputFact, InputRef};
        // inp[0] >= 1 -> inp[0]^2 >= 1 (multiplicative) -> abs>eps -> the
        // EffectiveZero check on inp[0]^2 is determined false (no branch).
        // This is what AbsGtEps could NOT do (abs>eps not multiplicative).
        let tracer = Tracer::builder().build();
        let traced: Trace = tracer.trace(
            1,
            0,
            1,
            &[(InputRef::Input(0), InputFact::AbsGeOne)],
            |i, _| {
                let sq = i[0].powi_explicit(2);
                let _x = sq.is_effective_zero();
                vec![sq]
            },
        );
        let branches = traced
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        assert_eq!(
            branches, 0,
            "inp^2 >= 1 determines its effective-zero check"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn run_lua_rejects_out_of_contract_input() {
        use crate::ir::{InputFact, InputRef};
        let tracer = Tracer::builder().build();
        // inp[1] declared abs>eps; the emitted code dropped the div guard, so the
        // harness must reject warp=0 rather than silently produce garbage.
        let traced: Trace = tracer.trace(
            2,
            0,
            1,
            &[(InputRef::Input(1), InputFact::AbsGtEps)],
            |i, _| vec![i[0] / i[1]],
        );
        // conforming: ok
        traced.run_lua(&[1.0, 2.0], &[]).unwrap();
        // violating: must panic (harness contract assert)
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            traced.run_lua(&[1.0, 0.0], &[])
        }));
        assert!(
            r.is_err(),
            "run_lua must reject an input that violates its declared fact"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn run_lua_rejects_abs_ge_one_violation() {
        use crate::ir::{InputFact, InputRef};
        // inp[0] declared >=1; the trace dropped the effective-zero guard on inp[0]^2,
        // so warp<1 is shader UB. The harness must reject it (documents the boundary).
        let tracer = Tracer::builder().build();
        let traced: Trace = tracer.trace(
            1,
            0,
            1,
            &[(InputRef::Input(0), InputFact::AbsGeOne)],
            |i, _| {
                let sq = i[0].powi_explicit(2);
                // a guarded division that the AbsGeOne fact suppresses
                vec![Sym::ONE / sq]
            },
        );
        traced.run_lua(&[1.5], &[]).unwrap(); // conforming (>=1): ok
        // 0.5 is positive and != 0, but < 1 -> violates the AbsGeOne contract.
        let r =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| traced.run_lua(&[0.5], &[])));
        assert!(
            r.is_err(),
            "warp<1 is out of the AbsGeOne contract -> harness rejects it"
        );
    }
}
