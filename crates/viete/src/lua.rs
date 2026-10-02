// SPDX-License-Identifier: MIT

use crate::ir::{Block, BlockId, BranchTest, CondSign, Instr, Operand, Term, Tree};

/// Emit the whole trie as a Lua function `<fn_name>(inp)` returning a table of K values.
/// `EffectiveZero` branches emit `if math.abs(cond) < threshold then ... else ... end`;
/// `ExactZero` branches (Div guards) emit `if cond == 0.0 then ... else ... end`.
pub fn emit_lua(tree: &Tree, threshold: f64, fn_name: &str) -> String {
    let mut out = String::new();
    // two-rounding helper at module scope: matches the f64 carrier exactly. A true
    // one-rounding FMA appears only in the SPIR-V backend (math.fma is absent from Lua 5.4).
    if tree_has_fma(tree) {
        out.push_str("local function fma(a, b, c) return a * b + c end\n");
    }
    // index-match pick: returns `a` when `x == lane`, else `b` (a/b are pooled
    // constants at the call site). Module-scope helper, like fma; the `if` lives
    // here, not in the decision trie.
    if tree_has_select_eq(tree) {
        out.push_str(
            "local function select(x, lane, a, b) if x == lane then return a else return b end end\n",
        );
    }
    if tree_has_select_lt(tree) {
        out.push_str(
            "local function select_lt(x, y, a, b) if x < y then return a else return b end end\n",
        );
    }
    // vector helpers (only when the `vectorize` pass emitted vec instrs). Vectors
    // are Lua arrays {x,y,..}; these are the verification mirror of the eventual
    // SPIR-V vec ops. vsel/vsel_lt take a SCALAR (broadcast) condition.
    if tree_has_vec(tree) {
        out.push_str(concat!(
            "local function splat(s, w) local t = {} for i = 1, w do t[i] = s end return t end\n",
            "local function vneg(a) local t = {} for i = 1, #a do t[i] = -a[i] end return t end\n",
            "local function vadd(a, b) local t = {} for i = 1, #a do t[i] = a[i] + b[i] end return t end\n",
            "local function vsub(a, b) local t = {} for i = 1, #a do t[i] = a[i] - b[i] end return t end\n",
            "local function vmul(a, b) local t = {} for i = 1, #a do t[i] = a[i] * b[i] end return t end\n",
            "local function vdiv(a, b) local t = {} for i = 1, #a do t[i] = a[i] / b[i] end return t end\n",
            "local function vfma(a, b, c) local t = {} for i = 1, #a do t[i] = a[i] * b[i] + c[i] end return t end\n",
            "local function vsel(x, y, a, b) local t = {} for i = 1, #a do if x[i] == y[i] then t[i] = a[i] else t[i] = b[i] end end return t end\n",
            "local function vsel_lt(x, y, a, b) local t = {} for i = 1, #a do if x[i] < y[i] then t[i] = a[i] else t[i] = b[i] end end return t end\n\n",
        ));
    }
    out.push_str(&format!("function {fn_name}(inp, param)\n"));
    // effective-zero threshold as a named codegen constant (SPIR-V gets its own).
    out.push_str(&format!("  local eps = {threshold:e}\n"));
    // constant preamble: only constants actually referenced are declared (indices
    // keep their original numbering, so Operand::Const(i) stays valid). Constants
    // orphaned by peepholes — e.g. `x * -1` rewritten to `-x` drops the -1 — vanish.
    let used: std::collections::HashSet<u32> = tree.collect_consts().into_iter().collect();
    for (i, r) in tree.consts.iter().enumerate() {
        if !used.contains(&(i as u32)) {
            continue;
        }
        out.push_str(&format!(
            "  local c{i} = ({}.0/{}.0)\n",
            r.numer(),
            r.denom()
        ));
    }
    // SSA values go in a table, not individual locals: Lua caps locals at 200 per
    // function, and a large trace (e.g. a 24-DOF Jacobian) has thousands of slots.
    // Table fields don't count against that cap, so the backend scales.
    out.push_str("  local v = {}\n");
    emit_block(tree, tree.root, 1, &mut out);
    out.push_str("end\n");
    out
}

fn tree_has_fma(tree: &Tree) -> bool {
    tree.blocks
        .iter()
        .any(|b| b.instrs.iter().any(|i| matches!(i, Instr::Fma { .. })))
}

fn tree_has_select_eq(tree: &Tree) -> bool {
    tree.blocks.iter().any(|b| {
        b.instrs
            .iter()
            .any(|i| matches!(i, Instr::Select(_, crate::ir::CmpKind::Eq, ..)))
    })
}

fn tree_has_select_lt(tree: &Tree) -> bool {
    tree.blocks.iter().any(|b| {
        b.instrs
            .iter()
            .any(|i| matches!(i, Instr::Select(_, crate::ir::CmpKind::Lt, ..)))
    })
}

fn tree_has_vec(tree: &Tree) -> bool {
    tree.blocks.iter().any(|b| {
        b.instrs.iter().any(|i| {
            matches!(
                i,
                Instr::Splat(..)
                    | Instr::Pack(..)
                    | Instr::VNeg(..)
                    | Instr::VAdd(..)
                    | Instr::VSub(..)
                    | Instr::VMul(..)
                    | Instr::VDiv(..)
                    | Instr::VFma(..)
                    | Instr::VSelect(..)
                    | Instr::Extract(..)
            )
        })
    })
}

fn operand(op: &Operand, inputs: u32) -> String {
    match op {
        Operand::Const(i) => format!("c{i}"),
        Operand::Input(i) => {
            if *i < inputs {
                format!("inp[{}]", i + 1)
            } else {
                format!("param[{}]", i - inputs + 1)
            }
        }
        Operand::Instr(s) => format!("v[{s}]"),
        Operand::Eps => "eps".to_string(),
    }
}

fn emit_instr(ins: &Instr, indent: &str, inputs: u32, out: &mut String) {
    let (slot, expr) = match ins {
        Instr::Add(s, a, b) => (
            *s,
            format!("{} + {}", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::Sub(s, a, b) => (
            *s,
            format!("{} - {}", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::Mul(s, a, b) => (
            *s,
            format!("{} * {}", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::Div(s, a, b) => (
            *s,
            format!("{} / {}", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::Neg(s, a) => (*s, format!("-({})", operand(a, inputs))),
        Instr::Sqrt(s, a) => (*s, format!("math.sqrt({})", operand(a, inputs))),
        Instr::Sin(s, a) => (*s, format!("math.sin({})", operand(a, inputs))),
        Instr::Cos(s, a) => (*s, format!("math.cos({})", operand(a, inputs))),
        Instr::Exp(s, a) => (*s, format!("math.exp({})", operand(a, inputs))),
        Instr::Ln(s, a) => (*s, format!("math.log({})", operand(a, inputs))),
        Instr::Abs(s, a) => (*s, format!("math.abs({})", operand(a, inputs))),
        Instr::Atan2(s, y, x) => (
            *s,
            format!("math.atan({}, {})", operand(y, inputs), operand(x, inputs)),
        ),
        Instr::Powi(s, a, n) => (
            *s,
            format!("({})^({})", operand(a, inputs), operand(n, inputs)),
        ),
        Instr::Powf(s, a, b) => (
            *s,
            format!("({})^({})", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::Select(s, cmp, a, b, on_match, on_miss) => {
            let helper = match cmp {
                crate::ir::CmpKind::Eq => "select",
                crate::ir::CmpKind::Lt => "select_lt",
            };
            (
                *s,
                format!(
                    "{helper}({}, {}, {}, {})",
                    operand(a, inputs),
                    operand(b, inputs),
                    operand(on_match, inputs),
                    operand(on_miss, inputs),
                ),
            )
        }
        Instr::Fma {
            s,
            neg_prod,
            a,
            b,
            c,
            neg_c,
        } => {
            let a_s = if *neg_prod {
                format!("-({})", operand(a, inputs))
            } else {
                operand(a, inputs)
            };
            let c_s = if *neg_c {
                format!("-({})", operand(c, inputs))
            } else {
                operand(c, inputs)
            };
            (*s, format!("fma({}, {}, {})", a_s, operand(b, inputs), c_s))
        }
        // ---- vector instructions ----
        Instr::Splat(s, w, a) => (*s, format!("splat({}, {w})", operand(a, inputs))),
        Instr::Pack(s, ops) => {
            let elems: Vec<String> = ops.iter().map(|o| operand(o, inputs)).collect();
            (*s, format!("{{{}}}", elems.join(", ")))
        }
        Instr::VNeg(s, a) => (*s, format!("vneg({})", operand(a, inputs))),
        Instr::VAdd(s, a, b) => (
            *s,
            format!("vadd({}, {})", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::VSub(s, a, b) => (
            *s,
            format!("vsub({}, {})", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::VMul(s, a, b) => (
            *s,
            format!("vmul({}, {})", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::VDiv(s, a, b) => (
            *s,
            format!("vdiv({}, {})", operand(a, inputs), operand(b, inputs)),
        ),
        Instr::VFma(s, a, b, c) => (
            *s,
            format!(
                "vfma({}, {}, {})",
                operand(a, inputs),
                operand(b, inputs),
                operand(c, inputs)
            ),
        ),
        Instr::VSelect(s, cmp, x, y, a, b) => {
            let helper = match cmp {
                crate::ir::CmpKind::Eq => "vsel",
                crate::ir::CmpKind::Lt => "vsel_lt",
            };
            (
                *s,
                format!(
                    "{helper}({}, {}, {}, {})",
                    operand(x, inputs),
                    operand(y, inputs),
                    operand(a, inputs),
                    operand(b, inputs),
                ),
            )
        }
        Instr::Extract(s, v, lane) => (*s, format!("{}[{}]", operand(v, inputs), lane + 1)),
    };
    out.push_str(&format!("{indent}v[{slot}] = {expr}\n"));
}

fn emit_block(tree: &Tree, id: BlockId, depth: usize, out: &mut String) {
    let indent = "  ".repeat(depth);
    let inputs = tree.inputs;
    let block: &Block = &tree.blocks[id.0 as usize];
    for ins in &block.instrs {
        emit_instr(ins, &indent, inputs, out);
    }
    match &block.term {
        Term::Branch {
            cond,
            test,
            cond_sign,
            on_true,
            on_false,
        } => {
            let cmp = match test {
                BranchTest::EffectiveZero => match cond_sign {
                    CondSign::NonNeg => format!("{} < eps", operand(cond, inputs)),
                    CondSign::NonPos => format!("-{} < eps", operand(cond, inputs)),
                    CondSign::Unknown => {
                        format!("math.abs({}) < eps", operand(cond, inputs))
                    }
                },
                BranchTest::ExactZero => format!("{} == 0.0", operand(cond, inputs)),
            };
            out.push_str(&format!("{indent}if {cmp} then\n"));
            emit_block(tree, *on_true, depth + 1, out);
            out.push_str(&format!("{indent}else\n"));
            emit_block(tree, *on_false, depth + 1, out);
            out.push_str(&format!("{indent}end\n"));
        }
        Term::Return { outputs, fatals } => {
            let outs: Vec<String> = outputs.iter().map(|o| operand(o, inputs)).collect();
            let fats: Vec<String> = fatals.iter().map(|o| operand(o, inputs)).collect();
            out.push_str(&format!(
                "{indent}return {{{}}}, {{{}}}\n",
                outs.join(", "),
                fats.join(", ")
            ));
        }
        Term::Fatal { .. } => {
            out.push_str(&format!("{indent}error(\"viete: div by zero\")\n"));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::api::Tracer;
    use crate::sym::Sym;
    use peano::prelude::*;

    #[test]
    fn select_lt_lowers_with_eps() {
        use crate::ir::{Block, BlockId, CmpKind, Instr, Operand, Term, Tree};
        let tree = Tree {
            inputs: 1,
            params: 0,
            consts: vec![
                num_rational::Rational32::new(0, 1),
                num_rational::Rational32::new(1, 1),
            ],
            blocks: vec![Block {
                instrs: vec![
                    Instr::Abs(0, Operand::Input(0)),
                    Instr::Select(
                        1,
                        CmpKind::Lt,
                        Operand::Instr(0),
                        Operand::Eps,
                        Operand::Const(1),
                        Operand::Const(0),
                    ),
                ],
                term: Term::Return {
                    outputs: vec![],
                    fatals: vec![Operand::Instr(1)],
                },
            }],
            root: BlockId(0),
        };
        let code = super::emit_lua(&tree, 1e-9, "trace");
        assert!(
            code.contains("local function select_lt("),
            "select_lt helper: {code}"
        );
        assert!(
            code.contains("select_lt(v[0], eps, c1, c0)"),
            "Lt call: {code}"
        );
        assert!(code.contains("local eps = 1e-9"), "eps local");
    }

    #[test]
    fn abs_lowers_to_math_abs() {
        use crate::ir::{Block, BlockId, Instr, Operand, Term, Tree};
        let tree = Tree {
            inputs: 1,
            params: 0,
            consts: vec![],
            blocks: vec![Block {
                instrs: vec![Instr::Abs(0, Operand::Input(0))],
                term: Term::Return {
                    outputs: vec![Operand::Instr(0)],
                    fatals: vec![],
                },
            }],
            root: BlockId(0),
        };
        assert!(super::emit_lua(&tree, 1e-9, "trace").contains("math.abs(inp[1])"));
    }

    #[test]
    fn nonneg_branch_lowers_without_abs() {
        use crate::ir::{Block, BlockId, BranchTest, CondSign, Instr, Operand, Term, Tree};
        let root = Block {
            instrs: vec![Instr::Mul(0, Operand::Input(0), Operand::Input(0))],
            term: Term::Branch {
                cond: Operand::Instr(0),
                test: BranchTest::EffectiveZero,
                cond_sign: CondSign::NonNeg,
                on_true: BlockId(1),
                on_false: BlockId(2),
            },
        };
        let arm = |v| Block {
            instrs: vec![],
            term: Term::Return {
                outputs: vec![v],
                fatals: vec![],
            },
        };
        let tree = Tree {
            inputs: 1,
            params: 0,
            consts: vec![],
            blocks: vec![root, arm(Operand::Input(0)), arm(Operand::Instr(0))],
            root: BlockId(0),
        };
        let code = super::emit_lua(&tree, 1e-9, "trace");
        assert!(code.contains("v[0] < eps"), "nonneg branch: {code}");
        assert!(!code.contains("math.abs"), "no abs for nonneg: {code}");
    }

    #[test]
    fn param_operand_lowers_to_param_table() {
        use crate::ir::{Block, BlockId, Instr, Operand, Term, Tree};
        // inputs = 2, params = 2 -> Input(0),Input(1) are inp; Input(2),Input(3) are param.
        let tree = Tree {
            inputs: 2,
            params: 2,
            consts: vec![],
            blocks: vec![Block {
                instrs: vec![Instr::Mul(0, Operand::Input(0), Operand::Input(3))],
                term: Term::Return {
                    outputs: vec![Operand::Instr(0)],
                    fatals: vec![],
                },
            }],
            root: BlockId(0),
        };
        let code = super::emit_lua(&tree, 1e-9, "trace");
        assert!(
            code.contains("function trace(inp, param)"),
            "two-table entry: {code}"
        );
        assert!(code.contains("inp[1]"), "Input(0) -> inp[1]: {code}");
        assert!(
            code.contains("param[2]"),
            "Input(3) -> param[3-2+1]=param[2]: {code}"
        );
    }

    #[test]
    fn emits_function_with_const_preamble_and_return_table() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(2, 0, 1, &[], |inp, _| {
            // uses a non-folding constant 1/2 as a multiplier -> pooled
            let half = Sym::from_rational(1, 2);
            vec![(inp[0] + inp[1]) * half]
        });
        let code = traced.emit_lua();
        assert!(code.contains("function trace(inp, param)"));
        assert!(code.contains("local c0 ="), "const preamble present");
        assert!(code.contains("return {"), "table return");
        assert!(
            code.contains("inp[1]") && code.contains("inp[2]"),
            "1-based inputs"
        );
    }

    #[test]
    fn fn_name_sets_the_emitted_function_name() {
        let tracer = Tracer::builder().fn_name("motor_exp").build();
        let traced = tracer.trace::<_>(1, 0, 1, &[], |inp, _| vec![inp[0] + inp[0]]);
        let code = traced.emit_lua();
        assert!(
            code.contains("function motor_exp(inp, param)"),
            "custom name used"
        );
        assert!(!code.contains("function trace("), "default name gone");
    }

    #[test]
    fn fma_lowers_to_helper_with_signs() {
        let tracer = Tracer::builder().build();
        // a*b - c -> fma(a, b, -(c))
        let traced = tracer.trace::<_>(3, 0, 1, &[], |i, _| vec![i[0] * i[1] - i[2]]);
        let code = traced.emit_lua();
        assert!(
            code.contains("local function fma(a, b, c) return a * b + c end"),
            "helper at module scope"
        );
        assert!(
            code.starts_with("local function fma"),
            "helper precedes the main function"
        );
        assert!(
            code.contains("fma(inp[1], inp[2], -(inp[3]))"),
            "fms renders negated addend, got:\n{code}"
        );
        assert!(
            !code.contains(" - inp[3]"),
            "subtraction folded into the fma call"
        );
    }

    #[test]
    fn no_fma_helper_when_no_fma() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(2, 0, 1, &[], |i, _| vec![i[0] + i[1]]);
        let code = traced.emit_lua();
        assert!(
            !code.contains("local function fma"),
            "no helper when no Fma node"
        );
    }
}
