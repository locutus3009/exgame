// SPDX-License-Identifier: MIT

use num_rational::Rational32;

/// A reference to a value: a pooled constant, a closure input, or an instruction result.
/// The three index spaces are independent (distinguished by variant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    Const(u32),
    Input(u32),
    Instr(u32),
    /// The codegen effective-zero threshold (`eps`). A leaf operand, like Const;
    /// created post-build by the Fatal-arm collapse pass, lowered as `eps`.
    Eps,
}

/// Comparison used by [`Instr::Select`]. `Eq`: `a == b` (index match / Kronecker
/// delta). `Lt`: `a < b` (e.g. `abs(cond) < eps` for a fatal status).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpKind {
    Eq,
    Lt,
}

/// One straight-line operation. First field of each variant is the result slot
/// (always a fresh `Instr` slot). `recip(x)` is lowered to `Div(slot, ONE, x)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    Add(u32, Operand, Operand),
    Sub(u32, Operand, Operand),
    Mul(u32, Operand, Operand),
    Div(u32, Operand, Operand),
    Neg(u32, Operand),
    Sqrt(u32, Operand),
    Sin(u32, Operand),
    Cos(u32, Operand),
    Exp(u32, Operand),
    Ln(u32, Operand),
    Abs(u32, Operand),
    Atan2(u32, Operand, Operand),
    Powi(u32, Operand, Operand), // exponent is a Const operand (Rational(n,1))
    Powf(u32, Operand, Operand),
    /// `on_match` if `a <cmp> b` else `on_miss`. `Eq`: index match / Kronecker
    /// delta (a==b, a=selector b=lane); `Lt`: a < b (abs(cond) < eps fatal). The
    /// pooled 1/0 are constant references, not literals. Straight-line, never
    /// forks; lowers to a Lua `select`/`select_lt` helper / SPIR-V `OpSelect`.
    Select(u32, CmpKind, Operand, Operand, Operand, Operand),
    /// Fused multiply-add: (neg_prod ? -(a*b) : a*b) + (neg_c ? -c : c).
    /// `b` never carries a sign (a factor negation folds onto `neg_prod`).
    Fma {
        s: u32,
        neg_prod: bool,
        a: Operand,
        b: Operand,
        c: Operand,
        neg_c: bool,
    },
    // ---- vector instructions (emitted only by the `vectorize` pass; width<=4) ----
    /// Broadcast a scalar operand to all `w` lanes.
    Splat(u32, u32, Operand),
    /// Gather 2..4 scalar operands into a vector (width = len).
    Pack(u32, Vec<Operand>),
    /// Lanewise negate.
    VNeg(u32, Operand),
    /// Lanewise add / sub / mul / div.
    VAdd(u32, Operand, Operand),
    VSub(u32, Operand, Operand),
    VMul(u32, Operand, Operand),
    VDiv(u32, Operand, Operand),
    /// Lanewise fused multiply-add `a*b + c` (no sign flags; signs are folded
    /// into operands upstream — elementwise mirror of the scalar carrier).
    VFma(u32, Operand, Operand, Operand),
    /// Lanewise select with a SCALAR (broadcast) condition `x <cmp> y`: picks
    /// vector `a` where it holds, else vector `b`.
    VSelect(u32, CmpKind, Operand, Operand, Operand, Operand),
    /// Read one lane of a vector as a scalar.
    Extract(u32, Operand, u32),
}

/// Why a path trapped. `Unexpected` carries the panic message and marks a tracer bug.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FatalKind {
    DivByZero,
    NonInvertible,
    Unexpected(String),
}

/// Which comparison a branch tests its condition with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BranchTest {
    EffectiveZero, // abs(cond) < threshold  (representation-select / is_effective_zero)
    ExactZero,     // cond == 0              (division-by-zero guard)
}

/// Known sign of an `EffectiveZero` branch condition, stamped at fork time so
/// the post-build lowering can simplify `abs(cond)`. `NonNeg` subsumes positive,
/// `NonPos` subsumes negative — three-way is all the abs site needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CondSign {
    Unknown,
    NonNeg,
    NonPos,
}

/// A domain predicate the caller guarantees about a trace input. Seeded at the
/// root (before any fork) and propagated through the monotone fact lattice.
/// Strength order: `AbsGeOne ⟹ AbsGtEps ⟹ Nonzero`. `AbsGeOne` (|x|≥1) is the
/// only one that survives a runtime `Mul` (it is multiplicative), so it is what
/// collapses recip forks on products like `warp²`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFact {
    Positive,
    Negative,
    NonPositive,
    NonNegative,
    Nonzero,
    AbsGtEps,
    AbsGeOne,
}

/// A constraint target: a true input or a param (boundary sugar). Resolved to
/// an absolute input index (`Param(j) -> INPUTS + j`) before tracing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRef {
    Input(u32),
    Param(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockId(pub u32);

/// A basic block: straight-line instructions then a terminator.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub instrs: Vec<Instr>,
    pub term: Term,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    Branch {
        cond: Operand,
        test: BranchTest,
        cond_sign: CondSign,
        on_true: BlockId,
        on_false: BlockId,
    },
    Return {
        outputs: Vec<Operand>,
        /// Implicit per-path fatal-status slots (`(divisor==0)?1:0` for guarded
        /// divisions). Checked by the host; empty when no guarded division ran.
        fatals: Vec<Operand>,
    },
    Fatal {
        kind: FatalKind,
    },
}

/// The raw decision trie. `blocks[root.0]` is the entry block.
#[derive(Clone, Debug)]
pub struct Tree {
    /// INPUTS — the split point: operands with index `< inputs` are true
    /// inputs, `>= inputs` are params (`param[idx - inputs]`).
    pub inputs: u32,
    /// PARAMS — number of param slots (`= total input indices - inputs`).
    pub params: u32,
    pub consts: Vec<Rational32>,
    pub blocks: Vec<Block>,
    pub root: BlockId,
}

impl Tree {
    /// Number of leaf terminators (`Return` or `Fatal`).
    pub fn leaf_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| !matches!(b.term, Term::Branch { .. }))
            .count()
    }

    /// Const-pool indices actually referenced anywhere in the tree (sorted, deduped).
    /// Sanity / future-DCE basis (with intern-on-use this equals the full pool).
    pub fn collect_consts(&self) -> Vec<u32> {
        let mut seen = std::collections::BTreeSet::new();
        let mut note = |op: &Operand| {
            if let Operand::Const(i) = op {
                seen.insert(*i);
            }
        };
        for b in &self.blocks {
            for ins in &b.instrs {
                for op in instr_operands(ins) {
                    note(op);
                }
            }
            match &b.term {
                Term::Branch { cond, .. } => note(cond),
                Term::Return { outputs, .. } => outputs.iter().for_each(&mut note),
                Term::Fatal { .. } => {}
            }
        }
        seen.into_iter().collect()
    }

    /// All `Fatal` leaves with their kind.
    pub fn fatal_leaves(&self) -> Vec<(BlockId, FatalKind)> {
        self.blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match &b.term {
                Term::Fatal { kind } => Some((BlockId(i as u32), kind.clone())),
                _ => None,
            })
            .collect()
    }

    /// Human-readable preorder dump of the trie (for inspection/tests).
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.pretty_block(self.root, 0, &mut out);
        out
    }

    fn pretty_block(&self, id: BlockId, depth: usize, out: &mut String) {
        let pad = "  ".repeat(depth);
        let b = &self.blocks[id.0 as usize];
        out.push_str(&format!("{pad}block {}:\n", id.0));
        for ins in &b.instrs {
            out.push_str(&format!("{pad}  {ins:?}\n"));
        }
        match &b.term {
            Term::Branch {
                cond,
                test,
                on_true,
                on_false,
                ..
            } => {
                out.push_str(&format!("{pad}  branch {cond:?} [{test:?}]:\n"));
                out.push_str(&format!("{pad}  true ->\n"));
                self.pretty_block(*on_true, depth + 2, out);
                out.push_str(&format!("{pad}  false ->\n"));
                self.pretty_block(*on_false, depth + 2, out);
            }
            Term::Return { outputs, fatals } => {
                out.push_str(&format!("{pad}  return {outputs:?} fatals {fatals:?}\n"))
            }
            Term::Fatal { kind } => out.push_str(&format!("{pad}  fatal {kind:?}\n")),
        }
    }
}

/// The result slot an instruction defines (its first field).
pub(crate) fn instr_result_slot(ins: &Instr) -> u32 {
    match ins {
        Instr::Add(s, ..)
        | Instr::Sub(s, ..)
        | Instr::Mul(s, ..)
        | Instr::Div(s, ..)
        | Instr::Neg(s, ..)
        | Instr::Sqrt(s, ..)
        | Instr::Sin(s, ..)
        | Instr::Cos(s, ..)
        | Instr::Exp(s, ..)
        | Instr::Ln(s, ..)
        | Instr::Abs(s, ..)
        | Instr::Atan2(s, ..)
        | Instr::Powi(s, ..)
        | Instr::Powf(s, ..)
        | Instr::Select(s, ..)
        | Instr::Fma { s, .. }
        | Instr::Splat(s, ..)
        | Instr::Pack(s, ..)
        | Instr::VNeg(s, ..)
        | Instr::VAdd(s, ..)
        | Instr::VSub(s, ..)
        | Instr::VMul(s, ..)
        | Instr::VDiv(s, ..)
        | Instr::VFma(s, ..)
        | Instr::VSelect(s, ..)
        | Instr::Extract(s, ..) => *s,
    }
}

/// Borrow the operand list of an instruction (result slot excluded).
pub(crate) fn instr_operands(ins: &Instr) -> Vec<&Operand> {
    match ins {
        Instr::Add(_, a, b)
        | Instr::Sub(_, a, b)
        | Instr::Mul(_, a, b)
        | Instr::Div(_, a, b)
        | Instr::Atan2(_, a, b)
        | Instr::Powi(_, a, b)
        | Instr::Powf(_, a, b)
        | Instr::VAdd(_, a, b)
        | Instr::VSub(_, a, b)
        | Instr::VMul(_, a, b)
        | Instr::VDiv(_, a, b) => vec![a, b],
        Instr::Select(_, _, a, b, on_match, on_miss)
        | Instr::VSelect(_, _, a, b, on_match, on_miss) => vec![a, b, on_match, on_miss],
        Instr::Neg(_, a)
        | Instr::Sqrt(_, a)
        | Instr::Sin(_, a)
        | Instr::Cos(_, a)
        | Instr::Exp(_, a)
        | Instr::Ln(_, a)
        | Instr::Abs(_, a)
        | Instr::VNeg(_, a)
        | Instr::Splat(_, _, a)
        | Instr::Extract(_, a, _) => vec![a],
        Instr::Fma { a, b, c, .. } | Instr::VFma(_, a, b, c) => vec![a, b, c],
        Instr::Pack(_, ops) => ops.iter().collect(),
    }
}

/// Mutable mirror of `instr_operands` (for operand rewriting passes).
pub(crate) fn instr_operands_mut(ins: &mut Instr) -> Vec<&mut Operand> {
    match ins {
        Instr::Add(_, a, b)
        | Instr::Sub(_, a, b)
        | Instr::Mul(_, a, b)
        | Instr::Div(_, a, b)
        | Instr::Atan2(_, a, b)
        | Instr::Powi(_, a, b)
        | Instr::Powf(_, a, b)
        | Instr::VAdd(_, a, b)
        | Instr::VSub(_, a, b)
        | Instr::VMul(_, a, b)
        | Instr::VDiv(_, a, b) => vec![a, b],
        Instr::Select(_, _, a, b, on_match, on_miss)
        | Instr::VSelect(_, _, a, b, on_match, on_miss) => vec![a, b, on_match, on_miss],
        Instr::Neg(_, a)
        | Instr::Sqrt(_, a)
        | Instr::Sin(_, a)
        | Instr::Cos(_, a)
        | Instr::Exp(_, a)
        | Instr::Ln(_, a)
        | Instr::Abs(_, a)
        | Instr::VNeg(_, a)
        | Instr::Splat(_, _, a)
        | Instr::Extract(_, a, _) => vec![a],
        Instr::Fma { a, b, c, .. } | Instr::VFma(_, a, b, c) => vec![a, b, c],
        Instr::Pack(_, ops) => ops.iter_mut().collect(),
    }
}

/// Overwrite an instruction's result slot (its first field).
pub(crate) fn set_instr_result_slot(ins: &mut Instr, slot: u32) {
    match ins {
        Instr::Add(s, ..)
        | Instr::Sub(s, ..)
        | Instr::Mul(s, ..)
        | Instr::Div(s, ..)
        | Instr::Neg(s, ..)
        | Instr::Sqrt(s, ..)
        | Instr::Sin(s, ..)
        | Instr::Cos(s, ..)
        | Instr::Exp(s, ..)
        | Instr::Ln(s, ..)
        | Instr::Abs(s, ..)
        | Instr::Atan2(s, ..)
        | Instr::Powi(s, ..)
        | Instr::Powf(s, ..)
        | Instr::Select(s, ..)
        | Instr::Fma { s, .. }
        | Instr::Splat(s, ..)
        | Instr::Pack(s, ..)
        | Instr::VNeg(s, ..)
        | Instr::VAdd(s, ..)
        | Instr::VSub(s, ..)
        | Instr::VMul(s, ..)
        | Instr::VDiv(s, ..)
        | Instr::VFma(s, ..)
        | Instr::VSelect(s, ..)
        | Instr::Extract(s, ..) => *s = slot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational32;

    #[test]
    fn tree_queries() {
        // root: v0 = in0 + c0 ; branch on v0 ; arms return.
        let root = Block {
            instrs: vec![Instr::Add(0, Operand::Input(0), Operand::Const(0))],
            term: Term::Branch {
                cond: Operand::Instr(0),
                test: BranchTest::EffectiveZero,
                cond_sign: CondSign::Unknown,
                on_true: BlockId(1),
                on_false: BlockId(2),
            },
        };
        let t_arm = Block {
            instrs: vec![],
            term: Term::Return {
                outputs: vec![Operand::Instr(0)],
                fatals: vec![],
            },
        };
        let f_arm = Block {
            instrs: vec![],
            term: Term::Fatal {
                kind: FatalKind::DivByZero,
            },
        };
        let tree = Tree {
            inputs: 1,
            params: 0,
            consts: vec![Rational32::new(1, 6)],
            blocks: vec![root, t_arm, f_arm],
            root: BlockId(0),
        };
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.collect_consts(), vec![0]);
        assert_eq!(
            tree.fatal_leaves(),
            vec![(BlockId(2), FatalKind::DivByZero)]
        );
    }

    #[test]
    fn fma_slot_and_operands() {
        let f = Instr::Fma {
            s: 7,
            neg_prod: true,
            a: Operand::Input(0),
            b: Operand::Input(1),
            c: Operand::Instr(3),
            neg_c: false,
        };
        assert_eq!(instr_result_slot(&f), 7);
        assert_eq!(
            instr_operands(&f),
            vec![&Operand::Input(0), &Operand::Input(1), &Operand::Instr(3)]
        );
    }

    #[test]
    fn vec_instr_operands_roundtrip() {
        let f = Instr::VFma(9, Operand::Instr(1), Operand::Instr(2), Operand::Instr(3));
        assert_eq!(instr_result_slot(&f), 9);
        assert_eq!(
            instr_operands(&f),
            vec![&Operand::Instr(1), &Operand::Instr(2), &Operand::Instr(3)]
        );
        let p = Instr::Pack(4, vec![Operand::Instr(5), Operand::Instr(6), Operand::Eps]);
        assert_eq!(instr_result_slot(&p), 4);
        assert_eq!(
            instr_operands(&p),
            vec![&Operand::Instr(5), &Operand::Instr(6), &Operand::Eps]
        );
        let mut s = Instr::Splat(7, 4, Operand::Input(0));
        for op in instr_operands_mut(&mut s) {
            *op = Operand::Instr(42);
        }
        assert_eq!(instr_operands(&s), vec![&Operand::Instr(42)]);
        let mut e = Instr::Extract(8, Operand::Instr(1), 2);
        set_instr_result_slot(&mut e, 99);
        assert_eq!(instr_result_slot(&e), 99);
    }

    #[test]
    fn pretty_dumps_structure() {
        let root = Block {
            instrs: vec![Instr::Add(0, Operand::Input(0), Operand::Const(0))],
            term: Term::Branch {
                cond: Operand::Instr(0),
                test: BranchTest::EffectiveZero,
                cond_sign: CondSign::Unknown,
                on_true: BlockId(1),
                on_false: BlockId(2),
            },
        };
        let t_arm = Block {
            instrs: vec![],
            term: Term::Return {
                outputs: vec![Operand::Instr(0)],
                fatals: vec![],
            },
        };
        let f_arm = Block {
            instrs: vec![],
            term: Term::Fatal {
                kind: FatalKind::DivByZero,
            },
        };
        let tree = Tree {
            inputs: 1,
            params: 0,
            consts: vec![num_rational::Rational32::new(1, 6)],
            blocks: vec![root, t_arm, f_arm],
            root: BlockId(0),
        };
        let s = tree.pretty();
        assert!(s.contains("block 0"));
        assert!(s.contains("branch"));
        assert!(s.contains("return"));
        assert!(s.contains("fatal"));
    }
}
