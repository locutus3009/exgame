// SPDX-License-Identifier: MIT

use crate::ir::{
    Block, BlockId, BranchTest, CondSign, FatalKind, InputFact, InputRef, Instr, Operand, Term,
    Tree, instr_operands, instr_operands_mut, instr_result_slot, set_instr_result_slot,
};
use crate::sym::{Handle, Sym};
use num_rational::Rational32;
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>;

/// Restores the previous panic hook when dropped (covers normal return AND unwind).
struct HookGuard(Option<PanicHook>);
impl Drop for HookGuard {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            std::panic::set_hook(h);
        }
    }
}

/// One recorded event on the current path: a straight-line instruction or a fork.
#[derive(Clone, Debug)]
pub(crate) enum Event {
    Instr(Instr),
    Branch(Operand, BranchTest, CondSign),
}

/// Per-trace mutable state. Lives in a thread-local; one fresh instance per `run_trace`.
pub(crate) struct Engine {
    /// Events recorded on the current (in-progress) path.
    pub(crate) events: Vec<Event>,
    /// Next runtime result slot (per-path, reset to 0 on `start_path`).
    next_slot: u32,
    /// Shared constant pool (persists across all paths of one trace).
    pub(crate) consts: Vec<Rational32>,
    const_map: HashMap<Rational32, u32>,
    /// Forced branch decisions to replay on the current path, extended at frontiers.
    pub(crate) trail: Vec<bool>,
    /// Index of the next fork on the current path.
    trail_pos: usize,
    /// Trail prefixes still to explore (the worklist).
    pub(crate) pending: Vec<Vec<bool>>,
    /// Per-path memo of fork decisions, keyed by (condition handle, test kind).
    memo: HashMap<(Handle, BranchTest), bool>,
    /// Per-path set of slots proven non-zero (from taken branches + propagation).
    pub(crate) nonzero: HashSet<Handle>,

    pub(crate) positive: HashSet<Handle>,
    pub(crate) negative: HashSet<Handle>,
    pub(crate) nonpositive: HashSet<Handle>,
    pub(crate) nonnegative: HashSet<Handle>,

    /// Per-path: handles proven `abs < ε` (effective-zero). Seeded from the
    /// EffectiveZero true-arm; propagated through magnitude-non-increasing ops.
    pub(crate) abs_lt_eps: HashSet<Handle>,
    /// Per-path: handles proven `abs > ε` (effective-nonzero). Seeded from the
    /// EffectiveZero false-arm; propagated through Neg/Sqrt/Mul-by-|c|>=2.
    pub(crate) abs_gt_eps: HashSet<Handle>,
    /// Domain predicates on inputs (caller-guaranteed). Persistent — set once
    /// before tracing, NOT cleared per path. An input may carry several facts
    /// at once (e.g. warp = {AbsGeOne, Positive}); queries scan the vec.
    pub(crate) input_facts: HashMap<u32, Vec<InputFact>>,
    /// Per-path: handles proven `abs >= 1`. Stronger than abs>eps and — unlike it
    /// — MULTIPLICATIVE for runtime operands (`|a|>=1 ∧ |b|>=1 ⟹ |ab|>=1`), so it
    /// survives `Mul`/`Powi` and reaches products like `warp²`. Implies abs>eps.
    pub(crate) abs_ge_one: HashSet<Handle>,
    /// Per-path slot -> the instruction that produced it (for peephole/inspection).
    pub(crate) defs: Vec<Instr>,
    /// Per-path hash-cons of emitted instructions, keyed by the instr with its
    /// result slot normalized to 0. RESET per path (unlike the shared const pool).
    pub(crate) cse: std::collections::HashMap<Instr, u32>,
    /// INPUTS (the true-input count); a param resolves to `Input(inputs + j)`.
    /// Set once at `run_trace` start, persistent across paths (like input_facts).
    pub(crate) inputs: u32,
}

impl Engine {
    pub(crate) fn new() -> Self {
        Self {
            events: Vec::new(),
            next_slot: 0,
            consts: Vec::new(),
            const_map: HashMap::new(),
            trail: Vec::new(),
            trail_pos: 0,
            pending: Vec::new(),
            memo: HashMap::new(),
            nonzero: HashSet::new(),
            positive: HashSet::new(),
            negative: HashSet::new(),
            nonpositive: HashSet::new(),
            nonnegative: HashSet::new(),
            abs_lt_eps: HashSet::new(),
            abs_gt_eps: HashSet::new(),
            input_facts: HashMap::new(),
            abs_ge_one: HashSet::new(),
            defs: Vec::new(),
            cse: HashMap::new(),
            inputs: 0,
        }
    }

    /// Intern a rational; equal value -> equal index. Load-bearing for trie-merge.
    pub(crate) fn intern_const(&mut self, r: Rational32) -> u32 {
        if let Some(&i) = self.const_map.get(&r) {
            return i;
        }
        let i = self.consts.len() as u32;
        self.consts.push(r);
        self.const_map.insert(r, i);
        i
    }

    pub(crate) fn alloc_slot(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Resolve a `Sym` handle into a graph `Operand`, interning constants on use.
    pub(crate) fn operand(&mut self, h: Handle) -> Operand {
        match h {
            Handle::Constant(r) => Operand::Const(self.intern_const(r)),
            Handle::Input(i) => Operand::Input(i),
            Handle::Param(j) => Operand::Input(self.inputs + j),
            Handle::Runtime(s) => Operand::Instr(s),
        }
    }

    /// Begin a new path with the given forced-decision prefix.
    pub(crate) fn start_path(&mut self, trail: Vec<bool>) {
        self.events.clear();
        self.next_slot = 0;
        self.trail = trail;
        self.trail_pos = 0;
        self.pending.clear();
        self.memo.clear();
        self.nonzero.clear();
        self.abs_lt_eps.clear();
        self.abs_gt_eps.clear();
        self.abs_ge_one.clear();
        self.positive.clear();
        self.negative.clear();
        self.nonpositive.clear();
        self.nonnegative.clear();
        self.defs.clear();
        self.cse.clear(); // LOAD-BEARING: per-path reset, do NOT share across paths
    }

    pub(crate) fn def_of(&self, slot: u32) -> Option<&Instr> {
        self.defs.get(slot as usize)
    }

    pub(crate) fn operand_to_handle(&self, op: Operand) -> Handle {
        match op {
            Operand::Const(i) => Handle::Constant(self.consts[i as usize]),
            Operand::Input(i) => Handle::Input(i),
            Operand::Instr(s) => Handle::Runtime(s),
            Operand::Eps => unreachable!("Eps is created post-build, never resolved to a Handle"),
        }
    }

    /// Proven `abs < ε`? (Only runtime slots are ever inserted; consts/inputs no.)
    pub(crate) fn is_abs_lt_eps(&self, h: Handle) -> bool {
        self.abs_lt_eps.contains(&h)
    }
    /// Proven `abs >= 1`? Const |r|>=1, a seeded runtime slot, or an `AbsGeOne`
    /// input. Multiplicative — propagated through Mul/Powi/Neg/Sqrt in seed_facts.
    pub(crate) fn is_abs_ge_one(&self, h: Handle) -> bool {
        match h {
            Handle::Constant(r) => r.numer().unsigned_abs() >= r.denom().unsigned_abs(),
            Handle::Input(k) => self
                .input_facts
                .get(&k)
                .is_some_and(|f| f.contains(&InputFact::AbsGeOne)),
            Handle::Param(j) => self.is_abs_ge_one(Handle::Input(self.inputs + j)),
            Handle::Runtime(_) => self.abs_ge_one.contains(&h),
        }
    }

    /// Proven `abs > ε`? `abs>=1` implies it; otherwise runtime slots from the
    /// per-path set and inputs from the `AbsGtEps` predicate.
    pub(crate) fn is_abs_gt_eps(&self, h: Handle) -> bool {
        if self.is_abs_ge_one(h) {
            return true; // |x| >= 1 ==> |x| > eps
        }
        match h {
            Handle::Input(k) => self
                .input_facts
                .get(&k)
                .is_some_and(|f| f.contains(&InputFact::AbsGtEps)),
            Handle::Param(j) => self.is_abs_gt_eps(Handle::Input(self.inputs + j)),
            _ => self.abs_gt_eps.contains(&h),
        }
    }

    /// Is this handle proven non-zero? `abs>=1` implies it; else Const != 0, a
    /// seeded runtime slot, or an input declared Nonzero/AbsGtEps/AbsGeOne.
    pub(crate) fn is_nonzero(&self, h: Handle) -> bool {
        if self.is_abs_ge_one(h) {
            return true; // |x| >= 1 ==> x != 0
        }
        match h {
            Handle::Constant(r) => !num_traits::Zero::is_zero(&r),
            Handle::Runtime(_) => self.nonzero.contains(&h),
            Handle::Param(j) => self.is_nonzero(Handle::Input(self.inputs + j)),
            Handle::Input(k) => self.input_facts.get(&k).is_some_and(|f| {
                f.iter().any(|x| {
                    matches!(
                        x,
                        InputFact::Nonzero
                            | InputFact::AbsGtEps
                            | InputFact::Positive
                            | InputFact::Negative
                    )
                })
            }),
        }
    }

    /// Proven `>= 0`? Const with numerator >= 0, an input declared
    /// `NonNegative`/`Positive`, or a runtime slot in the nonneg/positive set.
    pub(crate) fn is_nonneg(&self, h: Handle) -> bool {
        match h {
            Handle::Constant(r) => *r.numer() >= 0,
            Handle::Input(k) => self.input_facts.get(&k).is_some_and(|f| {
                f.iter()
                    .any(|x| matches!(x, InputFact::NonNegative | InputFact::Positive))
            }),
            Handle::Param(j) => self.is_nonneg(Handle::Input(self.inputs + j)),
            Handle::Runtime(_) => self.nonnegative.contains(&h) || self.positive.contains(&h),
        }
    }

    /// Proven `<= 0`? Const with numerator <= 0, an input declared
    /// `NonPositive`/`Negative`, or a runtime slot in the nonpos/negative set.
    pub(crate) fn is_nonpos(&self, h: Handle) -> bool {
        match h {
            Handle::Constant(r) => *r.numer() <= 0,
            Handle::Input(k) => self.input_facts.get(&k).is_some_and(|f| {
                f.iter()
                    .any(|x| matches!(x, InputFact::NonPositive | InputFact::Negative))
            }),
            Handle::Param(j) => self.is_nonpos(Handle::Input(self.inputs + j)),
            Handle::Runtime(_) => self.nonpositive.contains(&h) || self.negative.contains(&h),
        }
    }

    /// Proven `> 0`? Strict directly, or `>= 0` together with `!= 0` (closure).
    pub(crate) fn is_positive(&self, h: Handle) -> bool {
        match h {
            Handle::Constant(r) => *r.numer() > 0,
            Handle::Input(k) => self
                .input_facts
                .get(&k)
                .is_some_and(|f| f.contains(&InputFact::Positive)),
            Handle::Param(j) => self.is_positive(Handle::Input(self.inputs + j)),
            Handle::Runtime(_) => {
                self.positive.contains(&h) || (self.nonnegative.contains(&h) && self.is_nonzero(h))
            }
        }
    }

    /// Proven `< 0`? Strict directly, or `<= 0` together with `!= 0` (closure).
    pub(crate) fn is_negative(&self, h: Handle) -> bool {
        match h {
            Handle::Constant(r) => *r.numer() < 0,
            Handle::Input(k) => self
                .input_facts
                .get(&k)
                .is_some_and(|f| f.contains(&InputFact::Negative)),
            Handle::Param(j) => self.is_negative(Handle::Input(self.inputs + j)),
            Handle::Runtime(_) => {
                self.negative.contains(&h) || (self.nonpositive.contains(&h) && self.is_nonzero(h))
            }
        }
    }
}

thread_local! {
    pub(crate) static ENGINE: RefCell<Engine> = RefCell::new(Engine::new());
}

/// Emit a binary op: resolve operands, allocate a result slot, push the Instr.
/// Operands are resolved BEFORE slot allocation so slot ids stay deterministic.
/// `build` is `Fn` (not `FnOnce`) because it is called twice on a miss: once
/// with slot=0 to form the CSE key, and once with the real slot.
pub(crate) fn emit_bin(
    a: Handle,
    b: Handle,
    build: impl Fn(u32, Operand, Operand) -> Instr,
) -> Handle {
    ENGINE.with(|c| {
        let e = &mut *c.borrow_mut();
        let oa = e.operand(a);
        let ob = e.operand(b);
        let key = build(0, oa, ob); // normalized result slot -> CSE key
        if let Some(&slot) = e.cse.get(&key) {
            return Handle::Runtime(slot); // hit: facts already recorded at first emit
        }
        let slot = e.alloc_slot();
        let instr = build(slot, oa, ob);
        debug_assert_eq!(e.defs.len(), slot as usize, "defs indexed by slot");
        e.defs.push(instr.clone());
        let result = Handle::Runtime(slot);
        seed_facts(e, result, &instr, a, Some(b));
        e.events.push(Event::Instr(instr));
        e.cse.insert(key, slot);
        result
    })
}

/// Emit a unary op.
/// `build` is `Fn` (not `FnOnce`) — called twice on a miss for CSE keying.
pub(crate) fn emit_un(a: Handle, build: impl Fn(u32, Operand) -> Instr) -> Handle {
    ENGINE.with(|c| {
        let e = &mut *c.borrow_mut();
        let oa = e.operand(a);
        let key = build(0, oa); // normalized result slot -> CSE key
        if let Some(&slot) = e.cse.get(&key) {
            return Handle::Runtime(slot);
        }
        let slot = e.alloc_slot();
        let instr = build(slot, oa);
        debug_assert_eq!(e.defs.len(), slot as usize, "defs indexed by slot");
        e.defs.push(instr.clone());
        let result = Handle::Runtime(slot);
        seed_facts(e, result, &instr, a, None);
        e.events.push(Event::Instr(instr));
        e.cse.insert(key, slot);
        result
    })
}

/// Emit a `Select` (index match → pick): `on_match` if `selector == lane` else
/// `on_miss`. `lane`/`on_match`/`on_miss` are pooled constants (1/0 of a one-hot
/// seed become constant references, not literals). Straight-line — NO fork, and
/// no facts seeded (result is in {0, 1}). Operands resolved before slot alloc so
/// slot ids stay deterministic (same invariant as `emit_bin`).
pub(crate) fn emit_select(selector: Handle, lane: u32) -> Handle {
    ENGINE.with(|c| {
        let e = &mut *c.borrow_mut();
        let sel = e.operand(selector);
        let lane_op = e.operand(Handle::Constant(Rational32::new_raw(lane as i32, 1)));
        let one = e.operand(Handle::Constant(Rational32::new_raw(1, 1)));
        let zero = e.operand(Handle::Constant(Rational32::new_raw(0, 1)));
        let key = Instr::Select(0, crate::ir::CmpKind::Eq, sel, lane_op, one, zero);
        if let Some(&slot) = e.cse.get(&key) {
            return Handle::Runtime(slot);
        }
        let slot = e.alloc_slot();
        let instr = Instr::Select(slot, crate::ir::CmpKind::Eq, sel, lane_op, one, zero);
        debug_assert_eq!(e.defs.len(), slot as usize, "defs indexed by slot");
        e.defs.push(instr.clone());
        e.events.push(Event::Instr(instr));
        e.cse.insert(key, slot);
        e.nonnegative.insert(Handle::Runtime(slot)); // one-hot value in {0, 1}
        Handle::Runtime(slot)
    })
}

/// Emit `max(a, b)` as ONE straight-line instruction: `a < b ? b : a`. No
/// `is_effective_zero` fork — the decision trie is untouched, which is the whole
/// reason this exists instead of a comparison the tracer would have to decide at
/// trace time. Operands resolved before slot alloc, as in `emit_select`.
pub(crate) fn emit_max(a: Handle, b: Handle) -> Handle {
    ENGINE.with(|c| {
        let e = &mut *c.borrow_mut();
        let ao = e.operand(a);
        let bo = e.operand(b);
        let key = Instr::Select(0, crate::ir::CmpKind::Lt, ao, bo, bo, ao);
        if let Some(&slot) = e.cse.get(&key) {
            return Handle::Runtime(slot);
        }
        let slot = e.alloc_slot();
        let instr = Instr::Select(slot, crate::ir::CmpKind::Lt, ao, bo, bo, ao);
        debug_assert_eq!(e.defs.len(), slot as usize, "defs indexed by slot");
        e.defs.push(instr.clone());
        e.events.push(Event::Instr(instr));
        e.cse.insert(key, slot);
        Handle::Runtime(slot)
    })
}

/// Forward-propagate per-path facts onto a freshly-emitted instruction's result.
/// Single source of truth, keyed on the instruction kind; operand HANDLES are
/// passed (not just Operands) so constant values are available directly.
/// ≠0 lattice: Mul(both), Neg/Sqrt/Powi(operand), Exp(always); everything else
/// proves nothing. (abs-band rules are added in Task 4.)
fn seed_facts(e: &mut Engine, result: Handle, instr: &Instr, a: Handle, b: Option<Handle>) {
    match instr {
        Instr::Mul(..) => {
            let bh = b.expect("Mul is binary");
            if e.is_nonzero(a) && e.is_nonzero(bh) {
                e.nonzero.insert(result);
            }
            // abs-band only when one operand is a constant multiplier (the other
            // runtime). Const*const and *±1/*0 are folded before emit, so the
            // multiplier here is never 0/±1.
            let (x, c) = match (a, bh) {
                (Handle::Constant(c), other) => (other, Some(c)),
                (other, Handle::Constant(c)) => (other, Some(c)),
                _ => (a, None),
            };
            if let Some(c) = c {
                if e.is_abs_lt_eps(x) && abs_le_half(c) {
                    e.abs_lt_eps.insert(result);
                }
                if e.is_abs_gt_eps(x) && abs_ge_two(c) {
                    e.abs_gt_eps.insert(result);
                }
            }
            // abs>=1 IS multiplicative (unlike abs>eps): both factors >=1 -> >=1.
            // Works for runtime*runtime (this is what reaches warp^2).
            if e.is_abs_ge_one(a) && e.is_abs_ge_one(bh) {
                e.abs_ge_one.insert(result);
            }
            // sign: multiplicative table, with self-square as a special case.
            let (mut nn, mut np, mut pos, mut neg) = mul_sign(e, a, bh);
            if a == bh {
                nn = true; // x^2 >= 0 for any real x
                np = false;
                neg = false;
                pos = pos || e.is_nonzero(a); // x != 0 ==> x^2 > 0
            }
            set_sign(e, result, nn, np, pos, neg);
        }
        Instr::Neg(..) => {
            if e.is_nonzero(a) {
                e.nonzero.insert(result);
            }
            if e.is_abs_lt_eps(a) {
                e.abs_lt_eps.insert(result); // |-x| == |x|
            }
            if e.is_abs_gt_eps(a) {
                e.abs_gt_eps.insert(result);
            }
            if e.is_abs_ge_one(a) {
                e.abs_ge_one.insert(result); // |-x| == |x|
            }
            // sign flip
            set_sign(
                e,
                result,
                e.is_nonpos(a),
                e.is_nonneg(a),
                e.is_negative(a),
                e.is_positive(a),
            );
        }
        Instr::Sqrt(..) => {
            if e.is_nonzero(a) {
                e.nonzero.insert(result);
            }
            // sqrt grows small values, so it does NOT preserve abs<eps; it does
            // preserve abs>eps (sqrt(>eps) > sqrt(eps) >> eps for eps<1). Sound
            // even for x<-eps: the fact only ever drives "EffectiveZero=false",
            // and sqrt(neg)=NaN also satisfies "abs(NaN)<eps is false". See spec.
            if e.is_abs_gt_eps(a) {
                e.abs_gt_eps.insert(result);
            }
            if e.is_abs_ge_one(a) {
                e.abs_ge_one.insert(result); // sqrt(>=1) >= 1
            }
            // sqrt(a) >= 0 always; > 0 once a != 0
            set_sign(e, result, true, false, e.is_nonzero(a), false);
        }
        Instr::Powi(..) => {
            if e.is_nonzero(a) {
                e.nonzero.insert(result);
            }
            // |a|>=1 and exponent n>=0 -> |a^n| >= 1. (n is the Const operand.)
            if e.is_abs_ge_one(a)
                && let Some(Handle::Constant(n)) = b
                && *n.numer() >= 0
            {
                e.abs_ge_one.insert(result);
            }
            // sign by parity of the (constant) exponent
            if let Some(Handle::Constant(n)) = b {
                if *n.numer() % 2 == 0 {
                    // even (incl. negative even, incl. 0): result >= 0
                    set_sign(e, result, true, false, e.is_nonzero(a), false);
                } else {
                    // odd (incl. negative odd): preserves a's sign
                    set_sign(
                        e,
                        result,
                        e.is_nonneg(a),
                        e.is_nonpos(a),
                        e.is_positive(a),
                        e.is_negative(a),
                    );
                }
            }
        }
        Instr::Exp(..) => {
            e.nonzero.insert(result); // exp(x) > 0 for all finite x
            e.nonnegative.insert(result);
            e.positive.insert(result); // exp(x) > 0 for all finite x
            if e.is_nonneg(a) {
                e.abs_ge_one.insert(result); // exp(>=0) >= 1
            }
        }
        Instr::Add(..) => {
            let bh = b.expect("Add is binary");
            let pos =
                (e.is_positive(a) && e.is_nonneg(bh)) || (e.is_nonneg(a) && e.is_positive(bh));
            let neg =
                (e.is_negative(a) && e.is_nonpos(bh)) || (e.is_nonpos(a) && e.is_negative(bh));
            let nonneg = e.is_nonneg(a) && e.is_nonneg(bh);
            let nonpos = e.is_nonpos(a) && e.is_nonpos(bh);
            set_sign(e, result, nonneg, nonpos, pos, neg);
            // additive band strengthening: a definite-signed dominant term plus
            // a same-direction term keeps the dominant term's magnitude band.
            if e.is_positive(a) && e.is_nonneg(bh) {
                carry_band(e, result, a);
            }
            if e.is_positive(bh) && e.is_nonneg(a) {
                carry_band(e, result, bh);
            }
            if e.is_negative(a) && e.is_nonpos(bh) {
                carry_band(e, result, a);
            }
            if e.is_negative(bh) && e.is_nonpos(a) {
                carry_band(e, result, bh);
            }
            // structural: 1 + cos(_) >= 0 (either operand order)
            if (is_const_one(a) && is_cos_result(e, bh))
                || (is_const_one(bh) && is_cos_result(e, a))
            {
                e.nonnegative.insert(result);
            }
        }
        Instr::Sub(..) => {
            let bh = b.expect("Sub is binary");
            // a - b == a + (-b): flip b's sign predicates
            let (b_nn, b_np, b_pos, b_neg) = (
                e.is_nonpos(bh),
                e.is_nonneg(bh),
                e.is_negative(bh),
                e.is_positive(bh),
            );
            let pos = (e.is_positive(a) && b_nn) || (e.is_nonneg(a) && b_pos);
            let neg = (e.is_negative(a) && b_np) || (e.is_nonpos(a) && b_neg);
            let nonneg = e.is_nonneg(a) && b_nn;
            let nonpos = e.is_nonpos(a) && b_np;
            set_sign(e, result, nonneg, nonpos, pos, neg);
            // additive band strengthening (dominant term, possibly the -b side)
            if e.is_positive(a) && b_nn {
                carry_band(e, result, a);
            }
            if b_pos && e.is_nonneg(a) {
                carry_band(e, result, bh);
            }
            if e.is_negative(a) && b_np {
                carry_band(e, result, a);
            }
            if b_neg && e.is_nonpos(a) {
                carry_band(e, result, bh);
            }
            // structural: 1 - cos(_) >= 0
            if is_const_one(a) && is_cos_result(e, bh) {
                e.nonnegative.insert(result);
            }
        }
        Instr::Div(..) => {
            let bh = b.expect("Div is binary");
            let (nn, np, pos, neg) = mul_sign(e, a, bh);
            set_sign(e, result, nn, np, pos, neg);
        }
        Instr::Powf(..) => {
            // real powf domain is a >= 0; result is >= 0 (conservative: no sign)
            e.nonnegative.insert(result);
        }
        _ => {}
    }
}

/// Record the sign of `result` and cross-feed nonzero / the zero sentinel.
/// `pos`/`neg` are strict; they imply nonneg/nonpos and nonzero.
fn set_sign(
    e: &mut Engine,
    result: Handle,
    mut nonneg: bool,
    mut nonpos: bool,
    pos: bool,
    neg: bool,
) {
    if pos {
        nonneg = true;
        e.positive.insert(result);
        e.nonzero.insert(result);
    }
    if neg {
        nonpos = true;
        e.negative.insert(result);
        e.nonzero.insert(result);
    }
    if nonneg {
        e.nonnegative.insert(result);
    }
    if nonpos {
        e.nonpositive.insert(result);
    }
    // both bounds => exactly zero => effective-zero (correctness sentinel)
    if nonneg && nonpos {
        e.abs_lt_eps.insert(result);
    }
}

/// Sign of a product/quotient of `a` and `b` (multiplicative sign table).
/// Returns `(nonneg, nonpos, pos, neg)`.
fn mul_sign(e: &Engine, a: Handle, b: Handle) -> (bool, bool, bool, bool) {
    let (ann, anp, ap, an) = (
        e.is_nonneg(a),
        e.is_nonpos(a),
        e.is_positive(a),
        e.is_negative(a),
    );
    let (bnn, bnp, bp, bn) = (
        e.is_nonneg(b),
        e.is_nonpos(b),
        e.is_positive(b),
        e.is_negative(b),
    );
    let pos = (ap && bp) || (an && bn);
    let neg = (ap && bn) || (an && bp);
    let nonneg = (ann && bnn) || (anp && bnp);
    let nonpos = (ann && bnp) || (anp && bnn);
    (nonneg, nonpos, pos, neg)
}

/// Carry the dominant term's magnitude band onto a same-signed sum. Uses abs
/// (sign-agnostic), so a negated dominant operand is fine.
fn carry_band(e: &mut Engine, result: Handle, dom: Handle) {
    if e.is_abs_gt_eps(dom) {
        e.abs_gt_eps.insert(result);
    }
    if e.is_abs_ge_one(dom) {
        e.abs_ge_one.insert(result);
    }
}

/// Exactly the constant `1`.
fn is_const_one(h: Handle) -> bool {
    matches!(h, Handle::Constant(r) if *r.numer() == 1 && *r.denom() == 1)
}

/// A runtime result produced by `Cos`.
fn is_cos_result(e: &Engine, h: Handle) -> bool {
    matches!(h, Handle::Runtime(s) if matches!(e.def_of(s), Some(crate::ir::Instr::Cos(..))))
}

/// |c| <= 1/2, exactly, over u64 to avoid i32 overflow. (den > 0 for Rational32.)
fn abs_le_half(c: Rational32) -> bool {
    (c.numer().unsigned_abs() as u64) * 2 <= (c.denom().unsigned_abs() as u64)
}
/// |c| >= 2, exactly.
fn abs_ge_two(c: Rational32) -> bool {
    (c.numer().unsigned_abs() as u64) >= (c.denom().unsigned_abs() as u64) * 2
}

/// If `h` is a runtime slot produced by `Neg(inner)`, return `inner` as a Handle.
pub(crate) fn neg_of(h: Handle) -> Option<Handle> {
    ENGINE.with(|c| {
        let e = &*c.borrow();
        if let Handle::Runtime(s) = h
            && let Some(Instr::Neg(_, inner)) = e.def_of(s)
        {
            return Some(e.operand_to_handle(*inner));
        }
        None
    })
}

/// Record a fork on the condition handle and return the forced decision.
/// Replays `trail` if available; at a frontier takes `true` and enqueues
/// the `false` flip. This is the ONLY branch mechanism.
/// Same (handle, test) pair on the same path is memoized: no new branch event is
/// emitted and the earlier decision is returned directly.
/// ExactZero guards on handles already proven nonzero are suppressed entirely:
/// the "not zero" arm is returned immediately with no event or trail slot consumed.
pub(crate) fn fork(cond: Handle, test: BranchTest) -> bool {
    ENGINE.with(|c| {
        let e = &mut *c.borrow_mut();
        // Resolve a param to the input it becomes, so memo / nonzero / abs-band
        // sets key consistently with values that arrive as Input(INPUTS+j).
        let cond = match cond {
            Handle::Param(j) => Handle::Input(e.inputs + j),
            other => other,
        };
        // Suppress a div-by-zero guard whose divisor is proven nonzero: determined
        // "not zero" arm, no branch, no flip enqueued.
        if test == BranchTest::ExactZero && e.is_nonzero(cond) {
            return false;
        }
        // EffectiveZero determined by the abs-band: abs>eps => not effective-zero
        // (return false, "abs>=eps" arm); abs<eps => effective-zero (return true,
        // pruning the "abs>=eps" arm). No event, no trail slot — like the ExactZero
        // suppression above.
        if test == BranchTest::EffectiveZero {
            if e.is_abs_gt_eps(cond) {
                return false;
            }
            if e.is_abs_lt_eps(cond) {
                return true;
            }
        }
        if let Some(&d) = e.memo.get(&(cond, test)) {
            return d;
        }
        // Stamp the cond's sign for EffectiveZero (drives abs elision at
        // lowering). Only nonneg/nonpos matter; these are structural/input-
        // derived, hence path-stable (see insert()'s prefix assert).
        let cond_sign = if test == BranchTest::EffectiveZero {
            if e.is_nonneg(cond) {
                CondSign::NonNeg
            } else if e.is_nonpos(cond) {
                CondSign::NonPos
            } else {
                CondSign::Unknown
            }
        } else {
            CondSign::Unknown
        };
        let op = e.operand(cond);
        e.events.push(Event::Branch(op, test, cond_sign));
        let i = e.trail_pos;
        e.trail_pos += 1;
        let d = if i < e.trail.len() {
            e.trail[i]
        } else {
            let mut alt = e.trail.clone();
            alt.push(false);
            e.pending.push(alt);
            e.trail.push(true);
            true
        };
        e.memo.insert((cond, test), d);
        // Seed facts from the arm taken. Either test's false arm proves != 0.
        // EffectiveZero additionally splits the abs-band: false => abs>eps,
        // true => abs<eps.
        if !d {
            e.nonzero.insert(cond);
            if test == BranchTest::EffectiveZero {
                e.abs_gt_eps.insert(cond);
            }
        } else if test == BranchTest::EffectiveZero {
            e.abs_lt_eps.insert(cond);
        }
        d
    })
}

const SENT: BlockId = BlockId(u32::MAX);

/// Mutable trie under construction. `term: None` = block not yet filled.
struct BBlock {
    instrs: Vec<Instr>,
    term: Option<Term>,
}

pub(crate) struct TreeBuilder {
    blocks: Vec<BBlock>,
}

impl TreeBuilder {
    fn new() -> Self {
        Self {
            blocks: vec![BBlock {
                instrs: Vec::new(),
                term: None,
            }],
        }
    }

    /// Insert one root->leaf path described by its event stream + decision vector.
    fn insert(&mut self, events: &[Event], decisions: &[bool], leaf: Term) {
        let mut cur = 0usize;
        let mut acc: Vec<Instr> = Vec::new();
        let mut di = 0usize;
        for ev in events {
            match ev {
                Event::Instr(ins) => acc.push(ins.clone()),
                Event::Branch(cond, test, cond_sign) => {
                    let d = decisions[di];
                    di += 1;
                    match &self.blocks[cur].term {
                        None => {
                            self.blocks[cur].instrs = std::mem::take(&mut acc);
                            self.blocks[cur].term = Some(Term::Branch {
                                cond: *cond,
                                test: *test,
                                cond_sign: *cond_sign,
                                on_true: SENT,
                                on_false: SENT,
                            });
                        }
                        Some(Term::Branch {
                            cond: c2,
                            test: t2,
                            cond_sign: cs2,
                            ..
                        }) => {
                            assert_eq!(c2, cond, "branch condition prefix divergence");
                            assert_eq!(t2, test, "branch test-kind prefix divergence");
                            // Sign is structural / input-derived (branch arms seed
                            // only nonzero/abs-band, never sign), so it is the same
                            // wherever a given cond is tested.
                            assert_eq!(cs2, cond_sign, "branch cond_sign prefix divergence");
                            assert_eq!(
                                self.blocks[cur].instrs, acc,
                                "prefix instruction divergence"
                            );
                            acc.clear();
                        }
                        Some(_) => panic!("non-branch terminator where a branch was expected"),
                    }
                    // read/allocate the chosen child
                    let child = {
                        let (t, f) = match &self.blocks[cur].term {
                            Some(Term::Branch {
                                on_true, on_false, ..
                            }) => (*on_true, *on_false),
                            _ => unreachable!(),
                        };
                        let existing = if d { t } else { f };
                        if existing == SENT {
                            let nid = BlockId(self.blocks.len() as u32);
                            self.blocks.push(BBlock {
                                instrs: Vec::new(),
                                term: None,
                            });
                            if let Some(Term::Branch {
                                on_true, on_false, ..
                            }) = &mut self.blocks[cur].term
                            {
                                if d { *on_true = nid } else { *on_false = nid }
                            }
                            nid
                        } else {
                            existing
                        }
                    };
                    cur = child.0 as usize;
                }
            }
        }
        // leaf block (reached by exactly one decision vector -> filled once)
        match &self.blocks[cur].term {
            None => {
                self.blocks[cur].instrs = std::mem::take(&mut acc);
                self.blocks[cur].term = Some(leaf);
            }
            Some(_) => panic!("leaf block already filled (duplicate path?)"),
        }
    }

    fn finish(self) -> Vec<Block> {
        self.blocks
            .into_iter()
            .map(|b| Block {
                instrs: b.instrs,
                term: b
                    .term
                    .expect("unfilled block (max_leaves cut the enumeration?)"),
            })
            .collect()
    }
}

/// Trace configuration (built by `TracerBuilder`).
#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) lua_threshold: f64,
    pub(crate) max_leaves: Option<usize>,
    /// Name of the emitted entry function. Backend-general (not Lua-specific):
    /// every backend names its top-level function this.
    pub(crate) fn_name: String,
    /// If-convert all representation-select branches into one flat block
    /// (`flatten_branches`). Off by default (keeps the branched trie, which the
    /// pass-level tests inspect); opt in for the GPU-friendly divergence-free
    /// form via `TracerBuilder::flatten`.
    pub(crate) flatten: bool,
    /// Run the SLP `vectorize` pass (collapse lane-parallel elementwise groups
    /// into width-≤4 vector instrs). Off by default; opt in via
    /// `TracerBuilder::vectorize`. Requires `flatten` to be useful (operates on
    /// the flat block).
    pub(crate) vectorize: bool,
    /// Explicit output lane-groups (indices into the output list) the caller
    /// knows are lane-parallel — e.g. the per-component gradient lanes of an AD
    /// Jacobian. AD zero-folding makes the lanes structurally unique, so shape
    /// inference can't recover them; the caller declares them. Each group is
    /// tiled to ≤4 and forced through the vectorizer (which still checks
    /// op-compatibility per level). Empty ⇒ shape-hash auto-seeding only.
    pub(crate) vectorize_lanes: Vec<Vec<usize>>,
    /// Run the algebraic `fast_math_simplify` pass on the flat block before
    /// vectorize: cancellations like `(a/b)*b -> a`, `a/b*b/c -> a/c`,
    /// `sqrt(a)*sqrt(a) -> a`. NOT bit-exact with the f64 carrier (it changes
    /// rounding), only algebraically equal up to the consumer's tolerance.
    /// Off by default; opt in via `TracerBuilder::fast_math`.
    pub(crate) fast_math: bool,
}

/// The traced closure: state inputs + params -> outputs (boundary-level types;
/// params are seeded as `Input(INPUTS + j)` once inside). The `'a` keeps the
/// trait object's lifetime tied to the borrow (a bare `dyn` alias would default
/// to `'static` and reject non-`'static` closures).
type TraceFn<'a> = dyn Fn(&Vec<Sym>, &Vec<Sym>) -> Vec<Sym> + 'a;

/// Drive ALL paths via a worklist loop, one path per forced-decision prefix.
pub(crate) fn run_trace(
    cfg: &Config,
    constraints: &[(InputRef, InputFact)],
    f: &TraceFn<'_>,
    n_inputs: usize,
    n_params: usize,
    n_outputs: usize,
) -> Tree {
    ENGINE.with(|c| {
        let e = &mut *c.borrow_mut();
        *e = Engine::new();
        e.inputs = n_inputs as u32;
        for &(r, fact) in constraints {
            let abs = match r {
                InputRef::Input(k) => k,
                InputRef::Param(j) => n_inputs as u32 + j,
            };
            e.input_facts.entry(abs).or_default().push(fact);
        }
    });
    let inputs: Vec<Sym> = (0..n_inputs).map(|i| Sym::input(i as u32)).collect();
    let params: Vec<Sym> = (0..n_params).map(|i| Sym::param(i as u32)).collect();

    let mut builder = TreeBuilder::new();
    let mut worklist: Vec<Vec<bool>> = vec![Vec::new()];
    let mut leaves = 0usize;

    let _hook_guard = HookGuard(Some(std::panic::take_hook()));
    std::panic::set_hook(Box::new(|_| {}));

    while let Some(trail) = worklist.pop() {
        if let Some(cap) = cfg.max_leaves {
            assert!(leaves < cap, "viete: max_leaves ({cap}) exceeded");
        }
        ENGINE.with(|c| c.borrow_mut().start_path(trail));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&inputs, &params)));
        let (events, full_trail, pending, leaf) = ENGINE.with(|c| {
            let e = &mut *c.borrow_mut();
            let leaf = match result {
                Ok(outs) => {
                    assert_eq!(outs.len(), n_outputs);
                    let ops: Vec<Operand> = outs.iter().map(|s| e.operand(s.0)).collect();
                    // fatals are populated post-build by `collapse_fatal_branches`.
                    Term::Return {
                        outputs: ops,
                        fatals: vec![],
                    }
                }
                Err(payload) => Term::Fatal {
                    kind: classify(payload),
                },
            };
            (
                std::mem::take(&mut e.events),
                e.trail.clone(),
                std::mem::take(&mut e.pending),
                leaf,
            )
        });
        worklist.extend(pending);
        builder.insert(&events, &full_trail, leaf);
        leaves += 1;
    }

    let blocks = builder.finish();
    let consts = ENGINE.with(|c| c.borrow().consts.clone());
    let mut tree = Tree {
        inputs: n_inputs as u32,
        params: n_params as u32,
        consts,
        blocks,
        root: BlockId(0),
    };
    dce(&mut tree); // minimize first so the single-use gate sees true live counts
    fma_contract(&mut tree);
    hoist_invariant(&mut tree, cfg.flatten); // dominator-based CSE (+ optional if-conversion)
    dce(&mut tree); // sweep the Mul/Neg orphaned by contraction
    if cfg.fast_math {
        fast_math_simplify(&mut tree); // algebraic cancellations + reciprocal-CSE
        hoist_invariant(&mut tree, cfg.flatten); // re-CSE: reciprocals expose shared structure
        dce(&mut tree);
    }
    if cfg.vectorize {
        vectorize(&mut tree, &cfg.vectorize_lanes);
    }
    tree
}

/// Per-rule firing counts for the `fast_math` pass (for measurement / tests).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FastMathStats {
    pub mul_div_cancel: u32, // (a/b)*b -> a ; b*(a/b) -> a
    pub div_mul_cancel: u32, // (a*b)/b -> a ; (b*a)/b -> a
    pub chain_cancel: u32,   // (a/b)*(b/c) -> a/c ; (a/b)/(c/b) -> a/c
    pub sqrt_square: u32,    // sqrt(a)*sqrt(a) -> a
    pub recip_divisors: u32, // distinct divisors hoisted to a reciprocal
    pub recip_divs_cut: u32, // divisions turned into multiplications
}

/// Reciprocal-CSE: a divisor `b` used by >=2 divisions becomes one `recip = 1/b`,
/// and each `x/b` becomes `x * recip`. Reduces division COUNT (slow on GPU) at a
/// cost of +1 instruction per distinct divisor. Not bit-exact (1/b then *x is a
/// second rounding). Flat block only.
fn fast_math_recip_cse(tree: &mut Tree, st: &mut FastMathStats) {
    if tree.blocks.len() != 1 {
        return;
    }
    // Count how many divisions use each runtime divisor (Instr slot or Input/Param).
    let is_runtime = |op: &Operand| matches!(op, Operand::Instr(_) | Operand::Input(_));
    let mut freq: HashMap<Operand, u32> = HashMap::new();
    for ins in &tree.blocks[0].instrs {
        if let Instr::Div(_, _, b) = ins
            && is_runtime(b)
        {
            *freq.entry(*b).or_insert(0) += 1;
        }
    }
    let hoist: std::collections::HashSet<Operand> = freq
        .iter()
        .filter(|(_, n)| **n >= 2)
        .map(|(b, _)| *b)
        .collect();
    if hoist.is_empty() {
        return;
    }
    let one = pool_index(tree, num_rational::Rational32::new(1, 1));
    let mut next = next_free_local(&tree.blocks[0].instrs);
    let mut recip: HashMap<Operand, u32> = HashMap::new(); // divisor -> recip slot
    let old = std::mem::take(&mut tree.blocks[0].instrs);
    let mut out: Vec<Instr> = Vec::with_capacity(old.len() + hoist.len());
    for ins in old {
        if let Instr::Div(s, a, b) = ins
            && hoist.contains(&b)
        {
            let r = *recip.entry(b).or_insert_with(|| {
                let rs = next;
                next += 1;
                out.push(Instr::Div(rs, Operand::Const(one), b));
                st.recip_divisors += 1;
                rs
            });
            out.push(Instr::Mul(s, a, Operand::Instr(r)));
            st.recip_divs_cut += 1;
        } else {
            out.push(ins);
        }
    }
    tree.blocks[0].instrs = out;
}

/// First free local slot id in a block (max result slot + 1).
fn next_free_local(instrs: &[Instr]) -> u32 {
    instrs
        .iter()
        .map(instr_result_slot)
        .max()
        .map_or(0, |m| m + 1)
}

/// Algebraic simplification pass. NOT bit-exact: each rule preserves the exact
/// real value (modulo domain conditions already handled by the fatal mechanism)
/// but changes float rounding. Operates on the single flat block (post-flatten);
/// no-op otherwise. Cancellations that drop a `Div` are fatal-safe: the
/// div-by-zero status lives in a separate `Select` slot that survives DCE.
pub(crate) fn fast_math_simplify(tree: &mut Tree) -> FastMathStats {
    let mut st = FastMathStats::default();
    if tree.blocks.len() != 1 {
        return st;
    }
    let instrs = std::mem::take(&mut tree.blocks[0].instrs);
    let mut def: HashMap<u32, Instr> = HashMap::new();
    let mut remap: HashMap<u32, Operand> = HashMap::new();
    let mut out: Vec<Instr> = Vec::with_capacity(instrs.len());

    for mut ins in instrs {
        // Resolve operands through prior rewrites first (flat block is topo-ordered).
        for op in instr_operands_mut(&mut ins) {
            if let Operand::Instr(s) = *op
                && let Some(r) = remap.get(&s)
            {
                *op = *r;
            }
        }
        let slot = instr_result_slot(&ins);
        // Rules that collapse this slot to an existing operand (drop the instr).
        let collapse = match &ins {
            Instr::Mul(_, x, y) => fm_mul_collapse(&def, *x, *y, &mut st),
            Instr::Div(_, x, y) => fm_div_collapse(&def, *x, *y, &mut st),
            _ => None,
        };
        if let Some(op) = collapse {
            remap.insert(slot, op);
            continue;
        }
        // Rule that rewrites this slot to a cheaper instr (e.g. Mul(Div,Div) -> Div).
        if let Some(newins) = fm_rewrite(&def, &ins, &mut st) {
            def.insert(slot, newins.clone());
            out.push(newins);
        } else {
            def.insert(slot, ins.clone());
            out.push(ins);
        }
    }
    for op in term_operands_mut(&mut tree.blocks[0].term) {
        if let Operand::Instr(s) = *op
            && let Some(r) = remap.get(&s)
        {
            *op = *r;
        }
    }
    tree.blocks[0].instrs = out;
    fast_math_recip_cse(tree, &mut st); // div-by-shared-denominator -> reciprocal
    dce(tree); // sweep instrs orphaned by cancellation
    st
}

/// The `Instr` defining operand `op`, if it is a runtime slot.
fn fm_def(def: &HashMap<u32, Instr>, op: Operand) -> Option<&Instr> {
    match op {
        Operand::Instr(s) => def.get(&s),
        _ => None,
    }
}

/// `Mul(x, y)` collapse rules → the operand the product equals.
fn fm_mul_collapse(
    def: &HashMap<u32, Instr>,
    x: Operand,
    y: Operand,
    st: &mut FastMathStats,
) -> Option<Operand> {
    // (a/y)*y -> a  and  y*(a/y) -> a
    if let Some(Instr::Div(_, a, b)) = fm_def(def, x)
        && *b == y
    {
        st.mul_div_cancel += 1;
        return Some(*a);
    }
    if let Some(Instr::Div(_, a, b)) = fm_def(def, y)
        && *b == x
    {
        st.mul_div_cancel += 1;
        return Some(*a);
    }
    // sqrt(a)*sqrt(a) -> a  (same Sqrt slot, or two Sqrt of the same operand)
    if let (Some(Instr::Sqrt(_, a1)), Some(Instr::Sqrt(_, a2))) = (fm_def(def, x), fm_def(def, y))
        && a1 == a2
    {
        st.sqrt_square += 1;
        return Some(*a1);
    }
    None
}

/// `Div(x, y)` collapse rules → the operand the quotient equals.
fn fm_div_collapse(
    def: &HashMap<u32, Instr>,
    x: Operand,
    y: Operand,
    st: &mut FastMathStats,
) -> Option<Operand> {
    // (a*y)/y -> a  and  (y*a)/y -> a
    if let Some(Instr::Mul(_, p, q)) = fm_def(def, x) {
        if *q == y {
            st.div_mul_cancel += 1;
            return Some(*p);
        }
        if *p == y {
            st.div_mul_cancel += 1;
            return Some(*q);
        }
    }
    None
}

/// Rules that replace `ins` with a cheaper instruction (same result slot).
fn fm_rewrite(def: &HashMap<u32, Instr>, ins: &Instr, st: &mut FastMathStats) -> Option<Instr> {
    match ins {
        // (a/b)*(b/c) -> a/c
        Instr::Mul(s, x, y) => {
            if let (Some(Instr::Div(_, a, b1)), Some(Instr::Div(_, b2, c))) =
                (fm_def(def, *x), fm_def(def, *y))
                && b1 == b2
            {
                st.chain_cancel += 1;
                return Some(Instr::Div(*s, *a, *c));
            }
            None
        }
        // (a/b)/(c/b) -> a/c
        Instr::Div(s, x, y) => {
            if let (Some(Instr::Div(_, a, b1)), Some(Instr::Div(_, c, b2))) =
                (fm_def(def, *x), fm_def(def, *y))
                && b1 == b2
            {
                st.chain_cancel += 1;
                return Some(Instr::Div(*s, *a, *c));
            }
            None
        }
        _ => None,
    }
}

/// SLP vectorization pass: collapse lane-parallel elementwise groups in the flat
/// block into width-≤4 vector instrs. Bottom-up, isomorphism-grouped, memoised.
/// Elementwise only (no cross-lane reduction) ⇒ bit-for-bit with the scalar IR.
/// Operates on the single flat block (post-flatten); no-op otherwise.
pub(crate) fn vectorize(tree: &mut Tree, lane_hints: &[Vec<usize>]) {
    if tree.blocks.len() != 1 {
        return;
    }
    let (outputs, fatals) = match tree.blocks[0].term.clone() {
        Term::Return { outputs, fatals } => (outputs, fatals),
        _ => return,
    };
    let mut def: HashMap<u32, Instr> = HashMap::new();
    let mut max_slot = 0u32;
    for ins in &tree.blocks[0].instrs {
        let s = instr_result_slot(ins);
        max_slot = max_slot.max(s);
        def.insert(s, ins.clone());
    }
    let mut v = Vectorizer {
        instrs: tree.blocks[0].instrs.clone(),
        def,
        next: max_slot + 1,
        vmemo: HashMap::new(),
        smemo: HashMap::new(),
        shape_memo: HashMap::new(),
    };
    let new_outputs = v.rewrite_group_hinted(&outputs, lane_hints);
    let new_fatals = v.rewrite_group(&fatals);
    let instrs = std::mem::take(&mut v.instrs);
    tree.blocks[0] = Block {
        instrs,
        term: Term::Return {
            outputs: new_outputs,
            fatals: new_fatals,
        },
    };
    dce(tree); // drop scalar instrs orphaned by the rewrite
}

struct Vectorizer {
    instrs: Vec<Instr>,
    def: HashMap<u32, Instr>,
    next: u32,
    vmemo: HashMap<Vec<u32>, u32>,       // lane tuple -> vector slot
    smemo: HashMap<(Operand, u32), u32>, // (scalar, width) -> splat slot
    shape_memo: HashMap<u32, u64>,
}

impl Vectorizer {
    fn fresh(&mut self) -> u32 {
        let s = self.next;
        self.next += 1;
        s
    }

    /// Structural hash with leaf operands as wildcards: lane-parallel siblings
    /// (same ops, differing only in their leaf/gradient operands) hash equal,
    /// while the base value (different op structure) hashes apart.
    fn shape(&mut self, slot: u32) -> u64 {
        if let Some(&h) = self.shape_memo.get(&slot) {
            return h;
        }
        use std::hash::{Hash, Hasher};
        let ins = match self.def.get(&slot) {
            Some(i) => i.clone(),
            None => return 0,
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        op_tag(&ins).hash(&mut hasher);
        let child_slots: Vec<Option<u32>> = instr_operands(&ins)
            .iter()
            .map(|op| match op {
                Operand::Instr(s) => Some(*s),
                _ => None,
            })
            .collect();
        for cs in child_slots {
            match cs {
                Some(s) => self.shape(s).hash(&mut hasher),
                None => 0u64.hash(&mut hasher), // leaf wildcard
            }
        }
        let h = hasher.finish();
        self.shape_memo.insert(slot, h);
        h
    }

    /// Rewrite outputs using caller lane-hints first (forced groups, chunked to
    /// ≤4), then shape-hash auto-seeding for whatever the hints didn't cover.
    fn rewrite_group_hinted(&mut self, ops: &[Operand], hints: &[Vec<usize>]) -> Vec<Operand> {
        let mut result = ops.to_vec();
        let mut handled = vec![false; ops.len()];
        for group in hints {
            for chunk in group.chunks(4) {
                let lanes: Vec<u32> = chunk
                    .iter()
                    .filter_map(|&i| match ops.get(i) {
                        Some(Operand::Instr(s)) if self.def.contains_key(s) => Some(*s),
                        _ => None,
                    })
                    .collect();
                if lanes.len() < 2 || lanes.len() != chunk.len() || !self.group_vectorizable(&lanes)
                {
                    continue;
                }
                let vslot = self.build_vector(&lanes);
                for (lane, &i) in chunk.iter().enumerate() {
                    let e = self.fresh();
                    self.instrs
                        .push(Instr::Extract(e, Operand::Instr(vslot), lane as u32));
                    result[i] = Operand::Instr(e);
                    handled[i] = true;
                }
            }
        }
        // shape-hash auto for the remainder
        let rest: Vec<Operand> = ops
            .iter()
            .enumerate()
            .map(|(i, op)| if handled[i] { Operand::Eps } else { *op })
            .collect();
        let auto = self.rewrite_group(&rest);
        for (i, op) in auto.into_iter().enumerate() {
            if !handled[i] {
                result[i] = op;
            }
        }
        result
    }

    /// Rewrite a list of output operands: bucket the Instr-producing ones by
    /// shape, vectorise each ≤4 chunk that is genuinely lane-parallel, and route
    /// those outputs through `Extract`. Non-grouped operands pass through.
    fn rewrite_group(&mut self, ops: &[Operand]) -> Vec<Operand> {
        let mut result = ops.to_vec();
        // shape -> ordered list of output indices (preserve output order)
        let mut order: Vec<u64> = Vec::new();
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            if let Operand::Instr(s) = op
                && self.def.contains_key(s)
            {
                let sh = self.shape(*s);
                buckets.entry(sh).or_insert_with(|| {
                    order.push(sh);
                    Vec::new()
                });
                buckets.get_mut(&sh).unwrap().push(i);
            }
        }
        for sh in order {
            let idxs = buckets.remove(&sh).unwrap();
            for chunk in idxs.chunks(4) {
                if chunk.len() < 2 {
                    continue;
                }
                let lanes: Vec<u32> = chunk
                    .iter()
                    .map(|&i| match ops[i] {
                        Operand::Instr(s) => s,
                        _ => unreachable!(),
                    })
                    .collect();
                if !self.group_vectorizable(&lanes) {
                    continue;
                }
                let vslot = self.build_vector(&lanes);
                for (lane, &i) in chunk.iter().enumerate() {
                    let e = self.fresh();
                    self.instrs
                        .push(Instr::Extract(e, Operand::Instr(vslot), lane as u32));
                    result[i] = Operand::Instr(e);
                }
            }
        }
        result
    }

    /// Can these lanes form ONE elementwise vector op? Same op (and uniform
    /// flags / matching cmp + broadcast condition for select).
    fn group_vectorizable(&self, lanes: &[u32]) -> bool {
        let first = match self.def.get(&lanes[0]) {
            Some(i) => i,
            None => return false,
        };
        let rest = || lanes[1..].iter().filter_map(|l| self.def.get(l));
        match first {
            Instr::Neg(..) => lanes[1..]
                .iter()
                .all(|l| matches!(self.def.get(l), Some(Instr::Neg(..)))),
            Instr::Add(..) => lanes[1..]
                .iter()
                .all(|l| matches!(self.def.get(l), Some(Instr::Add(..)))),
            Instr::Sub(..) => lanes[1..]
                .iter()
                .all(|l| matches!(self.def.get(l), Some(Instr::Sub(..)))),
            Instr::Mul(..) => lanes[1..]
                .iter()
                .all(|l| matches!(self.def.get(l), Some(Instr::Mul(..)))),
            Instr::Div(..) => lanes[1..]
                .iter()
                .all(|l| matches!(self.def.get(l), Some(Instr::Div(..)))),
            Instr::Fma {
                neg_prod, neg_c, ..
            } => {
                let (np, nc) = (*neg_prod, *neg_c);
                rest().all(|i| matches!(i, Instr::Fma { neg_prod, neg_c, .. } if *neg_prod == np && *neg_c == nc))
                    && lanes[1..].len() + 1 == lanes.len()
            }
            Instr::Select(_, cmp, _, _, _, _) => {
                // cmp must match; the condition operands may differ per lane (a
                // vector condition is a valid OpSelect) so they are not required
                // to be identical — they get recursed/packed like any operand.
                let cmp = *cmp;
                rest().all(|i| matches!(i, Instr::Select(_, c2, ..) if *c2 == cmp))
            }
            _ => false,
        }
    }

    /// Build a vector value for the lane tuple (memoised). Vectorisable groups
    /// emit one V-op (recursing operands); others `Pack` the lane values.
    fn build_vector(&mut self, lanes: &[u32]) -> u32 {
        if let Some(&s) = self.vmemo.get(lanes) {
            return s;
        }
        let s = if self.group_vectorizable(lanes) {
            let first = self.def.get(&lanes[0]).unwrap().clone();
            match first {
                Instr::Neg(..) => {
                    let a = self.operand_vector(lanes, 0);
                    let s = self.fresh();
                    self.instrs.push(Instr::VNeg(s, a));
                    s
                }
                Instr::Add(..) => self.emit_bin(lanes, Instr::VAdd),
                Instr::Sub(..) => self.emit_bin(lanes, Instr::VSub),
                Instr::Mul(..) => self.emit_bin(lanes, Instr::VMul),
                Instr::Div(..) => self.emit_bin(lanes, Instr::VDiv),
                Instr::Fma {
                    neg_prod, neg_c, ..
                } => {
                    let mut a = self.operand_vector(lanes, 0);
                    if neg_prod {
                        let n = self.fresh();
                        self.instrs.push(Instr::VNeg(n, a));
                        a = Operand::Instr(n);
                    }
                    let b = self.operand_vector(lanes, 1);
                    let mut c = self.operand_vector(lanes, 2);
                    if neg_c {
                        let n = self.fresh();
                        self.instrs.push(Instr::VNeg(n, c));
                        c = Operand::Instr(n);
                    }
                    let s = self.fresh();
                    self.instrs.push(Instr::VFma(s, a, b, c));
                    s
                }
                Instr::Select(_, cmp, _, _, _, _) => {
                    let x = self.operand_vector(lanes, 0);
                    let y = self.operand_vector(lanes, 1);
                    let a = self.operand_vector(lanes, 2);
                    let b = self.operand_vector(lanes, 3);
                    let s = self.fresh();
                    self.instrs.push(Instr::VSelect(s, cmp, x, y, a, b));
                    s
                }
                _ => unreachable!("group_vectorizable gated the op"),
            }
        } else {
            let s = self.fresh();
            let ops = lanes.iter().map(|&l| Operand::Instr(l)).collect();
            self.instrs.push(Instr::Pack(s, ops));
            s
        };
        self.vmemo.insert(lanes.to_vec(), s);
        s
    }

    fn emit_bin(&mut self, lanes: &[u32], mk: impl Fn(u32, Operand, Operand) -> Instr) -> u32 {
        let a = self.operand_vector(lanes, 0);
        let b = self.operand_vector(lanes, 1);
        let s = self.fresh();
        self.instrs.push(mk(s, a, b));
        s
    }

    /// The vector operand at position `pos` across the lanes: a `Splat` if shared,
    /// a recursed vector if all are instr-results, else a `Pack` of scalars.
    fn operand_vector(&mut self, lanes: &[u32], pos: usize) -> Operand {
        let w = lanes.len() as u32;
        let toks: Vec<Operand> = lanes
            .iter()
            .map(|&l| *instr_operands(self.def.get(&l).unwrap())[pos])
            .collect();
        if toks.iter().all(|t| *t == toks[0]) {
            let key = (toks[0], w);
            if let Some(&s) = self.smemo.get(&key) {
                return Operand::Instr(s);
            }
            let s = self.fresh();
            self.instrs.push(Instr::Splat(s, w, toks[0]));
            self.smemo.insert(key, s);
            return Operand::Instr(s);
        }
        if toks.iter().all(|t| matches!(t, Operand::Instr(_))) {
            let sub: Vec<u32> = toks
                .iter()
                .map(|t| match t {
                    Operand::Instr(s) => *s,
                    _ => unreachable!(),
                })
                .collect();
            return Operand::Instr(self.build_vector(&sub));
        }
        let s = self.fresh();
        self.instrs.push(Instr::Pack(s, toks));
        Operand::Instr(s)
    }
}

/// A discriminant for shape hashing (vectorisable ops only need to be told
/// apart; everything else collapses to a single non-vectorisable tag).
fn op_tag(ins: &Instr) -> u8 {
    match ins {
        Instr::Neg(..) => 1,
        Instr::Add(..) => 2,
        Instr::Sub(..) => 3,
        Instr::Mul(..) => 4,
        Instr::Div(..) => 11,
        Instr::Fma {
            neg_prod: false,
            neg_c: false,
            ..
        } => 5,
        Instr::Fma {
            neg_prod: true,
            neg_c: false,
            ..
        } => 6,
        Instr::Fma {
            neg_prod: false,
            neg_c: true,
            ..
        } => 7,
        Instr::Fma {
            neg_prod: true,
            neg_c: true,
            ..
        } => 8,
        Instr::Select(_, crate::ir::CmpKind::Eq, ..) => 9,
        Instr::Select(_, crate::ir::CmpKind::Lt, ..) => 10,
        _ => 0,
    }
}

/// Classify a caught panic payload. Narrow recognition of expected kinds so real
/// bugs fall into `Unexpected`.
pub(crate) fn classify(payload: Box<dyn Any + Send>) -> FatalKind {
    let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("");
    if msg == "viete::fatal::div_by_zero" {
        FatalKind::DivByZero
    } else if msg.contains("`None` value") {
        FatalKind::NonInvertible
    } else {
        FatalKind::Unexpected(msg.to_string())
    }
}

/// Slots referenced by a terminator (branch cond / return outputs).
fn term_used(term: &Term) -> std::collections::HashSet<u32> {
    let mut s = std::collections::HashSet::new();
    let mut note = |op: &Operand| {
        if let Operand::Instr(slot) = op {
            s.insert(*slot);
        }
    };
    match term {
        Term::Branch { cond, .. } => note(cond),
        Term::Return { outputs, fatals } => {
            outputs.iter().for_each(&mut note);
            fatals.iter().for_each(&mut note);
        }
        Term::Fatal { .. } => {}
    }
    s
}

fn term_children(term: &Term) -> Vec<usize> {
    match term {
        Term::Branch {
            on_true, on_false, ..
        } => vec![on_true.0 as usize, on_false.0 as usize],
        _ => vec![],
    }
}

/// Block-scoped DCE: drop instructions whose result slot is referenced by neither
/// this block (later / terminator) nor any descendant. Returns the set of
/// ANCESTOR-defined slots this subtree still uses (propagated up to the parent).
fn dce_block(tree: &mut Tree, id: usize) -> std::collections::HashSet<u32> {
    let term = tree.blocks[id].term.clone();
    let mut used = term_used(&term);
    for child in term_children(&term) {
        used.extend(dce_block(tree, child));
    }
    let instrs = std::mem::take(&mut tree.blocks[id].instrs);
    let defined: std::collections::HashSet<u32> = instrs.iter().map(instr_result_slot).collect();
    let mut kept = Vec::with_capacity(instrs.len());
    for instr in instrs.into_iter().rev() {
        if used.contains(&instr_result_slot(&instr)) {
            for op in instr_operands(&instr) {
                if let Operand::Instr(s) = op {
                    used.insert(*s);
                }
            }
            kept.push(instr);
        }
    }
    kept.reverse();
    tree.blocks[id].instrs = kept;
    used.retain(|s| !defined.contains(s)); // only ancestor refs flow up
    used
}

pub(crate) fn dce(tree: &mut Tree) {
    dce_block(tree, tree.root.0 as usize);
}

// ===========================================================================
// Invariant hoisting (dominator-based CSE). The trie is a TREE (prefix-merge),
// so dominator = ancestor and "lowest dominating block of a set" = tree LCA.
// ===========================================================================

/// Parent block index per block (`usize::MAX` for the root).
fn parent_map(tree: &Tree) -> Vec<usize> {
    let mut parent = vec![usize::MAX; tree.blocks.len()];
    for i in 0..tree.blocks.len() {
        for ch in term_children(&tree.blocks[i].term) {
            parent[ch] = i;
        }
    }
    parent
}

/// Mutable operands of a terminator (branch cond / return outputs).
fn term_operands_mut(term: &mut Term) -> Vec<&mut Operand> {
    match term {
        Term::Branch { cond, .. } => vec![cond],
        Term::Return { outputs, fatals } => outputs.iter_mut().chain(fatals.iter_mut()).collect(),
        Term::Fatal { .. } => vec![],
    }
}

/// Rewrite every `Operand::Instr(slot)` in a block (instrs + term) via `f`.
fn map_block_refs(tree: &mut Tree, b: usize, mut f: impl FnMut(u32) -> u32) {
    for ins in &mut tree.blocks[b].instrs {
        for op in instr_operands_mut(ins) {
            if let Operand::Instr(s) = op {
                *s = f(*s);
            }
        }
    }
    for op in term_operands_mut(&mut tree.blocks[b].term) {
        if let Operand::Instr(s) = op {
            *s = f(*s);
        }
    }
}

/// Rename result slots and `Operand::Instr` refs to globally-unique ids via a
/// scoped DFS. Returns the number of globals assigned (= next free id).
fn normalize_global(tree: &mut Tree) -> u32 {
    let mut counter = 0u32;
    let mut map: HashMap<u32, u32> = HashMap::new(); // local slot -> global id (scoped)
    norm_block(tree, tree.root.0 as usize, &mut counter, &mut map);
    counter
}

fn norm_block(tree: &mut Tree, b: usize, counter: &mut u32, map: &mut HashMap<u32, u32>) {
    let mut saved: Vec<(u32, Option<u32>)> = Vec::new();
    let n = tree.blocks[b].instrs.len();
    for i in 0..n {
        // rewrite operands (reference earlier defs already in `map`) BEFORE
        // assigning this instr's own global id.
        let local = instr_result_slot(&tree.blocks[b].instrs[i]);
        for op in instr_operands_mut(&mut tree.blocks[b].instrs[i]) {
            if let Operand::Instr(s) = op {
                *s = *map.get(s).expect("operand references a defined slot");
            }
        }
        let g = *counter;
        *counter += 1;
        set_instr_result_slot(&mut tree.blocks[b].instrs[i], g);
        saved.push((local, map.insert(local, g)));
    }
    for op in term_operands_mut(&mut tree.blocks[b].term) {
        if let Operand::Instr(s) = op {
            *s = *map.get(s).expect("term operand references a defined slot");
        }
    }
    let children = term_children(&tree.blocks[b].term);
    for ch in children {
        norm_block(tree, ch, counter, map);
    }
    for (local, old) in saved.into_iter().rev() {
        match old {
            Some(g) => {
                map.insert(local, g);
            }
            None => {
                map.remove(&local);
            }
        }
    }
}

/// Inverse of `normalize_global`: assign per-path local slots (reset per path,
/// shared prefix) so the output matches viete's slot scheme.
fn denormalize(tree: &mut Tree) {
    let mut map: HashMap<u32, u32> = HashMap::new(); // global id -> local slot
    denorm_block(tree, tree.root.0 as usize, 0, &mut map);
}

fn denorm_block(tree: &mut Tree, b: usize, base: u32, map: &mut HashMap<u32, u32>) {
    let mut next = base;
    let n = tree.blocks[b].instrs.len();
    for i in 0..n {
        for op in instr_operands_mut(&mut tree.blocks[b].instrs[i]) {
            if let Operand::Instr(g) = op {
                *g = *map.get(g).expect("operand references a defined value");
            }
        }
        let g = instr_result_slot(&tree.blocks[b].instrs[i]);
        let local = next;
        next += 1;
        set_instr_result_slot(&mut tree.blocks[b].instrs[i], local);
        map.insert(g, local);
    }
    for op in term_operands_mut(&mut tree.blocks[b].term) {
        if let Operand::Instr(g) = op {
            *g = *map.get(g).expect("term operand references a defined value");
        }
    }
    let children = term_children(&tree.blocks[b].term);
    for ch in children {
        denorm_block(tree, ch, next, map);
    }
}

/// Tree LCA of a non-empty set of blocks.
fn lca_set(blocks: &[usize], parent: &[usize]) -> usize {
    let mut acc = blocks[0];
    for &b in &blocks[1..] {
        acc = lca2(acc, b, parent);
    }
    acc
}

fn lca2(a: usize, b: usize, parent: &[usize]) -> usize {
    let mut chain = HashSet::new();
    let mut x = a;
    loop {
        chain.insert(x);
        if x == usize::MAX {
            break;
        }
        x = parent[x];
    }
    let mut y = b;
    while !chain.contains(&y) {
        y = parent[y]; // root is in `chain`, so this terminates
    }
    y
}

/// Is `anc` an ancestor-or-equal of `node`?
fn dominates(anc: usize, mut node: usize, parent: &[usize]) -> bool {
    loop {
        if node == anc {
            return true;
        }
        if node == usize::MAX {
            return false;
        }
        node = parent[node];
    }
}

/// Pure & total instructions are hoist candidates; `Div` is excluded (it lives
/// below an exact-zero guard — hoisting above it would compute a trapping x/0).
fn is_hoistable(ins: &Instr) -> bool {
    !matches!(ins, Instr::Div(..))
}

/// Reorder a block's instrs so any same-block operand is defined before its use
/// (hoisting appends to the LCA's end, which can violate ordering).
fn topo_block(instrs: &mut Vec<Instr>) {
    let here: HashSet<u32> = instrs.iter().map(instr_result_slot).collect();
    let mut emitted: HashSet<u32> = HashSet::new();
    let mut remaining = std::mem::take(instrs);
    let mut out = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let mut progressed = false;
        let mut next = Vec::new();
        for ins in remaining.drain(..) {
            let ready = instr_operands(&ins).iter().all(|op| match op {
                Operand::Instr(s) => !here.contains(s) || emitted.contains(s),
                _ => true,
            });
            if ready {
                emitted.insert(instr_result_slot(&ins));
                out.push(ins);
                progressed = true;
            } else {
                next.push(ins);
            }
        }
        remaining = next;
        assert!(progressed, "cycle in block instrs (must be a DAG)");
    }
    *instrs = out;
}

/// Lift pure ops replicated across blocks to their tree LCA (operands permitting),
/// to a fixpoint. Operates on the globally-normalized tree; `next_global` is the
/// first free global id.
fn hoist_fixpoint(tree: &mut Tree, mut next_global: u32) {
    let parent = parent_map(tree);
    loop {
        // def_block: global id -> block defining it
        let mut def_block: HashMap<u32, usize> = HashMap::new();
        for (bi, b) in tree.blocks.iter().enumerate() {
            for ins in &b.instrs {
                def_block.insert(instr_result_slot(ins), bi);
            }
        }
        // group identical pure ops (key = instr with result slot zeroed)
        let mut groups: HashMap<Instr, Vec<u32>> = HashMap::new();
        for b in &tree.blocks {
            for ins in &b.instrs {
                if !is_hoistable(ins) {
                    continue;
                }
                let mut key = ins.clone();
                set_instr_result_slot(&mut key, 0);
                groups.entry(key).or_default().push(instr_result_slot(ins));
            }
        }
        // candidates: groups with >= 2 occurrences. Sort by minimum member id so
        // the round is deterministic (global ids come from the structural
        // normalize DFS, not from HashMap iteration order).
        let mut candidates: Vec<(Instr, Vec<u32>)> = groups
            .into_iter()
            .filter(|(_, ids)| ids.len() >= 2)
            .collect();
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by_key(|(_, ids)| *ids.iter().min().unwrap());

        // Build the whole round's batch. A full batch is safe: each group's
        // operands dominate its LCA, and for any cross-group dependency Y->X the
        // placement LCA_X dominates LCA_Y, so the single uniform rewrite below
        // always lands references on a dominating definition.
        let mut new_instrs: Vec<(usize, Instr)> = Vec::new();
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut removed: HashSet<u32> = HashSet::new();
        for (key, ids) in candidates {
            let blocks: Vec<usize> = ids.iter().map(|g| def_block[g]).collect();
            let l = lca_set(&blocks, &parent);
            debug_assert!(
                instr_operands(&key).iter().all(|op| match op {
                    Operand::Instr(g) => dominates(def_block[g], l, &parent),
                    _ => true,
                }),
                "group operand must dominate its LCA (per-path SSA invariant)"
            );
            let g = next_global;
            next_global += 1;
            let mut new_ins = key.clone();
            set_instr_result_slot(&mut new_ins, g);
            new_instrs.push((l, new_ins));
            for id in ids {
                remap.insert(id, g);
                removed.insert(id);
            }
        }
        // 1) remove every occurrence
        for b in &mut tree.blocks {
            b.instrs
                .retain(|ins| !removed.contains(&instr_result_slot(ins)));
        }
        // 2) append each merged op at its LCA
        for (l, ins) in new_instrs {
            tree.blocks[l].instrs.push(ins);
        }
        // 3) ONE uniform rewrite over all blocks (incl. the just-appended instrs)
        for b in 0..tree.blocks.len() {
            map_block_refs(tree, b, |s| *remap.get(&s).unwrap_or(&s));
        }
    }
}

/// Intern a rational into the (post-build) pool, returning its index.
fn pool_index(tree: &mut Tree, r: num_rational::Rational32) -> u32 {
    if let Some(i) = tree.consts.iter().position(|&c| c == r) {
        return i as u32;
    }
    tree.consts.push(r);
    (tree.consts.len() - 1) as u32
}

/// Append `status` to every `Return.fatals` reachable from block `start`.
fn add_fatal_to_returns(tree: &mut Tree, start: usize, status: Operand) {
    let children = term_children(&tree.blocks[start].term);
    if let Term::Return { fatals, .. } = &mut tree.blocks[start].term {
        fatals.push(status);
    }
    for ch in children {
        add_fatal_to_returns(tree, ch, status);
    }
}

fn is_fatal(tree: &Tree, b: usize) -> bool {
    matches!(tree.blocks[b].term, Term::Fatal { .. })
}

/// Collapse every `Branch` with exactly one `Fatal`-leaf arm into a fatal-status
/// slot + the inlined live arm. Operates in the normalized global-id space (so
/// merging the live child's instrs is plain concatenation). `next_global` is the
/// first free global id; bumped as `Abs`/`Select` are added.
fn collapse_fatal_branches(tree: &mut Tree, next_global: &mut u32) {
    let one = pool_index(tree, num_rational::Rational32::new(1, 1));
    let zero = pool_index(tree, num_rational::Rational32::new(0, 1));
    collapse_block(tree, tree.root.0 as usize, next_global, one, zero);
}

fn collapse_block(tree: &mut Tree, b: usize, ng: &mut u32, one: u32, zero: u32) {
    // `loop { match … }` (not `while let`): the body mutates `tree`, so the
    // term-borrow must be released each iteration before the mutation.
    #[allow(clippy::while_let_loop)]
    loop {
        let (cond, test, cond_sign, on_true, on_false) = match &tree.blocks[b].term {
            Term::Branch {
                cond,
                test,
                cond_sign,
                on_true,
                on_false,
            } => (
                *cond,
                *test,
                *cond_sign,
                on_true.0 as usize,
                on_false.0 as usize,
            ),
            _ => break,
        };
        let t_fatal = is_fatal(tree, on_true);
        let f_fatal = is_fatal(tree, on_false);
        if t_fatal == f_fatal {
            break; // neither, or both (both: leave as-is — does not occur here)
        }
        let live = if t_fatal { on_false } else { on_true };
        // status == 1 exactly when the FATAL arm would be taken (on_true is taken
        // when `test` holds): fatal-on-true -> (1,0); fatal-on-false -> (0,1).
        let (m, s) = if t_fatal { (one, zero) } else { (zero, one) };
        let (lhs, rhs, cmp) = match test {
            BranchTest::EffectiveZero => {
                let lhs = match cond_sign {
                    CondSign::NonNeg => cond, // abs(cond) == cond
                    CondSign::NonPos => {
                        let neg_id = *ng;
                        *ng += 1;
                        tree.blocks[b].instrs.push(Instr::Neg(neg_id, cond));
                        Operand::Instr(neg_id)
                    }
                    CondSign::Unknown => {
                        let abs_id = *ng;
                        *ng += 1;
                        tree.blocks[b].instrs.push(Instr::Abs(abs_id, cond));
                        Operand::Instr(abs_id)
                    }
                };
                (lhs, Operand::Eps, crate::ir::CmpKind::Lt)
            }
            BranchTest::ExactZero => (cond, Operand::Const(zero), crate::ir::CmpKind::Eq),
        };
        let status_id = *ng;
        *ng += 1;
        tree.blocks[b].instrs.push(Instr::Select(
            status_id,
            cmp,
            lhs,
            rhs,
            Operand::Const(m),
            Operand::Const(s),
        ));
        add_fatal_to_returns(tree, live, Operand::Instr(status_id));
        // inline the live arm into b (global ids -> concatenation is conflict-free)
        let live_instrs = std::mem::take(&mut tree.blocks[live].instrs);
        let live_term = std::mem::replace(
            &mut tree.blocks[live].term,
            Term::Fatal {
                kind: FatalKind::DivByZero, // orphan sentinel; block becomes unreachable
            },
        );
        tree.blocks[b].instrs.extend(live_instrs);
        tree.blocks[b].term = live_term;
        // loop: b's new term may itself be another Fatal-arm branch
    }
    let children = term_children(&tree.blocks[b].term);
    for ch in children {
        collapse_block(tree, ch, ng, one, zero);
    }
}

/// Drop blocks unreachable from the root and renumber `BlockId`s (collapsing
/// orphans the Fatal arm + the merged live block; they must not reach
/// hoist/leaf_count/lua).
fn compact_blocks(tree: &mut Tree) {
    let n = tree.blocks.len();
    let mut reachable = vec![false; n];
    let mut stack = vec![tree.root.0 as usize];
    while let Some(b) = stack.pop() {
        if reachable[b] {
            continue;
        }
        reachable[b] = true;
        for ch in term_children(&tree.blocks[b].term) {
            stack.push(ch);
        }
    }
    let mut remap = vec![u32::MAX; n];
    let mut next = 0u32;
    for (i, &r) in reachable.iter().enumerate() {
        if r {
            remap[i] = next;
            next += 1;
        }
    }
    let old = std::mem::take(&mut tree.blocks);
    let mut compact = Vec::with_capacity(next as usize);
    for (i, mut block) in old.into_iter().enumerate() {
        if !reachable[i] {
            continue;
        }
        if let Term::Branch {
            on_true, on_false, ..
        } = &mut block.term
        {
            *on_true = BlockId(remap[on_true.0 as usize]);
            *on_false = BlockId(remap[on_false.0 as usize]);
        }
        compact.push(block);
    }
    tree.blocks = compact;
    tree.root = BlockId(remap[tree.root.0 as usize]);
}

/// Dominator-based invariant hoisting (see module header): normalize to global
/// ids, collapse Fatal-arm branches into the fatal buffer, lift replicated pure
/// ops to their LCA, topo-reorder, de-normalize.
/// Next free global id after the normalized phase (= max result slot + 1).
fn next_free_global(tree: &Tree) -> u32 {
    tree.blocks
        .iter()
        .flat_map(|b| b.instrs.iter())
        .map(instr_result_slot)
        .max()
        .map_or(0, |m| m + 1)
}

/// Recursively if-convert the subtree at `b`: compute both arms of every
/// EffectiveZero branch, select outputs by `abs(cond) < eps`, mask each arm's
/// fatals by whether that arm is selected. Returns (instrs, outputs, fatals).
fn flatten(
    tree: &Tree,
    b: usize,
    ng: &mut u32,
    c0: Operand,
) -> (Vec<Instr>, Vec<Operand>, Vec<Operand>) {
    let mut instrs = tree.blocks[b].instrs.clone();
    match tree.blocks[b].term.clone() {
        Term::Return { outputs, fatals } => (instrs, outputs, fatals),
        Term::Branch {
            cond,
            test,
            cond_sign,
            on_true,
            on_false,
        } => {
            debug_assert_eq!(
                test,
                BranchTest::EffectiveZero,
                "collapse removes ExactZero branches"
            );
            let (ti, to, tf) = flatten(tree, on_true.0 as usize, ng, c0);
            let (fi, fo, ff) = flatten(tree, on_false.0 as usize, ng, c0);
            instrs.extend(ti);
            instrs.extend(fi);
            debug_assert_eq!(to.len(), fo.len(), "both arms return K outputs");
            // abs(cond): elided when the cond's sign is known.
            let a = match cond_sign {
                CondSign::NonNeg => cond, // abs(cond) == cond
                CondSign::NonPos => {
                    let neg_id = *ng;
                    *ng += 1;
                    instrs.push(Instr::Neg(neg_id, cond)); // abs(cond) == -cond
                    Operand::Instr(neg_id)
                }
                CondSign::Unknown => {
                    let abs_id = *ng;
                    *ng += 1;
                    instrs.push(Instr::Abs(abs_id, cond));
                    Operand::Instr(abs_id)
                }
            };
            // outputs: abs(cond) < eps ? Taylor(true) : closed(false)
            let mut outputs = Vec::with_capacity(to.len());
            for k in 0..to.len() {
                let id = *ng;
                *ng += 1;
                instrs.push(Instr::Select(
                    id,
                    crate::ir::CmpKind::Lt,
                    a,
                    Operand::Eps,
                    to[k],
                    fo[k],
                ));
                outputs.push(Operand::Instr(id));
            }
            // fatals: keep true-arm's when abs<eps, false-arm's when abs>=eps
            let mut fatals = Vec::with_capacity(tf.len() + ff.len());
            for f in tf {
                let id = *ng;
                *ng += 1;
                instrs.push(Instr::Select(
                    id,
                    crate::ir::CmpKind::Lt,
                    a,
                    Operand::Eps,
                    f,
                    c0,
                ));
                fatals.push(Operand::Instr(id));
            }
            for f in ff {
                let id = *ng;
                *ng += 1;
                instrs.push(Instr::Select(
                    id,
                    crate::ir::CmpKind::Lt,
                    a,
                    Operand::Eps,
                    c0,
                    f,
                ));
                fatals.push(Operand::Instr(id));
            }
            (instrs, outputs, fatals)
        }
        Term::Fatal { .. } => unreachable!("collapse_fatal_branches removed all Fatal leaves"),
    }
}

/// If-convert the whole (collapsed, hoisted) tree into a single straight-line
/// block: every EffectiveZero branch becomes select(s) over both arms.
fn flatten_branches(tree: &mut Tree) {
    let c0 = Operand::Const(pool_index(tree, num_rational::Rational32::new(0, 1)));
    let mut ng = next_free_global(tree);
    let (instrs, outputs, fatals) = flatten(tree, tree.root.0 as usize, &mut ng, c0);
    tree.blocks = vec![Block {
        instrs,
        term: Term::Return { outputs, fatals },
    }];
    tree.root = BlockId(0);
}

pub(crate) fn hoist_invariant(tree: &mut Tree, flatten: bool) {
    let mut next_global = normalize_global(tree);
    collapse_fatal_branches(tree, &mut next_global);
    compact_blocks(tree);
    hoist_fixpoint(tree, next_global);
    if flatten {
        flatten_branches(tree);
    }
    for b in &mut tree.blocks {
        topo_block(&mut b.instrs);
    }
    denormalize(tree);
}

/// Fuse `Mul` + a consuming `Add`/`Sub` into one `Fma`, canonicalizing every
/// single-use `Neg` into the two sign bits. Pure rewrite: the orphaned `Mul`/`Neg`
/// are left for DCE. Runs post-DCE so the single-use gate sees true live counts.
pub(crate) fn fma_contract(tree: &mut Tree) {
    contract_subtree(tree, tree.root.0 as usize);
}

/// Post-order: contract block `id` using use-counts gathered over ITS subtree
/// (this block's refs + all descendants'), then return those counts so the parent
/// can merge them. Slots reset per path (`start_path`), so sibling blocks reuse
/// slot numbers; a single global count would conflate them and wrongly block
/// in-arm contractions. Subtree-scoping is exact: a slot defined in `id` is
/// referenced unambiguously within id's subtree (descendants number strictly
/// higher, so they never redefine it).
fn contract_subtree(tree: &mut Tree, id: usize) -> HashMap<u32, u32> {
    let mut uses: HashMap<u32, u32> = HashMap::new();
    for ch in term_children(&tree.blocks[id].term) {
        for (slot, n) in contract_subtree(tree, ch) {
            *uses.entry(slot).or_insert(0) += n;
        }
    }
    {
        let mut bump = |op: &Operand| {
            if let Operand::Instr(s) = op {
                *uses.entry(*s).or_insert(0) += 1;
            }
        };
        for ins in &tree.blocks[id].instrs {
            for op in instr_operands(ins) {
                bump(op);
            }
        }
        match &tree.blocks[id].term {
            Term::Branch { cond, .. } => bump(cond),
            Term::Return { outputs, fatals } => {
                outputs.iter().for_each(&mut bump);
                fatals.iter().for_each(&mut bump);
            }
            Term::Fatal { .. } => {}
        }
    }
    contract_block(tree, id, &uses);
    uses
}

/// Rewrite `Add`/`Sub` in one block into `Fma` where the product is a same-block,
/// single-use `Mul`. Same-block lookup (defs is this block only) enforces the
/// block-local gate; cross-block products are never found, so never contracted.
fn contract_block(tree: &mut Tree, id: usize, use_count: &HashMap<u32, u32>) {
    let defs: HashMap<u32, Instr> = tree.blocks[id]
        .instrs
        .iter()
        .map(|ins| (instr_result_slot(ins), ins.clone()))
        .collect();
    let instrs = std::mem::take(&mut tree.blocks[id].instrs);
    let mut out = Vec::with_capacity(instrs.len());
    for ins in instrs {
        let rewritten = match &ins {
            Instr::Add(s, x, y) => try_fma(*s, *x, *y, false, &defs, use_count),
            Instr::Sub(s, x, y) => try_fma(*s, *x, *y, true, &defs, use_count),
            _ => None,
        };
        out.push(rewritten.unwrap_or(ins));
    }
    tree.blocks[id].instrs = out;
}

/// Build the `Fma` for `x (+/-) y` if one side is a contractible product.
/// `is_sub` distinguishes `Add(x,y)=x+y` from `Sub(x,y)=x-y`.
fn try_fma(
    s: u32,
    x: Operand,
    y: Operand,
    is_sub: bool,
    defs: &HashMap<u32, Instr>,
    use_count: &HashMap<u32, u32>,
) -> Option<Instr> {
    // product on the left: x is never negated by + or -, addend y is subtracted iff Sub.
    if let Some((a, b, neg_prod)) = as_product(x, defs, use_count) {
        let (c, neg_c) = addend(y, is_sub, defs, use_count);
        return Some(Instr::Fma {
            s,
            neg_prod,
            a,
            b,
            c,
            neg_c,
        });
    }
    // product on the right: in Sub the product is subtracted (negate it); addend x is positive.
    if let Some((a, b, neg_prod)) = as_product(y, defs, use_count) {
        let (c, neg_c) = addend(x, false, defs, use_count);
        return Some(Instr::Fma {
            s,
            neg_prod: neg_prod ^ is_sub,
            a,
            b,
            c,
            neg_c,
        });
    }
    None
}

/// If `op` is a same-block single-use `Mul` (optionally wrapped in a single-use
/// `Neg`), return its Neg-stripped factors and the product's sign.
fn as_product(
    op: Operand,
    defs: &HashMap<u32, Instr>,
    use_count: &HashMap<u32, u32>,
) -> Option<(Operand, Operand, bool)> {
    let Operand::Instr(k) = op else { return None };
    if use_count.get(&k) != Some(&1) {
        return None;
    }
    match defs.get(&k)? {
        Instr::Mul(_, fa, fb) => {
            let (a, b, neg) = fold_factor_negs(*fa, *fb, defs, use_count);
            Some((a, b, neg))
        }
        Instr::Neg(_, Operand::Instr(m)) => {
            if use_count.get(m) != Some(&1) {
                return None;
            }
            if let Some(Instr::Mul(_, fa, fb)) = defs.get(m) {
                let (a, b, neg) = fold_factor_negs(*fa, *fb, defs, use_count);
                Some((a, b, !neg)) // the Neg wrapper negates the whole product
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Fold single-use factor `Neg`s into the product sign (both factors collapse
/// onto one bit, so `b` need not carry a sign). Multi-use `Neg`s stay referenced.
fn fold_factor_negs(
    fa: Operand,
    fb: Operand,
    defs: &HashMap<u32, Instr>,
    use_count: &HashMap<u32, u32>,
) -> (Operand, Operand, bool) {
    let mut neg = false;
    let mut factors = [fa, fb];
    for f in factors.iter_mut() {
        if let Operand::Instr(j) = *f
            && use_count.get(&j) == Some(&1)
            && let Some(Instr::Neg(_, inner)) = defs.get(&j)
        {
            neg = !neg;
            *f = *inner;
        }
    }
    (factors[0], factors[1], neg)
}

/// Resolve the addend operand and its sign: start from the operator sign
/// (`base_neg`), then fold a single-use `Neg` on the addend into that sign.
fn addend(
    op: Operand,
    base_neg: bool,
    defs: &HashMap<u32, Instr>,
    use_count: &HashMap<u32, u32>,
) -> (Operand, bool) {
    if let Operand::Instr(j) = op
        && use_count.get(&j) == Some(&1)
        && let Some(Instr::Neg(_, inner)) = defs.get(&j)
    {
        return (*inner, !base_neg);
    }
    (op, base_neg)
}

#[cfg(test)]
pub(crate) fn abs_band_lt_rules() {
    use crate::ir::Instr;
    use num_rational::Rational32;
    with_fresh_engine(|e| {
        let x = Handle::Runtime(7);
        e.abs_lt_eps.insert(x);
        let neg = Handle::Runtime(8);
        seed_facts(e, neg, &Instr::Neg(8, Operand::Instr(7)), x, None);
        assert!(e.is_abs_lt_eps(neg), "Neg keeps abs<eps");
        let m = Handle::Runtime(9);
        let c = Handle::Constant(Rational32::new(1, 8));
        seed_facts(
            e,
            m,
            &Instr::Mul(9, Operand::Instr(7), Operand::Const(0)),
            x,
            Some(c),
        );
        assert!(e.is_abs_lt_eps(m), "Mul by |c|<=1/2 keeps abs<eps");
        let s = Handle::Runtime(10);
        seed_facts(e, s, &Instr::Sqrt(10, Operand::Instr(7)), x, None);
        assert!(!e.is_abs_lt_eps(s), "Sqrt does NOT keep abs<eps");
    });
}

#[cfg(test)]
pub(crate) fn abs_band_gt_rules() {
    use crate::ir::Instr;
    use num_rational::Rational32;
    with_fresh_engine(|e| {
        let x = Handle::Runtime(7);
        e.abs_gt_eps.insert(x);
        let s = Handle::Runtime(8);
        seed_facts(e, s, &Instr::Sqrt(8, Operand::Instr(7)), x, None);
        assert!(e.is_abs_gt_eps(s), "Sqrt keeps abs>eps");
        let m = Handle::Runtime(9);
        let c = Handle::Constant(Rational32::new(4, 1));
        seed_facts(
            e,
            m,
            &Instr::Mul(9, Operand::Instr(7), Operand::Const(0)),
            x,
            Some(c),
        );
        assert!(e.is_abs_gt_eps(m), "Mul by |c|>=2 keeps abs>eps");
        let m2 = Handle::Runtime(10);
        let c2 = Handle::Constant(Rational32::new(3, 2));
        seed_facts(
            e,
            m2,
            &Instr::Mul(10, Operand::Instr(7), Operand::Const(1)),
            x,
            Some(c2),
        );
        assert!(!e.is_abs_gt_eps(m2), "Mul by |c| in (1/2,2) proves nothing");
    });
}

#[cfg(test)]
pub(crate) fn sign_propagation_rules() {
    use crate::ir::{Instr, Operand};
    use num_rational::Rational32;
    with_fresh_engine(|e| {
        let x = Handle::Runtime(1); // unknown-sign runtime value
        // self-square Mul(x, x) ==> nonneg (any real), positive once nonzero
        let sq = Handle::Runtime(2);
        seed_facts(
            e,
            sq,
            &Instr::Mul(2, Operand::Instr(1), Operand::Instr(1)),
            x,
            Some(x),
        );
        assert!(
            e.is_nonneg(sq) && !e.is_positive(sq),
            "x^2 >= 0, sign of x unknown"
        );
        e.nonzero.insert(x);
        let sq2 = Handle::Runtime(3);
        seed_facts(
            e,
            sq2,
            &Instr::Mul(3, Operand::Instr(1), Operand::Instr(1)),
            x,
            Some(x),
        );
        assert!(e.is_positive(sq2), "x != 0 ==> x^2 > 0");

        // Neg flips
        let p = Handle::Runtime(4);
        e.positive.insert(p);
        let np = Handle::Runtime(5);
        seed_facts(e, np, &Instr::Neg(5, Operand::Instr(4)), p, None);
        assert!(e.is_negative(np), "-(positive) is negative");

        // Mul sign table: neg * neg => pos
        let n1 = Handle::Runtime(6);
        let n2 = Handle::Runtime(7);
        e.negative.insert(n1);
        e.negative.insert(n2);
        let m = Handle::Runtime(8);
        seed_facts(
            e,
            m,
            &Instr::Mul(8, Operand::Instr(6), Operand::Instr(7)),
            n1,
            Some(n2),
        );
        assert!(e.is_positive(m), "neg * neg ==> pos");

        // Div sign table: nonneg / nonpos => nonpos
        let a = Handle::Runtime(9);
        let b = Handle::Runtime(10);
        e.nonnegative.insert(a);
        e.nonpositive.insert(b);
        let d = Handle::Runtime(11);
        seed_facts(
            e,
            d,
            &Instr::Div(11, Operand::Instr(9), Operand::Instr(10)),
            a,
            Some(b),
        );
        assert!(e.is_nonpos(d), "nonneg / nonpos ==> nonpos");

        // Add: nonneg + nonneg => nonneg
        let s = Handle::Runtime(12);
        seed_facts(
            e,
            s,
            &Instr::Add(12, Operand::Instr(9), Operand::Instr(2)),
            a,
            Some(sq),
        );
        assert!(e.is_nonneg(s), "nonneg + nonneg ==> nonneg");

        // Sub: nonneg - nonpos => nonneg
        let sub = Handle::Runtime(13);
        seed_facts(
            e,
            sub,
            &Instr::Sub(13, Operand::Instr(9), Operand::Instr(10)),
            a,
            Some(b),
        );
        assert!(e.is_nonneg(sub), "nonneg - nonpos ==> nonneg");

        // Sqrt => nonneg; Exp => positive & (exp(nonneg) >= 1)
        let r = Handle::Runtime(14);
        seed_facts(e, r, &Instr::Sqrt(14, Operand::Instr(1)), x, None);
        assert!(e.is_nonneg(r), "sqrt >= 0");
        let ex = Handle::Runtime(15);
        seed_facts(e, ex, &Instr::Exp(15, Operand::Instr(9)), a, None); // a is nonneg
        assert!(
            e.is_positive(ex) && e.is_abs_ge_one(ex),
            "exp(nonneg) > 0 and >= 1"
        );

        // Powi: even => nonneg; odd => preserves sign
        let pe = Handle::Runtime(16);
        let two = Handle::Constant(Rational32::new(2, 1));
        seed_facts(
            e,
            pe,
            &Instr::Powi(16, Operand::Instr(1), Operand::Const(0)),
            x,
            Some(two),
        );
        assert!(e.is_nonneg(pe), "x^2 >= 0 via Powi");
        let po = Handle::Runtime(17);
        let three = Handle::Constant(Rational32::new(3, 1));
        seed_facts(
            e,
            po,
            &Instr::Powi(17, Operand::Instr(6), Operand::Const(0)),
            n1,
            Some(three),
        );
        assert!(e.is_negative(po), "neg^3 < 0");

        // Powf => nonneg
        let pf = Handle::Runtime(18);
        seed_facts(
            e,
            pf,
            &Instr::Powf(18, Operand::Instr(1), Operand::Instr(9)),
            x,
            Some(a),
        );
        assert!(e.is_nonneg(pf), "powf >= 0 (real domain)");

        // additive eps-band: (positive & abs_gt_eps) + nonneg ==> abs_gt_eps
        let big = Handle::Runtime(19);
        e.positive.insert(big);
        e.abs_gt_eps.insert(big);
        let summ = Handle::Runtime(20);
        seed_facts(
            e,
            summ,
            &Instr::Add(20, Operand::Instr(19), Operand::Instr(9)),
            big,
            Some(a),
        );
        assert!(e.is_abs_gt_eps(summ), "big>eps + nonneg stays >eps");

        // nonneg AND nonpos derived together ==> effective zero sentinel
        let z = Handle::Runtime(21);
        e.nonnegative.insert(z);
        e.nonpositive.insert(z);
        let zz = Handle::Runtime(22);
        seed_facts(e, zz, &Instr::Neg(22, Operand::Instr(21)), z, None);
        assert!(
            e.is_abs_lt_eps(zz),
            "value both >=0 and <=0 is effective zero"
        );
    });
}

#[cfg(test)]
pub(crate) fn sign_query_rules() {
    use num_rational::Rational32;
    with_fresh_engine(|e| {
        // constants resolve by numerator sign; zero is nonneg AND nonpos
        let pos_c = Handle::Constant(Rational32::new(3, 2));
        let neg_c = Handle::Constant(Rational32::new(-3, 2));
        let zero_c = Handle::Constant(Rational32::new(0, 1));
        assert!(e.is_positive(pos_c) && e.is_nonneg(pos_c) && !e.is_nonpos(pos_c));
        assert!(e.is_negative(neg_c) && e.is_nonpos(neg_c) && !e.is_nonneg(neg_c));
        assert!(e.is_nonneg(zero_c) && e.is_nonpos(zero_c));
        assert!(!e.is_positive(zero_c) && !e.is_negative(zero_c));

        // runtime nonneg + nonzero ==> positive by closure
        let x = Handle::Runtime(7);
        e.nonnegative.insert(x);
        assert!(
            e.is_nonneg(x) && !e.is_positive(x),
            "nonneg alone is not positive"
        );
        e.nonzero.insert(x);
        assert!(e.is_positive(x), "nonneg + nonzero ==> positive");

        // input facts drive sign queries
        e.input_facts
            .entry(0)
            .or_default()
            .push(crate::ir::InputFact::Positive);
        assert!(e.is_positive(Handle::Input(0)) && e.is_nonneg(Handle::Input(0)));
    });
}

#[cfg(test)]
pub(crate) fn reset_for_test() {
    ENGINE.with(|c| *c.borrow_mut() = Engine::new());
}

#[cfg(test)]
pub(crate) fn drain_events_for_test() -> Vec<Event> {
    ENGINE.with(|c| std::mem::take(&mut c.borrow_mut().events))
}

/// Test helper: run a closure against a fresh engine.
#[cfg(test)]
pub(crate) fn with_fresh_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    ENGINE.with(|c| {
        *c.borrow_mut() = Engine::new();
        f(&mut c.borrow_mut())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Tracer;
    #[cfg(feature = "lua")]
    use crate::ir::{Block, BlockId, CmpKind, Operand, Tree};
    use crate::ir::{Instr, Term};
    use crate::sym::Sym;
    use clifford::pga3::{Motor, Twist};
    use num_rational::Rational32;
    use peano::prelude::*;

    #[test]
    fn motor_exp_leaf_count() {
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

        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(6, 0, 16, &[], |inp, _| exp_coeffs::<Sym>(inp));

        // Forward NonZero propagation suppresses 2 of 3 Div guards.
        // The remaining DivByZero is on the structurally-impossible path
        // [u EffectiveZero=true] AND [u*DSINC_SCALE EffectiveZero=false],
        // which requires |u| < eps AND |u|*(1/100000) >= eps — a contradiction
        // that forward propagation alone cannot prove dead (backward alias
        // analysis of u*k = nonzero => u = nonzero is outside this task scope).
        // abs<eps propagates from the [abs(u)<eps] arm through u*DSINC_SCALE
        // (|1e-5| <= 1/2), so EffectiveZero(u*scale) is determined true and the
        // structurally-dead [abs(u)<eps] AND [abs(u*scale)>=eps] path is pruned.
        // abs<eps propagates from the [abs(u)<eps] arm through u*const, determining
        // the dependent EffectiveZero seams and pruning the dead Div-guard paths.
        assert_eq!(
            traced.leaf_count(),
            3,
            "abs-band determination prunes the dead Div-guard paths"
        );
        assert_eq!(
            traced.fatal_leaves().len(),
            0,
            "no fatal arms survive on valid data"
        );
        assert_eq!(traced.unexpected_count(), 0);
        // every Return leaf returns all 16 coefficients
        for b in &traced.tree().blocks {
            if let crate::ir::Term::Return { outputs, .. } = &b.term {
                assert_eq!(outputs.len(), 16);
            }
        }
    }

    #[test]
    fn div_by_runtime_is_branchless_with_fatal() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(2, 0, 1, &[], |inp, _| vec![inp[0] / inp[1]]);

        // No ExactZero fork: one straight-line leaf, no Fatal{DivByZero}.
        assert_eq!(traced.leaf_count(), 1, "division no longer forks");
        assert!(!traced.has_fatal(), "no Fatal leaf for runtime div-by-zero");
        let lua = traced.emit_lua();
        assert!(!lua.contains("== 0.0"), "no exact-zero branch");
        assert!(!lua.contains("error("), "no error() for div-by-zero");
        // the divisor (inp[1], not proven nonzero) gets a fatal-status select
        assert!(lua.contains("select("), "fatal-status select emitted");
    }

    #[test]
    fn same_handle_tested_twice_collapses_to_one_branch() {
        use crate::api::Tracer;
        use crate::ir::Term;

        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(1, 0, 1, &[], |inp, _| {
            let x = inp[0] + inp[0]; // a runtime value (one SSA slot)
            let _x = x.is_effective_zero(); // fork 1
            let _x = x.is_effective_zero(); // fork 2 — same handle, must be memoized
            vec![x]
        });
        let branches = traced
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        assert_eq!(
            branches, 1,
            "same-handle/test double check must collapse to ONE branch"
        );
        assert_eq!(traced.leaf_count(), 2);
    }

    #[cfg(feature = "lua")]
    #[test]
    fn normalize_recip_collapses_fatal_to_buffer() {
        use crate::api::Tracer;
        use crate::ir::Term;
        use crate::sym::Sym;

        // recip of a runtime value: honest try_recip forks (invertible vs zero);
        // the collapse pass turns the NonInvertible Fatal arm into a fatal slot.
        fn recip_of_sum<S>(inp: &[S]) -> Vec<S>
        where
            S: Scalar + StandardPart + FromRational + Copy,
        {
            let s = inp[0] + inp[1];
            vec![s.try_recip().unwrap()]
        }

        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(2, 0, 1, &[], |inp, _| recip_of_sum::<Sym>(inp));

        // No Fatal leaf: collapsed to a single straight-line Return + fatal slot.
        assert!(
            !traced.has_fatal(),
            "NonInvertible collapsed, no Fatal leaf"
        );
        assert_eq!(traced.unexpected_count(), 0);
        let returns = traced
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Return { .. }))
            .count();
        assert_eq!(returns, 1, "one straight-line return");
        // s = 3 -> invertible -> Ok(1/3); s = 0 -> fatal -> Err
        assert!((traced.run_lua(&[1.0, 2.0], &[]).unwrap()[0] - (1.0 / 3.0)).abs() < 1e-12);
        assert!(traced.run_lua(&[1.0, -1.0], &[]).is_err(), "s=0 -> Err");
    }

    #[test]
    fn unexpected_panic_is_classified_not_masked() {
        use crate::api::Tracer;
        use crate::ir::FatalKind;

        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(1, 0, 1, &[], |_inp, _| -> Vec<_> {
            panic!("boom unexpected")
        });
        assert_eq!(
            traced.unexpected_count(),
            1,
            "tracer-bug panic must surface as Unexpected"
        );
        match &traced.fatal_leaves()[0].1 {
            FatalKind::Unexpected(msg) => assert!(msg.contains("boom")),
            other => panic!("expected Unexpected, got {other:?}"),
        }
    }

    #[test]
    fn static_div_by_zero_is_classified() {
        use crate::api::Tracer;
        use crate::ir::FatalKind;
        use crate::sym::Sym;

        let tracer = Tracer::builder().build();
        // dividing by a structural-zero constant panics with the div_by_zero sentinel
        let traced = tracer.trace::<_>(1, 0, 1, &[], |inp, _| -> Vec<_> {
            vec![inp[0] / Sym::from_u32(0)]
        });
        assert_eq!(traced.leaf_count(), 1);
        let fatals = traced.fatal_leaves();
        assert_eq!(fatals.len(), 1);
        assert_eq!(fatals[0].1, FatalKind::DivByZero);
        assert_eq!(traced.unexpected_count(), 0);
    }

    #[test]
    fn intern_dedupes_by_value() {
        with_fresh_engine(|e| {
            let a = e.intern_const(Rational32::new(1, 6));
            let b = e.intern_const(Rational32::new(1, 120));
            let a2 = e.intern_const(Rational32::new(1, 6)); // same value
            assert_eq!(a, a2, "same value must give same index");
            assert_ne!(a, b);
            assert_eq!(e.consts.len(), 2, "only two distinct constants pooled");
        });
    }

    #[test]
    fn slots_count_up_from_zero() {
        with_fresh_engine(|e| {
            assert_eq!(e.alloc_slot(), 0);
            assert_eq!(e.alloc_slot(), 1);
        });
    }

    #[test]
    fn is_nonzero_base_cases() {
        use crate::sym::Handle;
        use num_rational::Rational32;
        crate::engine::reset_for_test();
        crate::engine::ENGINE.with(|c| {
            let e = &mut *c.borrow_mut();
            e.start_path(vec![]);
            assert!(
                e.is_nonzero(Handle::Constant(Rational32::new(2, 1))),
                "nonzero const"
            );
            assert!(
                !e.is_nonzero(Handle::Constant(Rational32::new(0, 1))),
                "zero const"
            );
            assert!(!e.is_nonzero(Handle::Input(0)), "input unknown");
            let r = Handle::Runtime(0);
            assert!(!e.is_nonzero(r), "runtime not seeded");
            e.nonzero.insert(r);
            assert!(e.is_nonzero(r), "runtime seeded");
        });
    }

    fn count_kind(tree: &crate::ir::Tree, pred: impl Fn(&Instr) -> bool) -> usize {
        tree.blocks
            .iter()
            .map(|b| b.instrs.iter().filter(|i| pred(i)).count())
            .sum()
    }

    /// Generic kernel for the intra-batch dependency test. Outer fork (F1) on i2
    /// computes `p = i0*i1` in BOTH arms (group X). Inner fork (F2) on i0 uses
    /// `sin(p)` in BOTH sub-arms (group Y, operand = the A1-arm's p). So X and Y
    /// are BOTH groups in the SAME round, and Y depends on an id X removes.
    #[cfg(feature = "lua")]
    fn intra_batch_kernel<S>(i: &[S]) -> Vec<S>
    where
        S: Scalar + StandardPart + EffectiveZero + Copy,
    {
        if i[2].is_effective_zero() {
            let p = i[0] * i[1];
            if i[0].is_effective_zero() {
                vec![p.sin_explicit() + i[0]]
            } else {
                vec![p.sin_explicit() + i[1]]
            }
        } else {
            let p = i[0] * i[1];
            vec![p.cos_explicit()]
        }
    }

    #[cfg(feature = "lua")]
    #[test]
    fn hoist_intra_batch_dependency() {
        use crate::api::Tracer;
        // X = Mul(i0,i1) in both F1 arms; Y = sin(X) in both F2 sub-arms. Y and X
        // are BOTH groups in round 1, and Y's operand is one of X's removed ids —
        // the uniform rewrite (applied to the appended instrs too) must redirect
        // Y to gX in the same round.
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(3, 0, 1, &[], |i, _| intra_batch_kernel::<Sym>(i));
        assert_eq!(
            count_kind(traced.tree(), |x| matches!(x, Instr::Mul(..))),
            1,
            "X (shared Mul) collapsed once"
        );
        assert_eq!(
            count_kind(traced.tree(), |x| matches!(x, Instr::Sin(..))),
            1,
            "Y (sin over X) collapsed once"
        );
        // bit-for-bit through each arm proves Y references gX, not a removed id
        // (a dangling ref would make run_lua read a nil slot -> panic/garbage).
        let cases = [
            vec![0.5f64, 0.7, 0.0],
            vec![0.5, 0.7, 0.3],
            vec![1e-12, 0.7, 0.0],
        ];
        for inp in cases {
            let got = traced.run_lua(&inp, &[]).unwrap();
            let want = intra_batch_kernel::<f64>(&inp);
            assert!(
                (got[0] - want[0]).abs() < 1e-12,
                "intra-batch mismatch at {inp:?}: {} vs {}",
                got[0],
                want[0]
            );
        }
    }

    #[test]
    fn hoist_batches_independent_groups() {
        use crate::api::Tracer;
        // sin(i0) and cos(i1) both replicated across the two arms; independent ->
        // both collapse in a single round.
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(3, 0, 1, &[], |i, _| {
            if i[2].is_effective_zero() {
                vec![i[0].sin_explicit() + i[1].cos_explicit()]
            } else {
                vec![i[0].sin_explicit() - i[1].cos_explicit()]
            }
        });
        assert_eq!(
            count_kind(traced.tree(), |x| matches!(x, Instr::Sin(..))),
            1
        );
        assert_eq!(
            count_kind(traced.tree(), |x| matches!(x, Instr::Cos(..))),
            1
        );
    }

    #[test]
    fn hoist_cascades_dependent_groups() {
        use crate::api::Tracer;
        // g = sin(i0), h = sin(g), both built INSIDE each arm (post-fork). g groups
        // first; h becomes groupable only after g merges -> both collapse over rounds.
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(2, 0, 1, &[], |i, _| {
            if i[1].is_effective_zero() {
                let g = i[0].sin_explicit();
                vec![g.sin_explicit() + i[1]]
            } else {
                let g = i[0].sin_explicit();
                vec![g.sin_explicit() - i[1]]
            }
        });
        // exactly 2 distinct Sin nodes survive (inner + outer), each collapsed once.
        assert_eq!(
            count_kind(traced.tree(), |x| matches!(x, Instr::Sin(..))),
            2
        );
    }

    #[test]
    fn hoist_is_deterministic() {
        use crate::api::Tracer;
        let build = || {
            Tracer::builder()
                .build()
                .trace::<_>(3, 0, 1, &[], |i, _| {
                    if i[2].is_effective_zero() {
                        vec![i[0].sin_explicit() + i[1].cos_explicit()]
                    } else {
                        vec![i[0].sin_explicit() - i[1].cos_explicit()]
                    }
                })
                .emit_lua()
        };
        assert_eq!(
            build(),
            build(),
            "hoisting output must be byte-stable across runs"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn hoist_collapses_replicated_pure_op() {
        // sin(i0) is computed in BOTH arms (post-fork) and is unary (fma never
        // fuses it), so it must collapse to one Sin hoisted to the LCA (root).
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(2, 0, 1, &[], |i, _| {
            if i[1].is_effective_zero() {
                vec![i[0].sin_explicit() + i[1]]
            } else {
                vec![i[0].sin_explicit() - i[1]]
            }
        });
        let sins = count_kind(traced.tree(), |x| matches!(x, Instr::Sin(..)));
        assert_eq!(
            sins, 1,
            "replicated sin hoisted to the LCA (root), computed once"
        );
        let s = (0.5f64).sin();
        assert!((traced.run_lua(&[0.5, 0.0], &[]).unwrap()[0] - s).abs() < 1e-12);
        assert!((traced.run_lua(&[0.5, 0.3], &[]).unwrap()[0] - (s - 0.3)).abs() < 1e-12);
    }

    #[test]
    fn hoist_leaves_single_branch_op_in_place() {
        // cos(i0) appears in ONE arm only -> must NOT be hoisted (no always-compute).
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(2, 0, 1, &[], |i, _| {
            if i[1].is_effective_zero() {
                vec![i[0].cos_explicit()]
            } else {
                vec![i[0]]
            }
        });
        let root_has_cos = traced.tree().blocks[0]
            .instrs
            .iter()
            .any(|x| matches!(x, Instr::Cos(..)));
        assert!(!root_has_cos, "single-branch cos stays in its arm");
        assert_eq!(
            count_kind(traced.tree(), |x| matches!(x, Instr::Cos(..))),
            1
        );
    }

    #[test]
    fn hoist_excludes_div() {
        // i0/i1 guarded in each arm; the Div must NOT be hoisted above its guards.
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(3, 0, 1, &[], |i, _| {
            if i[2].is_effective_zero() {
                vec![i[0] / i[1] + i[2]]
            } else {
                vec![i[0] / i[1] - i[2]]
            }
        });
        let root_has_div = traced.tree().blocks[0]
            .instrs
            .iter()
            .any(|x| matches!(x, Instr::Div(..)));
        assert!(
            !root_has_div,
            "Div must never hoist above its exact-zero guard"
        );
    }

    #[test]
    fn hoist_is_idempotent() {
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(2, 0, 1, &[], |i, _| {
            if i[1].is_effective_zero() {
                vec![i[0].sin_explicit() + i[1]]
            } else {
                vec![i[0].sin_explicit() - i[1]]
            }
        });
        let once = traced.tree().clone();
        let mut twice = once.clone();
        super::hoist_invariant(&mut twice, false);
        assert_eq!(twice.blocks, once.blocks, "second hoist pass is a no-op");
    }

    #[test]
    fn vectorize_flag_noop_is_identity() {
        // stub vectorize is a no-op: tree with the flag equals tree without it.
        let f = |i: &Vec<Sym>, _: &_| vec![i[0].sin_explicit() + i[1]];
        let plain = Tracer::builder()
            .flatten()
            .build()
            .trace::<_>(2, 0, 1, &[], f);
        let vec = Tracer::builder()
            .flatten()
            .vectorize()
            .build()
            .trace::<_>(2, 0, 1, &[], f);
        assert_eq!(
            plain.tree().blocks,
            vec.tree().blocks,
            "stub vectorize is a no-op"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn handbuilt_vec_tree_runs_bitforbit() {
        // hand-built: [a*c, b*c] via Splat(c) + VMul + two Extract.
        // inputs: a=inp0, b=inp1, c=inp2.
        let blk = Block {
            instrs: vec![
                Instr::Pack(0, vec![Operand::Input(0), Operand::Input(1)]), // {a,b}
                Instr::Splat(1, 2, Operand::Input(2)),                      // {c,c}
                Instr::VMul(2, Operand::Instr(0), Operand::Instr(1)),       // {a*c, b*c}
                Instr::Extract(3, Operand::Instr(2), 0),                    // a*c
                Instr::Extract(4, Operand::Instr(2), 1),                    // b*c
            ],
            term: Term::Return {
                outputs: vec![Operand::Instr(3), Operand::Instr(4)],
                fatals: vec![],
            },
        };
        let tree = Tree {
            inputs: 3,
            params: 0,
            consts: vec![],
            blocks: vec![blk],
            root: BlockId(0),
        };
        let _x = CmpKind::Eq;
        let src = crate::lua::emit_lua(&tree, 1e-9, "t");
        assert!(src.contains("vmul("), "vector mul emitted");
        assert!(src.contains("splat("), "splat emitted");
        let lua = mlua::Lua::new();
        lua.load(src).exec().expect("lua load");
        let f: mlua::Function = lua.globals().get("t").expect("entry fn");
        let inp = lua.create_table().unwrap();
        for (i, v) in [2.0, 5.0, 3.0].iter().enumerate() {
            inp.set(i + 1, *v).unwrap();
        }
        let (out, _fatal): (mlua::Table, mlua::Table) =
            f.call((inp, lua.create_table().unwrap())).expect("call");
        let got: Vec<f64> = (1..=2).map(|i| out.get::<f64>(i).unwrap()).collect();
        assert_eq!(got, vec![6.0, 15.0], "[a*c, b*c] bit-for-bit");
    }

    #[cfg(feature = "lua")]
    #[test]
    fn lane_parallel_pair_vectorizes() {
        // [a*c, b*c]: two Mul lanes sharing c -> one VMul + Splat(c) + Pack(a,b).
        let f = |i: &Vec<Sym>, _: &_| {
            let c = i[2];
            vec![i[0] * c, i[1] * c]
        };
        let t = Tracer::builder()
            .flatten()
            .vectorize()
            .build()
            .trace::<_>(3, 0, 2, &[], f);
        let instrs: Vec<&Instr> = t.tree().blocks.iter().flat_map(|b| &b.instrs).collect();
        assert!(
            instrs.iter().any(|x| matches!(x, Instr::VMul(..))),
            "lane-parallel muls vectorize"
        );
        assert!(
            instrs.iter().any(|x| matches!(x, Instr::Splat(..))),
            "shared operand splatted"
        );
        assert_eq!(
            t.run_lua(&[2.0, 5.0, 3.0], &[]).unwrap(),
            [6.0, 15.0],
            "bit-for-bit"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn lane_parallel_div_vectorizes() {
        // [a/c, b/c]: two Div lanes sharing the divisor c -> VDiv + Splat(c).
        // c is given AbsGtEps so the divisions are proven nonzero (no fatal here).
        let f = |i: &Vec<Sym>, _: &_| {
            let c = i[2];
            vec![i[0] / c, i[1] / c]
        };
        let t = Tracer::builder().flatten().vectorize().build().trace::<_>(
            3,
            0,
            2,
            &[(
                crate::ir::InputRef::Input(2),
                crate::ir::InputFact::AbsGtEps,
            )],
            f,
        );
        let instrs: Vec<&Instr> = t.tree().blocks.iter().flat_map(|b| &b.instrs).collect();
        assert!(
            instrs.iter().any(|x| matches!(x, Instr::VDiv(..))),
            "lane-parallel divisions vectorize"
        );
        assert_eq!(
            t.run_lua(&[6.0, 15.0, 3.0], &[]).unwrap(),
            [2.0, 5.0],
            "bit-for-bit"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn vdiv_preserves_fatal_mechanism() {
        // [a/c, b/c] with c NOT proven nonzero -> guarded divisions: VDiv for the
        // values, plus the separate (c==0) fatal-status selects. Vectorizing the
        // Div must not disturb the fatal buffer.
        let f = |i: &Vec<Sym>, _: &_| {
            let c = i[2];
            vec![i[0] / c, i[1] / c]
        };
        let t = Tracer::builder()
            .flatten()
            .vectorize()
            .build()
            .trace::<_>(3, 0, 2, &[], f);
        let instrs: Vec<&Instr> = t.tree().blocks.iter().flat_map(|b| &b.instrs).collect();
        assert!(
            instrs.iter().any(|x| matches!(x, Instr::VDiv(..))),
            "VDiv emitted"
        );
        // nonzero divisor -> Ok, bit-for-bit
        assert_eq!(t.run_lua(&[6.0, 15.0, 3.0], &[]).unwrap(), [2.0, 5.0]);
        // zero divisor -> the fatal slot fires -> Err (mechanism intact)
        assert!(
            t.run_lua(&[6.0, 15.0, 0.0], &[]).is_err(),
            "div-by-zero still trapped after vectorization"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn non_isomorphic_pair_does_not_vectorize() {
        // [a*b, a+b]: different ops -> different shape buckets -> no V-op.
        let f = |i: &Vec<Sym>, _: &_| vec![i[0] * i[1], i[0] + i[1]];
        let t = Tracer::builder()
            .flatten()
            .vectorize()
            .build()
            .trace::<_>(2, 0, 2, &[], f);
        let any_vec = t.tree().blocks.iter().flat_map(|b| &b.instrs).any(|x| {
            matches!(
                x,
                Instr::VMul(..) | Instr::VAdd(..) | Instr::VFma(..) | Instr::VSub(..)
            )
        });
        assert!(!any_vec, "heterogeneous ops do not group");
        assert_eq!(
            t.run_lua(&[2.0, 5.0], &[]).unwrap(),
            [10.0, 7.0],
            "bit-for-bit"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn repr_select_flattens_to_one_block() {
        let tracer = Tracer::builder().flatten().build();
        // one representation-select fork, both arms live (no division/fatal)
        let traced = tracer.trace::<_>(1, 0, 1, &[], |i: &Vec<Sym>, _| {
            if i[0].is_effective_zero() {
                vec![Sym::from_u32(7)]
            } else {
                vec![i[0]]
            }
        });
        let branches = traced
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        assert_eq!(branches, 0, "representation-select flattened away");
        assert_eq!(traced.leaf_count(), 1, "one straight-line block");
        // abs(x) < eps -> Taylor arm (7); else -> closed arm (x)
        assert_eq!(
            traced.run_lua(&[0.0], &[]).unwrap(),
            [7.0],
            "effective-zero -> 7"
        );
        assert_eq!(traced.run_lua(&[5.0], &[]).unwrap(), [5.0], "nonzero -> x");
    }

    #[cfg(feature = "lua")]
    #[test]
    fn noninvertible_collapses_to_fatal() {
        let tracer = Tracer::builder().build();
        // try_recip().unwrap(): false-arm = Div(ONE,x); true-arm (abs<eps) -> None
        // -> unwrap panic -> Fatal{NonInvertible}. The collapse pass removes it.
        let traced = tracer.trace::<_>(1, 0, 1, &[], |i: &Vec<Sym>, _| {
            vec![i[0].try_recip().unwrap()]
        });
        assert!(!traced.has_fatal(), "NonInvertible Fatal leaf collapsed");
        let lua = traced.emit_lua();
        assert!(!lua.contains("error("), "no error() for noninvertible");
        assert!(lua.contains("math.abs("), "abs(cond) emitted");
        assert!(lua.contains("select_lt("), "Lt fatal-status select emitted");
        // invertible -> Ok(1/x); zero / effective-zero -> Err
        assert_eq!(traced.run_lua(&[2.0], &[]).unwrap(), [0.5]);
        assert!(traced.run_lua(&[0.0], &[]).is_err(), "x=0 -> fatal -> Err");
        assert!(
            traced.run_lua(&[1e-12], &[]).is_err(),
            "effective-zero -> Err"
        );
    }

    #[test]
    fn normalize_denormalize_is_identity() {
        // a branchy trace so the trie has post-fork blocks with reused slots
        let tr = Tracer::builder().build();
        let traced = tr.trace::<_>(3, 0, 2, &[], |i, _| {
            let x = i[0] * i[1];
            if i[2].is_effective_zero() {
                vec![x + i[2], i[0] - i[1]]
            } else {
                vec![x - i[0], i[1] * i[2]]
            }
        });
        let before = traced.tree().clone();
        let mut after = before.clone();
        super::normalize_global(&mut after);
        super::denormalize(&mut after);
        assert_eq!(
            after.blocks, before.blocks,
            "normalize∘denormalize must be identity"
        );
    }

    #[test]
    fn param_resolves_to_high_index_input() {
        crate::engine::reset_for_test();
        crate::engine::ENGINE.with(|c| {
            let e = &mut *c.borrow_mut();
            e.start_path(vec![]);
            e.inputs = 5; // pretend INPUTS = 5
            // operand(Param(2)) must lower to Input(5 + 2)
            assert_eq!(e.operand(Handle::Param(2)), crate::ir::Operand::Input(7));
            // a fact on the resolved input is seen through the Param handle
            e.input_facts
                .entry(7)
                .or_default()
                .push(crate::ir::InputFact::AbsGeOne);
            assert!(
                e.is_abs_ge_one(Handle::Param(2)),
                "param sees fact at INPUTS+j"
            );
            assert!(e.is_nonzero(Handle::Param(2)));
        });
    }

    #[test]
    fn sym_param_is_a_pure_constructor() {
        // No engine interaction: constructing a param emits no events.
        crate::engine::reset_for_test();
        let p = Sym::param(3);
        assert!(matches!(p.0, Handle::Param(3)));
        assert!(crate::engine::drain_events_for_test().is_empty());
    }

    #[test]
    fn div_by_nonzero_const_is_suppressed() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(1, 0, 1, &[], |inp: &Vec<Sym>, _| {
            vec![inp[0] / Sym::from_u32(2)]
        });
        assert_eq!(
            traced.leaf_count(),
            1,
            "divisor is a nonzero const -> guard suppressed, no fork"
        );
        assert!(!traced.has_fatal());
    }

    #[test]
    fn mul_and_sqrt_propagate_nonzero() {
        let tracer = Tracer::builder().build();
        let mul = tracer.trace::<_>(3, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let x = inp[0] + inp[1];
            if x.is_effective_zero() {
                vec![x]
            } else {
                vec![inp[2] / (x * x)]
            }
        });
        assert_eq!(
            mul.leaf_count(),
            2,
            "Mul of nonzeros stays nonzero -> div suppressed"
        );
        assert!(!mul.has_fatal());
        let sq = tracer.trace::<_>(3, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let x = inp[0] + inp[1];
            if x.is_effective_zero() {
                vec![x]
            } else {
                vec![inp[2] / x.sqrt_explicit()]
            }
        });
        assert_eq!(
            sq.leaf_count(),
            2,
            "Sqrt of nonzero stays nonzero -> div suppressed"
        );
        assert!(!sq.has_fatal());
    }

    #[test]
    fn add_result_division_gets_fatal_slot() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(3, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let s = inp[0] + inp[1];
            vec![inp[2] / s]
        });
        // Add result is not known-nonzero -> branchless, one leaf + a fatal slot.
        assert_eq!(traced.leaf_count(), 1, "division is branchless");
        assert!(
            traced.emit_lua().contains("select("),
            "unsafe divisor -> fatal-status select"
        );
    }

    #[test]
    fn double_negation_is_peepholed() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(1, 0, 1, &[], |inp: &Vec<Sym>, _| vec![-(-inp[0])]);
        // outer Neg collapses to the inner operand; the emitted Lua has no nested negation
        let lua = traced.emit_lua();
        assert!(!lua.contains("-(-"), "no double negation in emitted code");
        // the single output is inp[0] itself (1-based -> inp[1]), returned directly
        assert!(
            lua.contains("return {inp[1]}"),
            "output is the bare input, not a re-negation"
        );
    }

    #[test]
    fn duplicate_sqrt_is_cse_collapsed() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(2, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let x = inp[0] + inp[1];
            let a = x.sqrt_explicit();
            let b = x.sqrt_explicit(); // identical to `a` -> must dedup to one Sqrt
            vec![a + b]
        });
        let sqrts = traced.emit_lua().matches("math.sqrt").count();
        assert_eq!(
            sqrts, 1,
            "duplicate sqrt(x) collapses to a single instruction"
        );
    }

    #[test]
    fn dce_drops_unused_instruction() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(2, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let _dead = inp[0] * inp[0]; // computed, never used in the output
            vec![inp[1]]
        });
        assert!(
            traced.tree().blocks[0].instrs.is_empty(),
            "the unused Mul is dead-code-eliminated from the root block"
        );
    }

    #[test]
    fn dce_keeps_slot_live_only_in_a_deep_leaf() {
        let tracer = Tracer::builder().build();
        // x is defined in the ROOT block but used ONLY in the else leaf (not in root,
        // not in the then leaf). Cross-block liveness must keep it.
        let traced = tracer.trace::<_>(2, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let x = inp[0] * inp[1];
            if inp[0].is_effective_zero() {
                vec![inp[1]]
            } else {
                vec![x]
            }
        });
        assert_eq!(
            traced.tree().blocks[0].instrs.len(),
            1,
            "x (used only in a deep leaf) survives DCE in the root block"
        );
    }

    #[test]
    fn div_chain_is_branchless() {
        let tracer = Tracer::builder().build();
        let traced = tracer.trace::<_>(4, 0, 1, &[], |inp: &Vec<Sym>, _| {
            let x = inp[0] + inp[1];
            if x.is_effective_zero() {
                vec![x]
            } else {
                let q = inp[2] / x; // x proven nonzero here -> no fatal slot
                vec![inp[3] / q] // q not proven nonzero -> fatal slot, no fork
            }
        });
        // Only the effective_zero fork makes 2 leaves; both divisions are branchless.
        assert_eq!(traced.leaf_count(), 2, "divisions add no forks");
        assert!(!traced.has_fatal(), "no Fatal leaves from division");
    }

    // Returns the sign bits + operands of the single Fma in a single-block tree.
    fn only_fma(
        t: &crate::api::Trace,
    ) -> (
        bool,
        bool,
        crate::ir::Operand,
        crate::ir::Operand,
        crate::ir::Operand,
    ) {
        let root = &t.tree().blocks[0];
        let fmas: Vec<&Instr> = root
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::Fma { .. }))
            .collect();
        assert_eq!(fmas.len(), 1, "exactly one Fma");
        assert!(
            !root.instrs.iter().any(|i| matches!(i, Instr::Mul(..))),
            "Mul DCE'd after fusion"
        );
        match fmas[0] {
            Instr::Fma {
                neg_prod,
                neg_c,
                a,
                b,
                c,
                ..
            } => (*neg_prod, *neg_c, *a, *b, *c),
            _ => unreachable!(),
        }
    }

    #[test]
    fn abs_band_propagates_through_neg_sqrt_and_const_mul() {
        abs_band_lt_rules(); // abs<eps: Neg keeps, Mul-by-1/8 keeps, Sqrt does NOT
        abs_band_gt_rules(); // abs>eps: Neg/Sqrt keep, Mul-by-4 keeps, Mul-by-3/2 not
    }

    #[test]
    fn sign_queries_resolve_const_input_runtime() {
        sign_query_rules();
    }

    #[test]
    fn sign_propagation_through_ops() {
        sign_propagation_rules();
    }

    #[test]
    fn flatten_drops_abs_for_nonneg_cond() {
        // an EffectiveZero fork on a nonneg square; flattened lowering.
        let tracer = Tracer::builder().flatten().build();
        let traced = tracer.trace::<_>(1, 0, 1, &[], |i: &Vec<Sym>, _| {
            let s = i[0] * i[0]; // nonneg square
            let out = if s.is_effective_zero() { Sym::ONE } else { s };
            vec![out]
        });
        let lua = traced.emit_lua();
        assert!(
            !lua.contains("math.abs"),
            "nonneg cond must lower without math.abs:\n{lua}"
        );
    }

    #[cfg(feature = "lua")]
    #[test]
    fn fast_math_reciprocal_cse_correct_and_fewer_divs() {
        // three divisions by the SAME runtime denominator -> one reciprocal + 3 muls
        fn kernel(i: &[Sym], _p: &[Sym]) -> Vec<Sym> {
            vec![i[0] / i[3], i[1] / i[3], i[2] / i[3]]
        }
        let plain = Tracer::builder()
            .flatten()
            .build()
            .trace::<_>(4, 0, 3, &[], |i, p| kernel(i, p));
        let fast =
            Tracer::builder()
                .flatten()
                .fast_math()
                .build()
                .trace::<_>(4, 0, 3, &[], |i, p| kernel(i, p));
        let inp = vec![6.0, 8.0, 10.0, 2.0];
        let p = plain.run_lua(&inp, &[]).unwrap();
        let f = fast.run_lua(&inp, &[]).unwrap();
        for k in 0..3 {
            assert!(
                (p[k] - f[k]).abs() < 1e-12 && (f[k] - [3.0, 4.0, 5.0][k]).abs() < 1e-12,
                "fast_math value k={k}: plain={} fast={}",
                p[k],
                f[k]
            );
        }
        let plain_divs = plain.emit_lua().matches(" / ").count();
        let fast_divs = fast.emit_lua().matches(" / ").count();
        assert!(
            fast_divs < plain_divs,
            "reciprocal-CSE must cut divisions: {plain_divs} -> {fast_divs}"
        );
    }

    #[test]
    fn fork_stamps_square_cond_nonneg() {
        crate::engine::reset_for_test();
        ENGINE.with(|c| c.borrow_mut().start_path(vec![]));
        // x*x is structurally nonneg; an EffectiveZero fork on it must stamp NonNeg.
        let x = Sym::input(0);
        let sq = x * x;
        let _x = sq.is_effective_zero(); // emits the EffectiveZero branch event
        let events = crate::engine::drain_events_for_test();
        let stamp = events.iter().find_map(|ev| match ev {
            Event::Branch(_, BranchTest::EffectiveZero, s) => Some(*s),
            _ => None,
        });
        assert_eq!(stamp, Some(CondSign::NonNeg), "square cond stamped NonNeg");
    }

    #[test]
    fn effective_zero_arms_seed_abs_band() {
        // Re-checking the SAME handle is determined (no second branch) — pins that
        // the abs-band machinery doesn't regress single-handle determination.
        let tr = crate::api::Tracer::builder().build();
        let t = tr.trace::<_>(1, 0, 1, &[], |i, _| {
            let x = i[0];
            let _x = x.is_effective_zero(); // branch 1
            let _x = x.is_effective_zero(); // determined; no second branch
            vec![x]
        });
        let branches = t
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        assert_eq!(
            branches, 1,
            "second check on same handle is determined, not re-branched"
        );
    }

    #[test]
    fn exp_is_always_nonzero_powi_is_unary() {
        // 1/exp(x): the Div guard must be suppressed — exp is never zero.
        let tr = crate::api::Tracer::builder().build();
        let t = tr.trace::<_>(1, 0, 1, &[], |i, _| {
            let e = i[0].exp_explicit();
            vec![Sym::ONE / e]
        });
        assert!(
            !t.tree().blocks.iter().any(|b| matches!(
                b.term,
                Term::Fatal {
                    kind: FatalKind::DivByZero
                }
            )),
            "exp(x) is nonzero -> no div-by-zero guard"
        );
    }

    #[test]
    fn fma_four_sign_forms() {
        let tr = crate::api::Tracer::builder().build();
        let t_add = tr.trace::<_>(3, 0, 1, &[], |i, _| vec![i[0] * i[1] + i[2]]);
        let t_sub = tr.trace::<_>(3, 0, 1, &[], |i, _| vec![i[0] * i[1] - i[2]]);
        let t_csub = tr.trace::<_>(3, 0, 1, &[], |i, _| vec![i[2] - i[0] * i[1]]);
        let t_nms = tr.trace::<_>(3, 0, 1, &[], |i, _| vec![-(i[0] * i[1]) - i[2]]);

        let (np, nc, _, _, _) = only_fma(&t_add);
        assert_eq!((np, nc), (false, false), "a*b + c");
        let (np, nc, _, _, _) = only_fma(&t_sub);
        assert_eq!((np, nc), (false, true), "a*b - c");
        let (np, nc, _, _, _) = only_fma(&t_csub);
        assert_eq!((np, nc), (true, false), "c - a*b");
        let (np, nc, _, _, _) = only_fma(&t_nms);
        assert_eq!((np, nc), (true, true), "-(a*b) - c");
    }

    #[test]
    fn fma_neg_canonicalization_unifies_forms() {
        // (-a)*b + c  and  -(a*b) + c  are structurally different but the same value.
        let tr = crate::api::Tracer::builder().build();
        let t1 = tr.trace::<_>(3, 0, 1, &[], |i, _| vec![(-i[0]) * i[1] + i[2]]);
        let t2 = tr.trace::<_>(3, 0, 1, &[], |i, _| vec![-(i[0] * i[1]) + i[2]]);

        let fields = |t: &crate::api::Trace| {
            let root = &t.tree().blocks[0];
            for ins in &root.instrs {
                if let Instr::Fma {
                    neg_prod,
                    neg_c,
                    a,
                    b,
                    c,
                    ..
                } = ins
                {
                    return (*neg_prod, *neg_c, *a, *b, *c);
                }
            }
            panic!("no Fma");
        };
        let f1 = fields(&t1);
        let f2 = fields(&t2);
        assert_eq!(f1, f2, "both forms canonicalize to the same Fma");
        assert_eq!((f1.0, f1.1), (true, false), "neg_prod=true, neg_c=false");
    }

    #[test]
    fn fma_multi_use_product_not_contracted() {
        // product reused (gp-style): not single-use -> must NOT contract.
        let tr = crate::api::Tracer::builder().build();
        let t = tr.trace::<_>(3, 0, 2, &[], |i, _| {
            let p = i[0] * i[1];
            vec![p + i[2], p]
        });
        let root = &t.tree().blocks[0];
        assert!(
            root.instrs.iter().any(|i| matches!(i, Instr::Mul(..))),
            "Mul survives (used twice)"
        );
        assert!(
            !root.instrs.iter().any(|i| matches!(i, Instr::Fma { .. })),
            "no Fma on a multi-use product"
        );
    }

    #[test]
    fn fma_cross_block_product_not_contracted() {
        // product in the root (prefix), its only consumer past a fork (child block).
        // single-use, but block-local gate forbids contraction.
        let tr = crate::api::Tracer::builder().build();
        let t = tr.trace::<_>(3, 0, 1, &[], |i, _| {
            let p = i[0] * i[1];
            if i[2].is_effective_zero() {
                vec![p + i[2]]
            } else {
                vec![i[2]]
            }
        });
        let any_fma = t
            .tree()
            .blocks
            .iter()
            .any(|b| b.instrs.iter().any(|i| matches!(i, Instr::Fma { .. })));
        let any_mul = t
            .tree()
            .blocks
            .iter()
            .any(|b| b.instrs.iter().any(|i| matches!(i, Instr::Mul(..))));
        assert!(!any_fma, "no cross-block FMA");
        assert!(any_mul, "the cross-block Mul stays live");
    }

    #[test]
    fn fma_contracts_inside_branch_arms() {
        // Both arms are FNMA (c - a*b); each product is single-use WITHIN its arm,
        // but sibling arms reuse the same per-path slot numbers. A global use-count
        // would conflate them and block both; per-subtree counting must contract both.
        let tr = crate::api::Tracer::builder().build();
        let t = tr.trace::<_>(4, 0, 1, &[], |i, _| {
            let g = i[0] + i[1];
            if g.is_effective_zero() {
                vec![i[2] - i[0] * i[1]]
            } else {
                vec![i[2] - i[3] * i[0]]
            }
        });
        let arms: Vec<_> = t
            .tree()
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Return { .. }))
            .collect();
        assert_eq!(arms.len(), 2, "two return arms");
        for b in &arms {
            assert!(
                b.instrs.iter().any(|i| matches!(i, Instr::Fma { .. })),
                "arm contracts to Fma"
            );
            assert!(
                !b.instrs.iter().any(|i| matches!(i, Instr::Mul(..))),
                "arm Mul DCE'd"
            );
        }
    }

    #[test]
    fn fma_contract_reaches_fixpoint_in_one_pass() {
        // Contraction creates no new Mul+Add/Sub pairs (Fma is inert to matching)
        // and preserves leaf-operand use-counts, so one pass saturates. Re-running
        // fma_contract (+DCE) on an already-contracted tree must be a no-op.
        // Exercises sum-of-products (Fma-with-Mul-addend) and in-arm fms/fnma.
        let tr = crate::api::Tracer::builder().build();
        let traced = tr.trace::<_>(4, 0, 2, &[], |i, _| {
            let s = i[0] * i[1] + i[2] * i[3]; // sum of products: fma(a,b, c*d)
            if i[0].is_effective_zero() {
                vec![s, i[2] - i[0] * i[3]]
            } else {
                vec![s, i[1] * i[2] - i[3]]
            }
        });
        let mut again = traced.tree().clone();
        fma_contract(&mut again);
        dce(&mut again);
        assert_eq!(
            &again.blocks,
            &traced.tree().blocks,
            "second FMA pass is a no-op"
        );
    }
}
