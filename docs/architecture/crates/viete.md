# viete — symbolic PGA3 tracer + backend codegen

Named after **François Viète**, founder of symbolic algebra (*logistica speciosa* —
reckoning with symbols instead of numbers). That is exactly what the crate does: it
runs a `clifford::Scalar` computation but, instead of substituting `f64` values,
operates on symbols/indices and records every operation into a graph.

Landed 2026-06-26. Spec/plan:
`docs/superpowers/{specs,plans}/2026-06-26-{sym-pga3-trace-design,viete-symbolic-tracer}.md` — deleted with the abandoned `docs/superpowers/` corpus; recoverable from git history.

## Purpose

Trace any generic `fn f<S: Scalar>(&[S; N]) -> [S; K]` (e.g. `Motor::exp`) over a
**symbolic** scalar `Sym`, enumerating every `is_effective_zero` branch into a raw
decision-trie IR, then emit a backend program from that IR. Today the backend is
**Lua**, executed on the host via `mlua` and verified against the same computation
on `f64`. The end goal is **GPU shader codegen (SPIR-V)** — viete is the *frontend*,
decoupled from the backend, so codegen correctness can be proven entirely on the
host (no GPU, no driver) before anything reaches the device. The shared `f64` shadow
oracle is the same generic `f`, instantiated at `f64`.

This is the realization of the "PGA throughout includes the shader" / "math substrate
precedes GPU API" direction: pick the scalar + substrate first, trace it, then emit
the shader from a verified IR.

## Architecture

Pipeline: `Sym` ops → engine events → worklist enumerates branches → trie merge →
`Tree` IR → backend emit → (Lua) execute & compare to `f64`.

```
src/
├── ir.rs       Operand / Instr / Block / BlockId / Term{Branch,Return,Fatal} / Tree
│               + queries (leaf_count, collect_consts, fatal_leaves, pretty)
├── sym.rs      Sym (a pure Handle, NO f64 shadow) implementing all clifford traits;
│               arithmetic pushes Instr; is_effective_zero forks; honest try_recip
├── engine.rs   thread-local Engine (event buffer, value-dedup const pool, trail +
│               worklist), fork, emit_bin/emit_un, TreeBuilder (trie merge),
│               run_trace (worklist + catch_unwind fatal classification), HookGuard
├── lua.rs      emit_lua(&Tree, threshold) -> String  (trie -> nested if/else)
└── api.rs      opaque Tracer / TracerBuilder / Trace<INPUTS,PARAMS,OUTPUTS> (public surface)
tests/acceptance.rs    Lua-vs-f64 on Motor::exp (non-trivial + near-seam inputs)
tests/jacobian_smoke.rs 24-DOF Jacobian of a newton joint law, constants fed as params
```

The IR is an **arena** (`Vec<Block>` + `BlockId`), not a `Box` tree — trie-insert is
an index rewrite, not a `mem::replace` dance. "Tree" refers to the *branch* structure;
inside a block the instructions are an ordinary SSA list (operands reference earlier
slots, so `let`-bound subexpressions are naturally shared).

## Key decisions (load-bearing)

1. **Full 2^k branch enumeration, modulo SSA-identity memoization.** Each *distinct*
   runtime condition forks independently (worklist + trail replay), faithful to the GPU
   (each is an independent `OpBranchConditional`); no value model — decisions are forced
   by the worklist. BUT a fork is memoized per path by `(Handle, BranchTest)`: the same
   SSA value tested the same way doesn't re-branch (exp's `cos_sq`+`sinc_sq` both test
   the same `u` → one branch). This is exact (SSA-identity, not value-CSE) and removes
   the provably-dead nested check; different slots with equal runtime value still fork
   independently.
2. **`Sym` is pure-symbolic** — just a `Handle` (`Constant(Rational32)` / `Input` /
   `Runtime` slot), no `f64` shadow. Constant folding + `x*0`/`x*1`/`x+0` special cases
   ride on `Rational32`. The oracle is the same generic `f` at `f64`, not a shadow.
3. **Decision-trie over an arena.** `Term::Branch{cond, test, on_true, on_false}` by
   `BlockId`; `test: BranchTest{EffectiveZero, ExactZero}` picks `math.abs(c)<ε` vs
   `c==0` in the backend.
4. **`Instr` is a per-variant enum** (arity in the type — no operand-count mistakes).
   `recip(x)` lowers to `Div(ONE, x)`; no `Recip` node. `Powi`'s exponent is a `Const`
   operand (the pool entry becomes the `OpConstant`).
5. **Honest `try_recip` + `catch_unwind` fatality.** `try_recip` is NOT special-cased
   to "always Some" — you cannot know inside it whether the consumer will `.unwrap()`
   (assertion) or `.is_none()`/`?` (control flow). So it stays honest: for a runtime
   value it forks via `is_effective_zero` (invertible arm → `Div(ONE,x)`; zero arm →
   `None`). A consumer's `.unwrap()` on `None` then **panics**, which the driver catches
   per-path with `catch_unwind` and turns into a `Term::Fatal` leaf — `catch_unwind`
   contains the panic to one path, the worklist keeps going. Panics are **classified**
   from the payload (`viete::fatal::div_by_zero` → `DivByZero`; `` `None` value `` →
   `NonInvertible`; anything else → `Unexpected(msg)`, which surfaces tracer **bugs**).
   A `HookGuard` (RAII) silences the panic hook during tracing and restores it on every
   exit path (incl. an internal panic) without swallowing it.
   **The `/` operator itself also forks** (exact-zero guard): a runtime divisor is a
   real runtime fault point, so `a / b` (user-level) forks `ExactZero` on `b` — the
   `b == 0` arm panics → `Fatal{DivByZero}`, the other emits `Div`. The guard is
   `== 0` (NOT the 1e-9 `is_effective_zero` threshold, which would falsely trap valid
   small divisors and diverge from f64). The guard belongs to source `/`: `recip`'s
   internal `Div(ONE,x)` is emitted unguarded (`try_recip` already proved invertibility).
6. **Value-dedup constant pool is a load-bearing trie-merge invariant**, not an
   optimization. The pool is shared append-only across re-runs; a prefix `Instr` carries
   `Operand::Const(idx)` and trie-merge asserts byte-equal prefixes between runs. Without
   dedup, append-only would re-add the same constant under a new index and the prefix
   would diverge → assert fails. (`num-rational`'s `Hash` normalizes, so equal values
   share an index.) Per-path slot ids reset to keep prefixes deterministic; operands are
   resolved **before** slot allocation so interning can't shift slot numbering.

## Public API

```rust
let tracer = Tracer::builder().lua_threshold(1e-9).max_leaves(Some(1024)).build();
// trace<INPUTS, PARAMS, OUTPUTS>(constraints, closure). The closure takes TWO
// arrays: state inputs and params. `constraints` (possibly empty) declares
// caller-guaranteed domain predicates on inputs/params via InputRef.
let traced: Trace<6,0,16> = tracer.trace(&[], |inp, _param| motor_exp_coeffs::<Sym>(inp));
let lua: String     = traced.emit_lua();             // raw Lua source (no feature needed)
let out: [f64; 16]  = traced.run_lua(&inputs, &[]);  // params slice, runtime len-checked == PARAMS
let dump: String    = traced.pretty();               // raw trie dump
```

The user writes ONE generic `fn …<S: Scalar>(&[S;N])->[S;K]`; the `::<Sym>` instance is
traced, the `::<f64>` instance is the oracle — so the closure is monomorphic (Rust
closures can't be type-generic) and the genericity lives in the user's fn. `mlua` is an
**optional** dependency behind a default-on `lua` feature, so consumers can drop the
vendored-C build with `--no-default-features` (it only gates `run_lua`).

### Param input category (boundary-only sugar)

A **param** is just an `Input` with its own index. `Sym::param(i)` is a pure
constructor (`Handle::Param`, like `from_u32`) so a value can reference param `i`
*before* tracing starts — e.g. a joint built once outside the closure with
`Sym::param(i)` in place of baked rationals. Inside `run_trace` the engine
resolves `Param(j) -> Input(INPUTS + j)`, so the IR, fact lattice, CSE/DCE all
see plain inputs — no `Operand::Param`, no second fact map. The split surfaces
only at the boundary: the `PARAMS` const generic, the `run_lua(&inputs, params)`
slice, the `InputRef { Input, Param }` constraint targets, and the Lua emitter
choosing `inp[i]` vs `param[i-INPUTS]` by index (entry `fn name(inp, param)`).
At runtime the param values are typically extracted from a real `f64` object
(e.g. `newton`'s `JointEdge::params() -> Vec<T>`) and fed straight into
`run_lua`. `M = 0` degenerates cleanly (empty slice, Lua byte-identical).

## Status & tests

16 unit + 4 acceptance tests; whole workspace green; clippy clean. Headline: traced
`Motor::exp` → emitted Lua → executed matches `Motor::<f64>::exp` within 1e-9 at a
non-trivial (closed-form arms) and a near-seam (Taylor arms) input. `Motor::exp` yields
exactly **7 leaves** (4 `Return` + 3 `Fatal{DivByZero}`): memoization collapses
`cos_sq`+`sinc_sq` on the same `u` to 2 representation-select branches, and the
exact-zero Div guard adds a fault arm to each of the two closed-arm divisions; the 3
fault leaves are unreachable on valid input (`l = √u ≠ 0` there), so the acceptance
comparison stays green. Global test invariant: **zero `Unexpected` leaves** (a
tracer-bug guard).

## Open tails

- **`Constant(0)` divisor mismatch with f64.** `Sym::Div` on a static `Constant(0)`
  divisor panics (`DivByZero`), but f64 `x/0.0 = inf`. Unreachable in `exp`/`log` (their
  divisors are runtime), so untested against f64. Deliberately untouched by the
  exact-zero Div-guard work (which handles *runtime* divisors); revisit if it ever
  becomes reachable.
- **SPIR-V backend** emitted from the same trie, verified against the already-proven Lua
  emitter (two backends from one IR must agree), then GPU integration.
- **Out of scope (deferred):** CSE / hash-consing, DCE, algebraic simplification,
  reachability pruning; tracing matrix inverse (`bareiss_inverse`, whose
  `try_recip().is_none()` pivot search is an invertibility-fork) and memory ops.
