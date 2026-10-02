// SPDX-License-Identifier: MIT

use crate::ir::{Block, BlockId, Instr, Operand, Term, Tree};
use std::collections::{HashMap, HashSet};

/// Emit the whole trie as a Lua function `<fn_name>(inp)` returning a table of K values.
/// `EffectiveZero` branches emit `if math.abs(cond) < threshold then ... else ... end`;
/// `ExactZero` branches (Div guards) emit `if cond == 0.0 then ... else ... end`.
/// The kernel's Pod boundary: how the traced ids are spelled on each side of
/// the generated `main`. Grouped into one argument because they are always
/// supplied and threaded together.
pub struct Boundary<'a> {
    /// Traced input id -> the GLSL rvalue that supplies it.
    pub inputs: &'a HashMap<u32, String>,
    /// Traced output id -> its lvalue AND a sign (±1). The destination is a raw
    /// Pod slot of the host screw/motor, whose blade+markup sign need not match
    /// the kernel's force/torque order. The lvalue can't negate, so the sign is
    /// applied to the rhs here. Inputs fold their sign into the rvalue string.
    pub outputs: &'a HashMap<u32, (String, i8)>,
    /// Traced fatal id -> the lvalue its flag is written to.
    pub fatals: &'a HashMap<u32, String>,
}

pub fn emit_glsl(
    tree: &Tree,
    threshold: f64,
    fn_name: &str,
    pre: String,
    // Text appended after the traced body, still inside `main`. Closes whatever
    // `pre` opened — the term loop of a reduction kernel — and stores the
    // accumulator. Purely textual: the emitter does not parse it.
    footer: String,
    boundary: &Boundary<'_>,
) -> String {
    let (inputs_map, output_map, fatal_map) = (boundary.inputs, boundary.outputs, boundary.fatals);
    let mut out = String::new();

    out.push_str(&format!("// {fn_name}\n"));
    out.push_str("void main() {\n");
    out.push_str(&pre);
    out.push_str(&format!("    const float eps = {threshold:e};\n"));
    let used: HashSet<u32> = tree.collect_consts().into_iter().collect();
    for (i, r) in tree.consts.iter().enumerate() {
        if !used.contains(&(i as u32)) {
            continue;
        }
        if *r.numer() == 0 {
            out.push_str(&format!("    const float c{i} = 0.0;\n",));
        } else if *r.numer() == *r.denom() {
            out.push_str(&format!("    const float c{i} = 1.0;\n",));
        } else {
            out.push_str(&format!(
                "    const float c{i} = ({}.0/{}.0);\n",
                r.numer(),
                r.denom()
            ));
        }
    }
    emit_block(
        tree, tree.root, 1, &mut out, inputs_map, output_map, fatal_map,
    );
    out.push_str(&footer);
    out.push_str("}\n");
    out
}

/// Fatal slots the emitted kernel writes: the widest `Return.fatals` reachable
/// from the root. After `flatten` there is exactly one `Return`, and every
/// invocation writes all of its slots, so this is both the count the host must
/// scan and the minimum width of the kernel's fatal array.
///
/// `Tree::fatal_leaves` is NOT this number: it counts `Fatal` leaf blocks, which
/// the flatten pass collapses into exactly these status operands.
pub fn fatal_slots(tree: &Tree) -> usize {
    fn walk(tree: &Tree, id: BlockId) -> usize {
        match &tree.blocks[id.0 as usize].term {
            Term::Return { fatals, .. } => fatals.len(),
            Term::Branch {
                on_true, on_false, ..
            } => walk(tree, *on_true).max(walk(tree, *on_false)),
            Term::Fatal { .. } => 0,
        }
    }
    walk(tree, tree.root)
}

fn operand(op: &Operand, inputs_map: &HashMap<u32, String>) -> String {
    match op {
        Operand::Const(i) => format!("c{i}"),
        Operand::Input(i) => inputs_map[i].to_string(),
        Operand::Instr(s) => format!("v{s}"),
        Operand::Eps => "eps".to_string(),
    }
}

fn emit_instr(ins: &Instr, indent: &str, out: &mut String, inputs_map: &HashMap<u32, String>) {
    let (type_, slot, expr) = match ins {
        Instr::Add(s, a, b) => (
            "float",
            *s,
            format!("{} + {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::Sub(s, a, b) => (
            "float",
            *s,
            format!("{} - {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::Mul(s, a, b) => (
            "float",
            *s,
            format!("{} * {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::Div(s, a, b) => (
            "float",
            *s,
            format!("{} / {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::Neg(s, a) => ("float", *s, format!("-({})", operand(a, inputs_map))),
        Instr::Sqrt(s, a) => ("float", *s, format!("sqrt({})", operand(a, inputs_map))),
        Instr::Sin(s, a) => ("float", *s, format!("sin({})", operand(a, inputs_map))),
        Instr::Cos(s, a) => ("float", *s, format!("cos({})", operand(a, inputs_map))),
        Instr::Exp(s, a) => ("float", *s, format!("exp({})", operand(a, inputs_map))),
        Instr::Ln(s, a) => ("float", *s, format!("log({})", operand(a, inputs_map))),
        Instr::Abs(s, a) => ("float", *s, format!("abs({})", operand(a, inputs_map))),
        Instr::Atan2(s, y, x) => (
            "float",
            *s,
            format!(
                "atan({}, {})",
                operand(y, inputs_map),
                operand(x, inputs_map)
            ),
        ),
        Instr::Powi(s, a, n) => (
            "float",
            *s,
            format!(
                "pow({}, {})",
                operand(a, inputs_map),
                operand(n, inputs_map)
            ),
        ),
        Instr::Powf(s, a, b) => (
            "float",
            *s,
            format!(
                "pow({}, {})",
                operand(a, inputs_map),
                operand(b, inputs_map)
            ),
        ),
        Instr::Select(s, cmp, a, b, on_match, on_miss) => {
            // Scalar `a cmp b ? on_match : on_miss`. GLSL does allow `?:` on a
            // scalar bool, but parenthesise the condition so the emit never leans on
            // operator precedence and reads the same as the vec `mix` form above.
            let op = match cmp {
                crate::ir::CmpKind::Eq => "==",
                crate::ir::CmpKind::Lt => "<",
            };
            (
                "float",
                *s,
                format!(
                    "({} {op} {}) ? {} : {}",
                    operand(a, inputs_map),
                    operand(b, inputs_map),
                    operand(on_match, inputs_map),
                    operand(on_miss, inputs_map),
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
                format!("-({})", operand(a, inputs_map))
            } else {
                operand(a, inputs_map)
            };
            let c_s = if *neg_c {
                format!("-({})", operand(c, inputs_map))
            } else {
                operand(c, inputs_map)
            };
            (
                "float",
                *s,
                format!("fma({}, {}, {})", a_s, operand(b, inputs_map), c_s),
            )
        }

        // ---- vector instructions ----
        // vec4 is the FIXED lane type, so a group narrower than 4 (`w`/`ops` < 4)
        // still constructs a full vec4: the unused high lanes are never `Extract`-ed,
        // so padding them is harmless. Splat broadcasts the scalar to all 4; Pack
        // repeats its last operand into the tail (avoids a stray 0.0 in a divisor).
        Instr::Splat(s, _w, a) => {
            let x = operand(a, inputs_map);
            ("vec4", *s, format!("vec4({x}, {x}, {x}, {x})"))
        }
        Instr::Pack(s, ops) => {
            let mut elems: Vec<String> = ops.iter().map(|o| operand(o, inputs_map)).collect();
            let pad = elems.last().cloned().unwrap_or_else(|| "0.0".to_string());
            while elems.len() < 4 {
                elems.push(pad.clone());
            }
            ("vec4", *s, format!("vec4({})", elems.join(", ")))
        }
        Instr::VNeg(s, a) => ("vec4", *s, format!("-({})", operand(a, inputs_map))),
        Instr::VAdd(s, a, b) => (
            "vec4",
            *s,
            format!("{} + {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::VSub(s, a, b) => (
            "vec4",
            *s,
            format!("{} - {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::VMul(s, a, b) => (
            "vec4",
            *s,
            format!("{} * {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::VDiv(s, a, b) => (
            "vec4",
            *s,
            format!("{} / {}", operand(a, inputs_map), operand(b, inputs_map)),
        ),
        Instr::VFma(s, a, b, c) => (
            "vec4",
            *s,
            format!(
                "fma({}, {}, {})",
                operand(a, inputs_map),
                operand(b, inputs_map),
                operand(c, inputs_map)
            ),
        ),
        Instr::VSelect(s, cmp, x, y, a, b) => {
            // Per-lane `x cmp y ? a : b`. GLSL has no `<`/`==`/`?:` on vec4, so the
            // comparison goes through the componentwise intrinsic (`lessThan` /
            // `equal` → bvec4) and the select through `mix(x, y, bvec)`, which
            // picks its SECOND argument where the mask is true. Hence `mix(b, a, …)`
            // to match `vsel`'s "condition true ⇒ a" (see the Lua carrier's vsel).
            let helper = match cmp {
                crate::ir::CmpKind::Eq => "equal",
                crate::ir::CmpKind::Lt => "lessThan",
            };
            (
                "vec4",
                *s,
                format!(
                    "mix({}, {}, {helper}({}, {}))",
                    operand(b, inputs_map),
                    operand(a, inputs_map),
                    operand(x, inputs_map),
                    operand(y, inputs_map),
                ),
            )
        }
        Instr::Extract(s, v, lane) => (
            "float",
            *s,
            format!(
                "{}.{}",
                operand(v, inputs_map),
                match *lane {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    3 => "w",
                    _ => unreachable!(),
                }
            ),
        ),
    };
    out.push_str(&format!("{indent}{type_} v{slot} = {expr};\n"));
}

fn emit_block(
    tree: &Tree,
    id: BlockId,
    depth: usize,
    out: &mut String,
    inputs_map: &HashMap<u32, String>,
    output_map: &HashMap<u32, (String, i8)>,
    fatal_map: &HashMap<u32, String>,
) {
    let indent = "    ".repeat(depth);
    let block: &Block = &tree.blocks[id.0 as usize];
    for ins in &block.instrs {
        emit_instr(ins, &indent, out, inputs_map);
    }
    match &block.term {
        /*
        Term::Branch {
            cond,
            test,
            cond_sign,
            on_true,
            on_false,
        } => {
            let cmp = match test {
                BranchTest::EffectiveZero => match cond_sign {
                    CondSign::NonNeg => format!("{} < eps", operand(cond, inputs_map)),
                    CondSign::NonPos => format!("-{} < eps", operand(cond, inputs_map)),
                    CondSign::Unknown => {
                        format!("abs({}) < eps", operand(cond, inputs_map))
                    }
                },
                BranchTest::ExactZero => format!("{} == 0.0", operand(cond, inputs_map)),
            };
            out.push_str(&format!("{indent}if {cmp} then\n"));
            emit_block(
                tree,
                *on_true,
                depth + 1,
                out,
                inputs_map,
                output_map,
                fatal_map,
            );
            out.push_str(&format!("{indent}else\n"));
            emit_block(
                tree,
                *on_false,
                depth + 1,
                out,
                inputs_map,
                output_map,
                fatal_map,
            );
            out.push_str(&format!("{indent}end\n"));
        }*/
        Term::Return { outputs, fatals } => {
            let outs: Vec<String> = outputs.iter().map(|o| operand(o, inputs_map)).collect();
            let fats: Vec<String> = fatals.iter().map(|o| operand(o, inputs_map)).collect();
            for (i, f) in outs.iter().enumerate() {
                let (lhs, sign) = &output_map[&(i as u32)];
                let rhs = if *sign < 0 {
                    format!("-({f})")
                } else {
                    f.clone()
                };
                out.push_str(&format!("{indent}{lhs} = {rhs};\n"));
            }
            for (i, f) in fats.iter().enumerate() {
                // A fatal operand with no slot to land in would be a silently
                // dropped division guard. Refuse to emit rather than lose it.
                let Some(lhs) = fatal_map.get(&(i as u32)) else {
                    panic!(
                        "viete: kernel writes {} fatal slot(s) but the boundary maps \
                         only {}; size the fatal array from `fatal_slots`",
                        fats.len(),
                        fatal_map.len()
                    );
                };
                out.push_str(&format!("{indent}{lhs} = {f};\n"));
            }
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block as IrBlock, BlockId, Instr, Operand, Term, Tree};

    /// One `Return` carrying two fatal-status operands, as the flatten pass
    /// leaves a kernel with two guarded divisions.
    fn two_fatal_tree() -> Tree {
        Tree {
            inputs: 2,
            params: 0,
            consts: vec![],
            blocks: vec![IrBlock {
                instrs: vec![
                    Instr::Div(0, Operand::Input(0), Operand::Input(1)),
                    Instr::Abs(1, Operand::Input(1)),
                ],
                term: Term::Return {
                    outputs: vec![Operand::Instr(0)],
                    fatals: vec![Operand::Instr(1), Operand::Instr(1)],
                },
            }],
            root: BlockId(0),
        }
    }

    fn emit_with_fatal_slots(tree: &Tree, slots: u32) -> String {
        let inputs: HashMap<u32, String> = [(0, "a".to_string()), (1, "b".to_string())]
            .into_iter()
            .collect();
        let outputs: HashMap<u32, (String, i8)> = [(0, ("o".to_string(), 1))].into_iter().collect();
        let fatals: HashMap<u32, String> = (0..slots).map(|i| (i, format!("f[{i}]"))).collect();
        emit_glsl(
            tree,
            1e-9,
            "probe",
            String::new(),
            String::new(),
            &Boundary {
                inputs: &inputs,
                outputs: &outputs,
                fatals: &fatals,
            },
        )
    }

    /// The slot count is read off the `Return`, and a boundary of exactly that
    /// width receives every operand.
    #[test]
    fn fatal_slots_counts_the_return_operands() {
        let tree = two_fatal_tree();
        assert_eq!(fatal_slots(&tree), 2);
        let glsl = emit_with_fatal_slots(&tree, 2);
        assert!(glsl.contains("f[0] = v1;"), "{glsl}");
        assert!(glsl.contains("f[1] = v1;"), "{glsl}");
    }

    /// A boundary narrower than the trace must stop codegen — in `build.rs`,
    /// that is the build — instead of dropping a division guard.
    #[test]
    #[should_panic(expected = "kernel writes 2 fatal slot(s) but the boundary maps only 1")]
    fn too_few_fatal_slots_refuses_to_emit() {
        emit_with_fatal_slots(&two_fatal_tree(), 1);
    }

    /// A reduction kernel opens its term loop in `pre` and closes it in
    /// `footer`, so the footer must land after the traced body and still inside
    /// `main`.
    #[test]
    fn footer_is_emitted_after_the_body_and_before_the_closing_brace() {
        let tree = Tree {
            inputs: 2,
            params: 0,
            consts: vec![],
            blocks: vec![IrBlock {
                instrs: vec![Instr::Add(0, Operand::Input(0), Operand::Input(1))],
                term: Term::Return {
                    outputs: vec![Operand::Instr(0)],
                    fatals: vec![],
                },
            }],
            root: BlockId(0),
        };
        let inputs: HashMap<u32, String> = [(0, "a".to_string()), (1, "b".to_string())]
            .into_iter()
            .collect();
        let outputs: HashMap<u32, (String, i8)> =
            [(0, ("acc.c[0]".to_string(), 1))].into_iter().collect();

        let glsl = emit_glsl(
            &tree,
            1e-9,
            "probe",
            "    for (uint k = 0u; k < n; ++k) {\n".to_string(),
            "    }\n    out.data[0].c[0] = acc.c[0];\n".to_string(),
            &Boundary {
                inputs: &inputs,
                outputs: &outputs,
                fatals: &HashMap::new(),
            },
        );

        let body = glsl
            .find("acc.c[0] =")
            .expect("body assigns the accumulator");
        let footer = glsl.find("out.data[0].c[0]").expect("footer present");
        let close = glsl.rfind('}').expect("main closes");
        assert!(body < footer, "footer must follow the traced body:\n{glsl}");
        assert!(footer < close, "footer must be inside main:\n{glsl}");
    }
}
