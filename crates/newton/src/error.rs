// SPDX-License-Identifier: MIT

//! Crate-wide error, shared by the whole step pipeline: the accelerator PRODUCES
//! it (a connection kernel failed) and the integrator + `Mechanism::step`
//! PROPAGATE it up. Backend-agnostic (the CPU path never produces one; the Lua
//! path does).

/// Failure of one joint's connection-kernel evaluation, propagated up to
/// `Mechanism::step`.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// No compiled kernel registered for this joint's shader name.
    NoShader(&'static str),
    /// The kernel trapped (e.g. a `NonInvertible` `error()` in Lua).
    Trap { shader: &'static str, msg: String },
    /// One or more guarded-division fatal slots fired (0-based site indices).
    Fatal {
        shader: &'static str,
        sites: Vec<usize>,
    },
    /// The accelerator worker thread is gone (channel send/recv failed) — the
    /// dispatch cannot complete.
    WorkerGone,
    /// An unexpected backend (e.g. mlua) operation failed while running a kernel.
    Backend { shader: &'static str, msg: String },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::NoShader(s) => write!(f, "no kernel for shader {s}"),
            EvalError::Trap { shader, msg } => write!(f, "kernel {shader} trapped: {msg}"),
            EvalError::Fatal { shader, sites } => {
                write!(f, "kernel {shader} fatal at site(s) {sites:?}")
            }
            EvalError::WorkerGone => write!(f, "accelerator worker is gone"),
            EvalError::Backend { shader, msg } => {
                write!(f, "kernel {shader} backend error: {msg}")
            }
        }
    }
}
impl std::error::Error for EvalError {}
