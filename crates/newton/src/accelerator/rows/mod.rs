// SPDX-License-Identifier: MIT

//! Row bakers — one module per stage. Each owns its stage's word layout and
//! documents it in the module header; nothing else in the tree may assume it.

pub(crate) mod assemble;
pub(crate) mod body_post;
pub(crate) mod copy;
pub(crate) mod fixed;
pub(crate) mod gather;
pub(crate) mod gemm;
pub(crate) mod matvec;
pub(crate) mod reduce;
