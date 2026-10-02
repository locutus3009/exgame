// SPDX-License-Identifier: MIT

mod ad;
mod motor;
mod objects;
mod screw;

pub use ad::{Differential, Dynamics};
pub use motor::Motor;
pub use objects::{Direction, Line, Plane, Point};
pub use screw::{Co, Contra, Screw, Twist, Variance, Wrench, pairing};

use crate::algebra::Pga3;
use crate::algebra::mv::Mv;
use peano::prelude::*;

/// Six degrees of freedom of se(3), in canonical Twist order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dof {
    Tx,
    Ty,
    Tz,
    Rx,
    Ry,
    Rz,
}

impl Dof {
    pub const ALL: [Dof; 6] = [Dof::Tx, Dof::Ty, Dof::Tz, Dof::Rx, Dof::Ry, Dof::Rz];
}

/// Objects a Motor can conjugate (sandwich M·X·M̃). By-value `to_mv` so it works
/// for both narrow-carrier types (Twist via widen) and `Mv`-wrapping objects.
pub trait Conjugatable<S: Ring> {
    fn to_mv(&self) -> Mv<Pga3, S>;
    fn from_mv(mv: Mv<Pga3, S>) -> Self;
}

// Raw multivectors are conjugatable.
impl<S: Ring> Conjugatable<S> for Mv<Pga3, S> {
    fn to_mv(&self) -> Mv<Pga3, S> {
        *self
    }
    fn from_mv(mv: Mv<Pga3, S>) -> Self {
        mv
    }
}
