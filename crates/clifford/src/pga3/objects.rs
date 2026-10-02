// SPDX-License-Identifier: MIT

//! PGA3 geometric objects on top of the strata-based Mv: Point/Plane/Line/
//! Direction. The markup is dual (⋆₃); signs are derived from the +I anchor.

use super::Conjugatable;
use crate::algebra::mv::Mv;
use crate::algebra::mv::{
    direction_coords, line_direction, line_moment, make_direction, make_plane, plane_normal,
    plane_offset,
};
use crate::algebra::{Algebra, Pga3};
use peano::prelude::*;

type M<S> = Mv<Pga3, S>;
type V3<S> = Vector<Succ<Succ<Succ<Z>>>, S>;

// ── Geometric objects ───────────────────────────────────────────
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point<S: Ring>(M<S>);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Plane<S: Ring>(M<S>);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Line<S: Ring>(M<S>);
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Direction<S: Ring>(M<S>);

impl<S: Ring> Point<S> {
    pub fn new(v: V3<S>) -> Self {
        Point(crate::algebra::mv::make_point(&v))
    }
    pub fn from_mv(m: M<S>) -> Self {
        Point(m)
    }
    pub fn weight(&self) -> S {
        self.0.get((Pga3::DIM - 1) & !Pga3::ZERO_MASK)
    }
    pub fn as_mv(&self) -> M<S> {
        self.0
    }
    /// Line through two points: P ∨ Q (regressive = meet).
    pub fn join(&self, other: &Self) -> Line<S> {
        Line(self.0.meet(&other.0))
    }
}

impl<S: Ring + Invertible> Point<S> {
    /// Coordinates with projective normalization (division by the weight).
    pub fn coords(&self) -> V3<S> {
        crate::algebra::mv::position(&self.0)
    }
}

impl<S: Ring> Plane<S> {
    /// Public d-form: a·x + b·y + c·z + d = 0. Incidence P∧π = 0
    /// in the derived markup gives n·r = −d (e0 coefficient = −d) — the negation
    /// is an API convention, pinned by the three-planes test.
    pub fn new(a: S, b: S, c: S, d: S) -> Self {
        let mut n = V3::<S>::ZERO;
        n[0] = a;
        n[1] = b;
        n[2] = c;
        Self::from_incidence(n, -d)
    }
    /// Incidence form: n·r = offset.
    pub fn from_incidence(normal: V3<S>, offset: S) -> Self {
        Plane(make_plane(&normal, offset))
    }
    pub fn normal(&self) -> V3<S> {
        plane_normal(&self.0)
    }
    /// d from the public d-form (= −incidence offset).
    pub fn offset(&self) -> S {
        -plane_offset(&self.0)
    }
    pub fn as_mv(&self) -> M<S> {
        self.0
    }
    /// Line of intersection of two planes: π ∧ ρ (outer = wedge).
    pub fn meet(&self, other: &Self) -> Line<S> {
        Line(self.0.wedge(&other.0))
    }
}

impl<S: Ring> Line<S> {
    pub fn direction(&self) -> V3<S> {
        line_direction(&self.0)
    }
    pub fn moment(&self) -> V3<S> {
        line_moment(&self.0)
    }
    pub fn as_mv(&self) -> M<S> {
        self.0
    }
    /// Plane through a line and a point (regressive = meet).
    pub fn join(&self, point: &Point<S>) -> Plane<S> {
        Plane(self.0.meet(&point.0))
    }
}

impl<S: Scalar> Line<S> {
    /// Euclidean norm of the direction part (= |p − q| for unit points).
    pub fn weight_norm(&self) -> S {
        self.0.norm()
    }
    /// Plücker lever arm |moment| / |direction|; 0 for lines through the origin.
    pub fn distance_to_origin(&self) -> S {
        let m = self.moment();
        (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt_explicit() / self.weight_norm()
    }
    /// Point of intersection of a line with a plane: ℓ ∧ π (outer = wedge).
    pub fn meet_plane(&self, plane: &Plane<S>) -> Point<S> {
        Point(self.0.wedge(&plane.0))
    }
}

impl<S: Scalar + StandardPart> Line<S> {
    /// Regularized length √(d² + ε²): the derivative is finite through
    /// coinciding anchors (see Mv::soft_norm).
    pub fn soft_weight_norm(&self, eps: S) -> S {
        self.0.soft_norm(eps)
    }
}

// Weighted (homogeneous) coordinates: sum and scale are meaningful —
// for example, a mass-weighted sum of points gives the centroid. Plus deepen and
// Conjugatable for the sandwich with a motor.
macro_rules! object_ops {
    ($Ty:ident) => {
        impl<S: Ring> core::ops::Add for $Ty<S> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                $Ty(self.0 + rhs.0)
            }
        }
        impl<S: Ring> core::ops::Mul<S> for $Ty<S> {
            type Output = Self;
            fn mul(self, s: S) -> Self {
                $Ty(self.0.scale(s))
            }
        }
        impl<S: Ring> $Ty<S> {
            pub fn deepen<T: Ring>(&self) -> $Ty<T>
            where
                S: crate::Lift<T>,
            {
                $Ty(self.0.deepen())
            }
        }
        impl<S: Ring> Conjugatable<S> for $Ty<S> {
            fn to_mv(&self) -> M<S> {
                self.0
            }
            fn from_mv(mv: M<S>) -> Self {
                $Ty(mv)
            }
        }
    };
}
object_ops!(Point);
object_ops!(Plane);
object_ops!(Line);
object_ops!(Direction);

impl<S: Ring> Direction<S> {
    pub fn new(v: V3<S>) -> Self {
        Direction(make_direction(&v))
    }
    pub fn from_mv(m: M<S>) -> Self {
        Direction(m)
    }
    pub fn coords(&self) -> V3<S> {
        direction_coords(&self.0)
    }
    pub fn as_mv(&self) -> M<S> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pga_objects_meet_and_join() {
        use crate::pga3::{Plane, Point};
        fn close3(a: &Vector<N3, f64>, b: [f64; 3]) -> bool {
            (*a.get::<Z>() - b[0]).abs() < 1e-9
                && (*a.get::<Succ<Z>>() - b[1]).abs() < 1e-9
                && (*a.get::<Succ<Succ<Z>>>() - b[2]).abs() < 1e-9
        }

        // Point ↔ coordinates (projective normalization) round-trip.
        let p = Point::new(vector![1.5, -2.0, 3.0]);
        assert!(close3(&p.coords(), [1.5, -2.0, 3.0]));

        // Plane: n·r = offset (fixed by the incidence P∧π=0). x=2,y=3,z=4
        // intersect at (2,3,4).
        let px = Plane::from_incidence(vector![1.0, 0.0, 0.0], 2.0);
        let py = Plane::from_incidence(vector![0.0, 1.0, 0.0], 3.0);
        let pz = Plane::from_incidence(vector![0.0, 0.0, 1.0], 4.0);
        let pt = px.meet(&py).meet_plane(&pz);
        assert!(close3(&pt.coords(), [2.0, 3.0, 4.0]), "{:?}", pt.coords());
    }

    // ── Restored invariants of the legacy objects (A7) ───────────────────
    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn point_roundtrip() {
        let p = Point::new(vector![1.5, -2.0, 3.0]);
        let c = p.coords();
        assert!(
            approx(c[0], 1.5) && approx(c[1], -2.0) && approx(c[2], 3.0),
            "{c:?}"
        );
        assert!(approx(p.weight(), 1.0));
    }

    #[test]
    fn point_coords_are_scale_invariant() {
        // Homogeneity: weight ≠ 1 does not change the position (projective normalization).
        let p = Point::new(vector![1.5, -2.0, 3.0]);
        let scaled = Point::from_mv(p.as_mv().scale(2.0));
        assert_eq!(scaled.coords(), p.coords());
    }

    #[test]
    fn plane_roundtrip() {
        // Public d-form: a·x+b·y+c·z+d = 0 — the readers return the input.
        let pl = Plane::new(1.0f64, 2.0, 3.0, -4.0);
        assert_eq!(pl.normal(), vector![1.0, 2.0, 3.0]);
        assert_eq!(pl.offset(), -4.0);
    }

    #[test]
    fn direction_has_zero_weight() {
        use crate::algebra::Pga3;
        use crate::algebra::markup::i3_blade;
        let d = Direction::new(vector![1.0, 0.0, 0.0]);
        assert_eq!(d.as_mv().get(i3_blade::<Pga3>()), 0.0); // weight (I₃) is zero
        assert_eq!(d.coords(), vector![1.0, 0.0, 0.0]);
    }

    #[test]
    fn point_joined_with_itself_is_degenerate() {
        let p = Point::<f64>::new(vector![1.0, 2.0, 3.0]);
        let l = p.join(&p);
        for s in 0..16 {
            assert!(l.as_mv().get(s).abs() < 1e-12, "slot {s}");
        }
    }

    #[test]
    fn line_direction_is_segment_vector() {
        let p = Point::<f64>::new(vector![1.0, 0.0, 0.0]);
        let q = Point::new(vector![0.0, 0.0, 0.0]);
        let l = p.join(&q);
        // direction ∥ ±(p − q) = ±[1,0,0]; weight_norm = |p − q| = 1
        let d = l.direction();
        assert!(
            approx(d[0].abs(), 1.0) && approx(d[1], 0.0) && approx(d[2], 0.0),
            "{d:?}"
        );
        assert!(approx(l.weight_norm(), 1.0));
    }

    #[test]
    fn line_distance_to_origin() {
        // line through (0,1,0) along x: lever arm = 1
        let p = Point::new(vector![1.0, 1.0, 0.0]);
        let q = Point::new(vector![-1.0, 1.0, 0.0]);
        let l = p.join(&q);
        assert!(
            approx(l.distance_to_origin(), 1.0),
            "{}",
            l.distance_to_origin()
        );
        // line through the origin: lever arm = 0
        let l0 = Point::new(vector![1.0, 0.0, 0.0]).join(&Point::new(vector![-1.0, 0.0, 0.0]));
        assert!(
            approx(l0.distance_to_origin(), 0.0),
            "{}",
            l0.distance_to_origin()
        );
    }
}
