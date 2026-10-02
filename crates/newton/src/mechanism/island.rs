// SPDX-License-Identifier: MIT

use super::component::Component;
use crate::integrator::implicit::cache::NewtonCache;
use aristotle::{WorldId, WorldKey};
use bytemuck::Pod;
use clifford::pga3::{Twist, Wrench};
use indexmap::IndexMap;
use joints::JointEdge;
use peano::prelude::*;

/// The island's bodies, by their `WorldId`.
pub(crate) type Bodies<T, S> = IndexMap<WorldId, Box<dyn Component<T, S>>>;
/// Joints BETWEEN the island's bodies, by the joint's `WorldId`.
pub(crate) type Joints<T> = IndexMap<WorldId, JointEdge<T>>;
/// The island's wrench buffer: clones of the body-owned `accum_wrench` keys.
pub(crate) type Wrenches<T> = IndexMap<WorldId, WorldKey<Wrench<T>>>;

/// One connectivity island: owns its bodies, the joints BETWEEN them and a
/// reusable wrench buffer. INVARIANT (upheld by every mutator of
/// `Islands`): the joint graph over `bodies` is connected — exactly one component.
///
/// The fields are private: the island topology is changed only by its mutator methods, and the step
/// takes the three step-time maps via `with()`. `origin` moves EXCLUSIVELY
/// through `recompute_centroid` — so the invariant "origin ≡ centre of mass" cannot
/// be bypassed. The island itself is a passive container, it knows no physics/topology.
pub(crate) struct Island<T: Scalar + Pod, S: Scalar + From<T> + Into<T>> {
    bodies: Bodies<T, S>,
    joints: Joints<T>,
    /// The island's wrench buffer: clones of the body-owned `accum_wrench` keys. The map
    /// is PERSISTENT as long as the island topology is stable — keys are added on
    /// body insertion and dropped on removal (insert_body/remove_body/merge_from),
    /// and the step only zeroes the VALUES (no clear/reallocation). The slots in the World
    /// storage belong to the bodies; the map holds only key clones.
    wrenches: Wrenches<T>,
    origin: Vector3<S>,
    /// Solver state that survives between steps (the warm approximate inverse
    /// and the matrix layout). Lives here because islands hand out disjoint
    /// `&mut` while the integrator itself is shared across islands stepped
    /// concurrently. Repartition rebuilds the island, so the cache is dropped
    /// exactly when it stops being meaningful.
    solver: NewtonCache<T>,
}

impl<T: Scalar + Pod, S: Scalar + From<T> + Into<T>> Island<T, S> {
    pub(crate) fn empty() -> Self {
        Self::empty_at(Vector3::ZERO)
    }

    /// An empty island with a given S anchor. Needed on a split: fragments inherit
    /// the parent's origin, so that the local body poses (relative to the parent's origin)
    /// are interpreted correctly BEFORE re-centring.
    pub(crate) fn empty_at(origin: Vector3<S>) -> Self {
        Self {
            bodies: IndexMap::new(),
            joints: IndexMap::new(),
            wrenches: IndexMap::new(),
            origin,
            solver: NewtonCache::new(),
        }
    }

    pub(crate) fn with(&self) -> (&Bodies<T, S>, &Joints<T>, &Wrenches<T>) {
        (&self.bodies, &self.joints, &self.wrenches)
    }

    pub(crate) fn with_mut(
        &mut self,
    ) -> (
        &mut Bodies<T, S>,
        &mut Joints<T>,
        &mut Wrenches<T>,
        &mut NewtonCache<T>,
    ) {
        (
            &mut self.bodies,
            &mut self.joints,
            &mut self.wrenches,
            &mut self.solver,
        )
    }

    /// See `Mechanism::inspect_solver_residual`.
    #[cfg(test)]
    pub(crate) fn solver_inverse_residual(&self) -> T {
        self.solver.inverse_residual()
    }

    pub(crate) fn origin(&self) -> Vector3<S> {
        self.origin
    }

    // ── reads ──
    pub(crate) fn contains_body(&self, id: WorldId) -> bool {
        self.bodies.contains_key(&id)
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
    pub(crate) fn get_body(&self, id: WorldId) -> Option<&dyn Component<T, S>> {
        self.bodies.get(&id).map(|v| &**v)
    }
    pub(crate) fn get_body_mut(&mut self, id: WorldId) -> Option<&mut Box<dyn Component<T, S>>> {
        self.bodies.get_mut(&id)
    }
    pub(crate) fn body_ids(&self) -> impl Iterator<Item = &WorldId> {
        self.bodies.keys()
    }
    pub(crate) fn bodies_iter(&self) -> impl Iterator<Item = &Box<dyn Component<T, S>>> {
        self.bodies.values()
    }
    pub(crate) fn bodies(&self) -> &IndexMap<WorldId, Box<dyn Component<T, S>>> {
        &self.bodies
    }
    pub(crate) fn joints(&self) -> &IndexMap<WorldId, JointEdge<T>> {
        &self.joints
    }

    // ── point mutators ──
    pub(crate) fn insert_body(&mut self, id: WorldId, body: Box<dyn Component<T, S>>) {
        // Register a clone of the body-owned wrench-accumulation key in the step-time map —
        // it is persistent while the body lives in the island (zeroed, not rebuilt).
        self.wrenches.insert(id, body.body().accum_wrench.clone());
        self.bodies.insert(id, body);
    }
    pub(crate) fn remove_body(&mut self, id: WorldId) -> Option<Box<dyn Component<T, S>>> {
        self.wrenches.swap_remove(&id);
        self.bodies.swap_remove(&id)
    }
    pub(crate) fn insert_joint(&mut self, id: WorldId, edge: JointEdge<T>) {
        self.joints.insert(id, edge);
    }
    /// Drop every joint incident to `id`.
    pub(crate) fn retain_joints_incident(&mut self, id: WorldId) {
        self.joints.retain(|_, e| e.a() != id && e.b() != id);
    }
    pub(crate) fn drain_joints(&mut self) -> indexmap::map::Drain<'_, WorldId, JointEdge<T>> {
        self.joints.drain(..)
    }
    pub(crate) fn set_joints(&mut self, joints: IndexMap<WorldId, JointEdge<T>>) {
        self.joints = joints;
    }

    // ── bulk (drain into another / take apart) ──
    /// Consume into (bodies, joints) for repartition.
    pub(crate) fn into_parts(self) -> (Bodies<T, S>, Joints<T>) {
        (self.bodies, self.joints)
    }
}

impl<T, S> Island<T, S>
where
    T: Scalar + StandardPart + Pod,
    S: Scalar + From<T> + Into<T>,
{
    /// Re-bind `origin` to the mass-weighted centre of mass and shift EVERY
    /// body motor so that the absolute world positions are preserved. The residual
    /// offset δ is small (origin is already ≈ COM), so it is computed in T; S is touched
    /// only by `origin += S::from(δ)`. The body-frame momentum is left alone. Idempotent.
    /// Fresh singleton island: origin = body position (lifted into S), the body motor
    /// is localized (sits at its own origin, translation ≈ identity). This is exactly
    /// "pull the linear position into T → lift into S → explicit motor shift".
    pub(crate) fn from_body(entity: Box<dyn Component<T, S>>) -> Self {
        let id = entity.id();
        let mut isl = Self::empty();
        isl.insert_body(id, entity);
        isl.recompute_centroid();
        // Singleton with no joints → empty cache; this also resets any stale
        // incidence of a body that arrived here via detach → add_body.
        isl.recompute_incidence();
        isl
    }

    /// Re-express all bodies relative to `new_origin` (shifting the motors by
    /// the anchor difference), preserving absolute positions. The difference is computed in S
    /// (exactly) and lowered into T — the "S diff → lower" seam primitive. Needed when
    /// merging islands with DIFFERENT origins: incoming bodies are interpreted relative to
    /// their old origin, and before insertion they must be brought to the common one.
    pub(crate) fn reanchor_to(&mut self, new_origin: Vector3<S>) {
        // d = old_origin − new_origin (in S), lowered into T: local' = local + d.
        let d_global = self.origin - new_origin;
        let d_tmp: [S; 3] = d_global.into();
        let d: Vector3<T> = Vector3::from(d_tmp.map(|v| v.into()));
        let shift = Twist::<T>::new(&d, &Vector3::ZERO).exp(T::ONE);
        for e in self.bodies.values_mut() {
            let b = e.body_mut();
            b.pose.write(b.pose.read().compose(&shift));
        }
        self.origin = new_origin;
    }

    /// Merge the bodies and joints of `other` into `self` (for `union`). `other` is first
    /// brought to the origin of `self`, so that local poses remain consistent.
    pub(crate) fn merge_from(&mut self, mut other: Island<T, S>) {
        other.reanchor_to(self.origin);
        for (k, v) in other.bodies {
            // Move the clone of its accumulation key together with the body (the map is persistent).
            self.wrenches.insert(k, v.body().accum_wrench.clone());
            self.bodies.insert(k, v);
        }
        for (k, v) in other.joints {
            self.joints.insert(k, v);
        }
    }

    pub(crate) fn recompute_centroid(&mut self) {
        if self.bodies.is_empty() {
            self.origin = Vector3::ZERO;
            return;
        }
        // δ = Σ mᵢ rᵢ / M, in T-local coordinates (rᵢ is the position relative to
        // the current origin).
        let mut total_m = T::ZERO;
        let mut wx = Vector3::ZERO;
        for e in self.bodies.values() {
            let m = e.body().inertia.mass();
            let r = e.body().position();
            total_m += m;
            for k in 0..3 {
                wx[k] += r[k] * m;
            }
        }
        // An island with no dynamic mass (only kinematic reference bodies,
        // mass()=0) has no CoM — do not move origin (otherwise wx·(1/0)=NaN).
        if total_m.standard_part().is_effective_zero() {
            return;
        }
        let delta = wx.scale(T::ONE / total_m);

        // Spatial shift by −δ via LEFT composition: only the translational part
        // of the motor moves, the rotation stays intact (translations commute).
        let shift = Twist::new(&(-delta), &Vector3::ZERO).exp(T::ONE);
        for e in self.bodies.values_mut() {
            let b = e.body_mut();
            b.pose.write(b.pose.read().compose(&shift));
        }
        // Advance the S anchor by the (small) δ.
        let delta: [T; 3] = delta.into();
        self.origin += Vector3::from(delta.map(|v| v.into()));
    }

    /// Rebuild the per-body incidence cache (`RigidBody::incident_terms`) from
    /// the island's current joints. Called on topology change — alongside
    /// `recompute_centroid`. The logic is shared with direct callers of `dispatch` —
    /// see `crate::accelerator::bake_incidence`.
    pub(crate) fn recompute_incidence(&mut self) {
        crate::accelerator::bake_incidence(&mut self.bodies, &self.joints, &self.wrenches);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Inert, Inertia, RigidBody};
    use aristotle::World;
    use clifford::pga3::Twist;
    use std::sync::Arc;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }
    fn body_at(world: Arc<World>, x: f32, y: f32, z: f32, mass: f32) -> Box<dyn Component<f32>> {
        let b = RigidBody::new(world.clone(), Inertia::isotropic(world, mass, 1.0));
        b.pose
            .write(Twist::new(&Vector3::from([x, y, z]), &Vector3::ZERO).exp(1.0));
        Inert::new(b)
    }
    fn abs_pos(isl: &Island<f32, f32>, id: WorldId) -> [f32; 3] {
        let r = isl.get_body(id).unwrap().body().position();
        let o = isl.origin();
        [r[0] + o[0], r[1] + o[1], r[2] + o[2]]
    }

    #[test]
    fn recompute_centroid_equal_masses_midpoint_and_preserves_abs() {
        let world = Arc::new(World::builder().usual::<f32>());

        let b1 = body_at(world.clone(), 0.0, 0.0, 0.0, 1.0);
        let b2 = body_at(world.clone(), 4.0, 0.0, 0.0, 1.0);
        let (id1, id2) = (b1.id(), b2.id());
        let mut isl: Island<f32, f32> = Island::empty();
        isl.insert_body(id1, b1);
        isl.insert_body(id2, b2);
        isl.recompute_centroid();
        // origin at midpoint
        let o = isl.origin();
        assert!(approx(o[0], 2.0, 1e-6), "origin.x = {}", o[0]);
        // absolute positions preserved
        let p1 = abs_pos(&isl, id1);
        let p2 = abs_pos(&isl, id2);
        assert!(approx(p1[0], 0.0, 1e-6) && approx(p2[0], 4.0, 1e-6));
        // Σ m r = 0 in local frame
        let r1 = isl.get_body(id1).unwrap().body().position();
        let r2 = isl.get_body(id2).unwrap().body().position();
        assert!(approx(r1[0] + r2[0], 0.0, 1e-6));
    }

    #[test]
    fn recompute_centroid_mass_weighted() {
        let world = Arc::new(World::builder().usual::<f32>());

        // m=10@0, m=1@11 → COM at 1.0
        let b1 = body_at(world.clone(), 0.0, 0.0, 0.0, 10.0);
        let b2 = body_at(world.clone(), 11.0, 0.0, 0.0, 1.0);
        let mut isl: Island<f32, f32> = Island::empty();
        isl.insert_body(b1.id(), b1);
        isl.insert_body(b2.id(), b2);
        isl.recompute_centroid();
        assert!(
            approx(isl.origin()[0], 1.0, 1e-6),
            "origin.x = {}",
            isl.origin()[0]
        );
    }

    #[test]
    fn recompute_centroid_3d_weighted() {
        let world = Arc::new(World::builder().usual::<f32>());

        // m=2@(1,0,0), m=1@(0,3,0), m=1@(0,0,4); Σm=4 → COM (0.5,0.75,1.0)
        let b1 = body_at(world.clone(), 1.0, 0.0, 0.0, 2.0);
        let b2 = body_at(world.clone(), 0.0, 3.0, 0.0, 1.0);
        let b3 = body_at(world.clone(), 0.0, 0.0, 4.0, 1.0);
        let mut isl: Island<f32, f32> = Island::empty();
        isl.insert_body(b1.id(), b1);
        isl.insert_body(b2.id(), b2);
        isl.insert_body(b3.id(), b3);
        isl.recompute_centroid();
        let o = isl.origin();
        assert!(approx(o[0], 0.5, 1e-6), "x={}", o[0]);
        assert!(approx(o[1], 0.75, 1e-6), "y={}", o[1]);
        assert!(approx(o[2], 1.0, 1e-6), "z={}", o[2]);
    }

    #[test]
    fn recompute_centroid_is_idempotent() {
        let world = Arc::new(World::builder().usual::<f32>());

        let b1 = body_at(world.clone(), 1.0, 2.0, 0.0, 3.0);
        let b2 = body_at(world.clone(), 5.0, -1.0, 2.0, 2.0);
        let mut isl: Island<f32, f32> = Island::empty();
        isl.insert_body(b1.id(), b1);
        isl.insert_body(b2.id(), b2);
        isl.recompute_centroid();
        let o1 = isl.origin();
        isl.recompute_centroid();
        let o2 = isl.origin();
        for k in 0..3 {
            assert!(
                approx(o1[k], o2[k], 1e-6),
                "axis {k}: {} vs {}",
                o1[k],
                o2[k]
            );
        }
    }

    #[test]
    fn recompute_centroid_does_not_touch_momentum() {
        let world = Arc::new(World::builder().usual::<f32>());

        use clifford::pga3::Wrench;
        let b = RigidBody::new(world.clone(), Inertia::isotropic(world, 2.0, 1.0));
        b.pose
            .write(Twist::new(&Vector3::from([7.0, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));
        b.momentum
            .write(Wrench::new(&Vector3::from([1.0, 2.0, 3.0]), &Vector3::ZERO));
        let body = Inert::new(b);
        let id = <Inert<f32> as Component<f32>>::id(&body);
        let mut isl: Island<f32, f32> = Island::empty();
        isl.insert_body(id, body);
        isl.recompute_centroid();
        let bb = isl.get_body(id).unwrap().body();
        let m = bb.momentum.read();
        assert_eq!(m.force().split(), [1.0, 2.0, 3.0]);
        // single body → origin sits exactly on it, local position ≈ 0
        let r = isl.get_body(id).unwrap().body().position();
        assert!(approx(r[0], 0.0, 1e-6), "local x = {}", r[0]);
        assert!(
            approx(isl.origin()[0], 7.0, 1e-6),
            "origin x = {}",
            isl.origin()[0]
        );
    }

    /// Rotation is preserved by the spatial-translation shift (only translation
    /// part of the motor moves).
    #[test]
    fn recompute_centroid_preserves_rotation() {
        let world = Arc::new(World::builder().usual::<f32>());

        use std::f32::consts::FRAC_PI_2;
        let b = RigidBody::new(world.clone(), Inertia::isotropic(world, 1.0, 1.0));
        // translate to (5,0,0) and rotate 90° about +Z
        b.pose.write(
            Twist::new(
                &Vector3::from([5.0, 0.0, 0.0]),
                &Vector3::from([0.0, 0.0, FRAC_PI_2]),
            )
            .exp(1.0),
        );
        let body = Inert::new(b);
        let id = <Inert<f32> as Component<f32>>::id(&body);
        let rot_before = isl_rot_maps_x(<Inert<f32> as Component<f32>>::body(&body));
        let mut isl: Island<f32, f32> = Island::empty();
        isl.insert_body(id, body);
        isl.recompute_centroid();
        let rot_after = isl_rot_maps_x(isl.get_body(id).unwrap().body());
        for k in 0..3 {
            assert!(
                approx(rot_before[k], rot_after[k], 1e-6),
                "rotation changed on axis {k}: {} vs {}",
                rot_before[k],
                rot_after[k]
            );
        }
    }

    // Image of world +X direction under the body's rotation (translation-free
    // probe of the orientation part).
    fn isl_rot_maps_x(b: &RigidBody<f32>) -> [f32; 3] {
        use clifford::pga3::Point;
        let p1 = b
            .pose
            .read()
            .conjugate(&Point::new(Vector3::from([1.0, 0.0, 0.0])))
            .coords();
        let p0 = b.pose.read().conjugate(&Point::new(Vector3::ZERO)).coords();
        [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
    }
}
