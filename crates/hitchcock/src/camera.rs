// SPDX-License-Identifier: MIT

use crate::{
    MarkerKind,
    rig::{RigBuilder, RigDebug},
};
use aristotle::{Epoch, World, WorldId, WorldKey};
use async_trait::async_trait;
use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Motor, Twist, Wrench},
};
use newton::{
    Component, ForceField, Inert, Inertia, Mechanism, RigidBody,
    gravity::GravityPropagator,
    indexmap::IndexMap,
    integrator::{ImplicitIntegrator, Newton},
};
use peano::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use tokio::sync::RwLock;

/// The mechanism the camera is following, if it is still alive. `Weak` so the
/// camera never keeps a scene alive, `Option` because there may be no target,
/// `RwLock` because the target is swapped from outside the step.
type TargetSlot<T, S> = Arc<RwLock<Option<Weak<Mechanism<T, S>>>>>;

pub(crate) const fn m_camera<T: Scalar>() -> T {
    T::ONE
}

/// A kinematic marker body that sits at the CoM of the TRACKED mechanism. The body's
/// pose is LOCAL to the camera island's origin, while the target is given in ABSOLUTE coordinates
/// (in S). Therefore the difference "target − camera anchor" is taken IN S (`centroid_s`,
/// `anchor`), and only the small remainder is lowered into T — otherwise, for scenes far from the global
/// zero, two large nearly equal S numbers would give catastrophic
/// cancellation in T. This is exactly "absolute positions ALWAYS in S" at the boundary
/// of an inter-mechanism reference. Implemented as a full `Component` because
/// it needs the island's S anchor — which is not available to an `Entity` closure (pure T).
struct TargetTracker<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    id: WorldId,
    body: RigidBody<T>,
    target: TargetSlot<T, S>,
}

#[async_trait]
impl<T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>, S: Scalar + From<T> + Into<T>>
    Component<T, S> for TargetTracker<T, S>
{
    async fn pre_step(&mut self, _epoch: &Epoch<T>, anchor: &Vector3<S>) {
        let mut guard = self.target.write().await;
        if let Some(target) = guard.as_ref().and_then(|w| w.upgrade()) {
            // The target's absolute CoM in S, minus the camera island's S anchor → a small local
            // remainder, lowered into T. The difference is taken in S (no cancellation).
            let p_target_s = target.centroid().await.unwrap().coords();
            let v_target = target.centroid_velocity().await.unwrap();

            let tmp = p_target_s - *anchor;
            let tmp: [S; 3] = tmp.into();
            let local = Vector3::<T>::from(tmp.map(|v| v.into()));
            self.body
                .pose
                .write(Twist::new(&local, &Vector3::ZERO).exp(T::ONE));
            // Velocity is translation-invariant (the island frame is inertial between
            // re-anchorings), it does not need the anchor.
            let v_twist = Twist::new(&v_target.linear(), &Vector3::ZERO);
            self.body.momentum.write(self.body.inertia.apply(&v_twist));
        } else {
            *guard = None;
        };
    }
    fn body(&self) -> &RigidBody<T> {
        &self.body
    }
    fn body_mut(&mut self) -> &mut RigidBody<T> {
        &mut self.body
    }
    fn id(&self) -> WorldId {
        self.id
    }
}

//#[derive(Debug)]
pub struct Camera<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    id: WorldId,
    /// The eye body.
    real: WorldId,
    anchor: WorldId,
    target: TargetSlot<T, S>,
    // TODO: change to the zoom explicitly?
    r_object: T,
    mechanism: Arc<Mechanism<T, S>>,
    rig: RigDebug<T>,
}

impl<T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>, S: Scalar + From<T> + Into<T>>
    Camera<T, S>
{
    pub async fn retarget(&mut self, target: Option<Arc<Mechanism<T, S>>>) {
        let mut guard = self.target.write().await;
        *guard = target.map(|arc| Arc::downgrade(&arc));
    }

    pub fn set_distance(&mut self, distance: T) {
        self.r_object = distance;
        // TODO: update _all_ joints in the camera mechanism
    }

    pub async fn position(&self) -> Vector3<T> {
        self.mechanism
            .body_absolute_position(self.real)
            .await
            .unwrap()
    }

    pub fn id(&self) -> WorldId {
        self.id
    }

    /// World position of a body-local point (anchor): `island origin ⊕ pose·local`
    /// (poses are local to the origin after COM binding).
    async fn world_of(&self, id: WorldId, local: Vector3<T>) -> Vector3<T> {
        self.mechanism.body_world_point(id, local).await.unwrap()
    }

    /// [`MarkerKind`] of a body id (the named rig bodies; anything else is an
    /// attachment).
    fn kind_of(&self, id: WorldId) -> MarkerKind {
        if id == self.real {
            MarkerKind::Eye
        } else if id == self.rig.intermediate {
            MarkerKind::Intermediate
        } else if id == self.anchor {
            MarkerKind::Anchor
        } else if id == self.rig.target_tracker {
            MarkerKind::TargetTracker
        } else {
            MarkerKind::Attachment
        }
    }

    /// True if a body-local offset is the centre of mass (no frame geometry).
    fn is_com(local: Vector3<T>) -> bool {
        local[0].standard_part().is_effective_zero()
            && local[1].standard_part().is_effective_zero()
            && local[2].standard_part().is_effective_zero()
    }

    /// World-space debug markers: the rig's bodies and every recorded attachment
    /// point, each tagged with its [`MarkerKind`]. For visualization only.
    pub async fn markers(&self) -> Vec<(Vector3<T>, MarkerKind)> {
        let z = Vector3::ZERO;
        let mut out = vec![
            (self.world_of(self.real, z).await, MarkerKind::Eye),
            (
                self.world_of(self.rig.intermediate, z).await,
                MarkerKind::Intermediate,
            ),
            (self.world_of(self.anchor, z).await, MarkerKind::Anchor),
            (
                self.world_of(self.rig.target_tracker, z).await,
                MarkerKind::TargetTracker,
            ),
        ];
        for &(id, local) in &self.rig.attachments {
            out.push((self.world_of(id, local).await, MarkerKind::Attachment));
        }
        out
    }

    /// World-space line segments for each longitudinal spring (anchor → anchor).
    pub async fn spring_segments(&self) -> Vec<(Vector3<T>, Vector3<T>)> {
        let a: Vec<_> = self
            .rig
            .springs
            .iter()
            .map(|((a_id, la), (b_id, lb))| async move {
                (
                    self.world_of(*a_id, *la).await,
                    self.world_of(*b_id, *lb).await,
                )
            })
            .collect();
        let mut res = Vec::new();
        for p in a {
            res.push(p.await);
        }
        res
    }

    /// Rigid-frame spokes — DERIVED: every recorded attachment whose offset ≠ COM
    /// yields a centre-of-mass → anchor segment, tagged with the body's
    /// [`MarkerKind`]. Works for ANY body that has off-centre anchors, so the
    /// frame stays correct automatically as the link topology changes.
    pub async fn frame_segments(&self) -> Vec<(Vector3<T>, Vector3<T>, MarkerKind)> {
        let z = Vector3::ZERO;
        let a: Vec<_> = self
            .rig
            .attachments
            .iter()
            .filter(|(_, local)| !Self::is_com(*local))
            .map(|&(id, local)| async move {
                (
                    self.world_of(id, z).await,
                    self.world_of(id, local).await,
                    self.kind_of(id),
                )
            })
            .collect();
        let mut res = Vec::new();
        for v in a {
            res.push(v.await);
        }
        res
    }
}

/// Holds all cameras and the shared gravity propagator they read as a sensor
/// (one world ⇒ one gravity). The anchor queries it via `force_on_probe`; no
/// camera registers with it, so cameras never perturb the simulation.
//#[derive(Debug)]
pub struct CameraField<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    world: Arc<World>,
    cameras: Arc<RwLock<HashMap<WorldId, Camera<T, S>>>>,
    gravity: Arc<GravityPropagator<T, S>>,
}

impl<
    T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod + 'static + Send + Sync,
    S: Scalar + StandardPart + From<T> + Into<T> + 'static,
> CameraField<T, S>
where
    <T as StandardPart>::Real: PartialOrd,
    <S as StandardPart>::Real: PartialOrd,
{
    pub fn new(world: Arc<World>, gravity: Arc<GravityPropagator<T, S>>) -> Self {
        Self {
            world,
            cameras: Arc::new(RwLock::new(HashMap::new())),
            gravity,
        }
    }

    /// Returns the camera's world id and the mechanism that fully represents
    /// the camera. Include the mechanism in the overall integration scheme to
    /// start moving the camera; pass the id back to update_camera / deregister
    /// to address the camera record after construction.
    pub async fn add_camera(
        self: Arc<Self>,
        target: Arc<Mechanism<T, S>>,
        r_object: T,
        linear: T,
    ) -> (WorldId, Arc<Mechanism<T, S>>) {
        let id = WorldId::get();

        let two = T::ONE + T::ONE;

        // The target's absolute CoM in S → lowered into T for the initial placement of the camera
        // bodies (the bodies are pure T; the islands re-anchor from them immediately on add/connect).
        let c_s = target.centroid().await.unwrap().coords();
        let tmp: [S; 3] = c_s.into();
        let base = Vector3::<T>::from(tmp.map(|v| v.into()));
        let mut initial_real_pos = base;
        let initial_intermediate_pos = base;
        let mut initial_anchor_pos = base;

        initial_real_pos[2] += r_object;
        initial_anchor_pos[1] += -r_object;

        let initial_real_vel = target.centroid_velocity().await.unwrap().linear();
        let initial_intermediate_vel = target.centroid_velocity().await.unwrap().linear();
        let initial_anchor_vel = target.centroid_velocity().await.unwrap().linear();

        // TODO: calculate r_object form a bounding box for a target
        // TODO: position camera's bodies "properly"

        // The camera rig shares the target's accelerator — the one GPU point.
        let mech = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
            target.accelerator().await,
        ))));
        let real = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                self.world.clone(),
                &initial_real_pos,
                &initial_real_vel,
                m_camera(),
            )))
            .await;
        let intermediate = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                self.world.clone(),
                &initial_intermediate_pos,
                &initial_intermediate_vel,
                m_camera(),
            )))
            .await;
        let anchor = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                self.world.clone(),
                &initial_anchor_pos,
                &initial_anchor_vel,
                m_camera(),
            )))
            .await;

        let target_holder = Arc::new(RwLock::new(None));
        let target_tracker = mech
            .add_body(Box::new(TargetTracker {
                id: WorldId::get(),
                // A kinematic reference body: pose and velocity are set in pre_step,
                // excluded from the solver's unknowns (see Inertia::Kinematic).
                body: RigidBody::new(self.world.clone(), Inertia::Kinematic),
                target: target_holder.clone(),
            }))
            .await;

        let s1 = r_object * two.sqrt_explicit();
        let z = T::ZERO;

        // Softening length for the rig joints: a small fraction of the object
        // radius (scene-scaled, not an absolute magic constant). Regularizes the
        // anchor-distance singularity (`1/d` at coincidence) without perturbing the
        // O(r_object)-scale springs.
        let softening = r_object * T::from_rational(1, 1000);
        let mut rig = RigBuilder::new(self.world.clone(), &mech, linear, softening);

        // TODO: most springs rest at exactly their initial anchor distance → zero
        // pretension, so the equilibrium is soft (geometric stiffness only, no
        // stress-stiffening). This deliberately frees the rotational DOFs, but the
        // cage is low-frequency; check the Hessian at equilibrium for unintended
        // near-zero modes beyond the intended free rotations.

        // Springs — offset anchors orient the rig.

        // real → intermediate
        rig.spring(
            real,
            Vector3::from([r_object, z, -r_object]),
            intermediate,
            Vector3::from([r_object, z, z]),
            z,
        )
        .await;
        rig.spring(
            real,
            Vector3::from([-r_object, z, -r_object]),
            intermediate,
            Vector3::from([-r_object, z, z]),
            z,
        )
        .await;
        rig.spring(
            real,
            Vector3::from([z, r_object, z]),
            intermediate,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            real,
            Vector3::from([z, -r_object, z]),
            intermediate,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;

        // intermediate → target
        rig.spring(
            intermediate,
            Vector3::from([r_object, -r_object, z]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([-r_object, -r_object, z]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([z, -r_object, -r_object]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([z, -r_object, r_object]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;

        rig.spring(
            intermediate,
            Vector3::from([r_object, r_object, z]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([-r_object, r_object, z]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([z, r_object, -r_object]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([z, r_object, r_object]),
            target_tracker,
            Vector3::from([z, z, z]),
            s1,
        )
        .await;

        // intermediate → anchor
        rig.spring(
            intermediate,
            Vector3::from([r_object, -r_object, z]),
            anchor,
            Vector3::from([z, z, z]),
            r_object,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([-r_object, -r_object, z]),
            anchor,
            Vector3::from([z, z, z]),
            r_object,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([z, -r_object, -r_object]),
            anchor,
            Vector3::from([z, z, z]),
            r_object,
        )
        .await;
        rig.spring(
            intermediate,
            Vector3::from([z, -r_object, r_object]),
            anchor,
            Vector3::from([z, z, z]),
            r_object,
        )
        .await;

        // anchor → target
        rig.spring(
            anchor,
            Vector3::from([z, z, z]),
            target_tracker,
            Vector3::from([z, z, z]),
            r_object,
        )
        .await;

        // Torsional (spin/roll) rate dampers. intermediate↔target_tracker damps
        // against the despun (ω=0) kinematic tracker, removing absolute spin;
        // real and anchor are despun transitively through intermediate.
        rig.torsional(real, intermediate).await;
        rig.torsional(intermediate, anchor).await;
        rig.torsional(intermediate, target_tracker).await;

        // Transverse (orbital-swing) rate dampers to the despun tracker: a DOF
        // neither the axial springs (radial) nor the torsional dampers (spin) touch.
        rig.transverse(intermediate, target_tracker).await;
        rig.transverse(anchor, target_tracker).await;

        // The anchor is not registered with the propagator: the camera reads
        // gravity as a sensor (CameraFieldArc::accumulate) but never gravitates
        // the world.
        let mut camera = Camera {
            id: WorldId::get(),
            real,
            anchor,
            target: target_holder,
            r_object,
            mechanism: mech.clone(),
            rig: rig.into_debug(intermediate, target_tracker),
        };
        camera.retarget(Some(target)).await;

        mech.add_field(Box::new(CameraFieldArc(self.clone()))).await;

        let mut guard = self.cameras.write().await;
        let _g = guard.insert(id, camera);

        (id, mech)
    }

    pub async fn deregister(&self, id: WorldId) {
        let mut guard = self.cameras.write().await;
        let _g = guard.remove(&id).unwrap();
    }

    pub async fn update_camera<F>(&self, id: WorldId, f: F)
    where
        F: AsyncFn(&mut Camera<T, S>),
    {
        let mut guard = self.cameras.write().await;
        if let Some(c) = guard.get_mut(&id) {
            f(c).await;
        }
    }

    pub async fn inspect_camera<F, R>(&self, id: WorldId, f: F) -> Option<R>
    where
        F: AsyncFn(&Camera<T, S>) -> R,
    {
        let guard = self.cameras.read().await;
        if let Some(c) = guard.get(&id) {
            Some(f(c).await)
        } else {
            None
        }
    }

    /// The eye's world-space pose. The view transform is its inverse. Absolute
    /// (island origin ⊕ local pose) — body poses are local after COM binding.
    pub async fn viewport(&self, id: WorldId) -> Option<Motor<T>> {
        let camera_info = {
            let guard = self.cameras.read().await;
            guard.get(&id).map(|c| (c.mechanism.clone(), c.real))
        }; // <-- Read lock guard drops here!

        // 2. Await the pose if camera was found
        if let Some((mech, real)) = camera_info {
            Some(mech.body_absolute_pose(real).await?)
        } else {
            None
        }
    }
}

pub(crate) struct CameraFieldArc<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
>(Arc<CameraField<T, S>>);

#[async_trait]
impl<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + StandardPart + From<T> + Into<T>,
> ForceField<T, S> for CameraFieldArc<T, S>
where
    <T as StandardPart>::Real: PartialOrd,
    <S as StandardPart>::Real: PartialOrd,
{
    async fn accumulate(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        out: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        epoch: &Epoch<T>,
        origin: &Vector3<S>,
    ) {
        let warp = *epoch.warp();

        let mut guard = self.0.cameras.write().await;
        for cam in guard.values_mut() {
            let anchor_id = &cam.anchor;
            let Some(anchor) = bodies.get(anchor_id) else {
                continue;
            };

            // Probe in the camera island's S frame (origin): the distance to the gravitators
            // is computed in S, the force is lowered into T.
            let probe = self.0.gravity.charge_of(&**anchor, origin);
            let inv_warp2 = T::ONE / warp.powi_explicit(2);

            let tmp = &out[anchor_id];
            tmp.write(tmp.read() + self.0.gravity.force_on_probe(&probe) * inv_warp2);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use newton::Accelerator;

    // Each test builds its own accelerator (the single-GPU sharing is a
    // production concern; isolated tests just need a working one).
    fn accel(world: Arc<World>) -> Arc<Accelerator<f32>> {
        Arc::new(Accelerator::builder(world.clone()).build())
    }

    // ========================================================================
    // Family A — construction & API invariants through CameraField::add_camera.
    // ========================================================================

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_pt(a: Vector3<f32>, b: Vector3<f32>, eps: f32) -> bool {
        a.split()
            .iter()
            .zip(b.split().iter())
            .all(|(x, y)| (x - y).abs() < eps)
    }

    /// One stationary body at `pos` with `mass`. Newton is already in scope;
    /// the target has no joints/fields so the integrator choice is immaterial.
    async fn target_mech_with_body(
        world: Arc<World>,
        pos: Vector3<f32>,
        mass: f32,
    ) -> Arc<Mechanism<f32, f32>> {
        let target = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
            accel(world.clone()),
        ))));
        target
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &pos,
                &Vector3::ZERO,
                mass,
            )))
            .await;
        target
    }

    /// Build a camera via the public path with gravity isolated to zero (G=0).
    async fn make_camera(
        world: Arc<World>,
        target: Arc<Mechanism<f32, f32>>,
        r_object: f32,
        c0: f32,
    ) -> (
        Arc<CameraField<f32, f32>>,
        WorldId,
        Arc<Mechanism<f32, f32>>,
    ) {
        let gravity = Arc::new(GravityPropagator::new(0.0));
        let field = Arc::new(CameraField::new(world, gravity));
        let (cam_id, cam_mech) = field.clone().add_camera(target, r_object, c0).await;
        (field, cam_id, cam_mech)
    }

    #[tokio::test]
    async fn add_camera_returns_mechanism_with_four_bodies() {
        let world = Arc::new(World::builder().usual::<f32>());
        let target = target_mech_with_body(world.clone(), Vector3::ZERO, 1.0).await;
        let (_field, _cam_id, cam_mech) = make_camera(world, target, 10.0, 40.0).await;
        let bodies: Vec<_> = cam_mech.components().await.into_iter().flatten().collect();
        assert_eq!(bodies.len(), 4, "camera mechanism must contain four bodies");
        let set: std::collections::HashSet<_> = bodies.into_iter().collect();
        assert_eq!(set.len(), 4, "ids must be distinct");
    }

    #[tokio::test]
    async fn add_camera_inherits_target_centroid_velocity() {
        // Target with a single moving body → non-zero centroid velocity.
        let world = Arc::new(World::builder().usual::<f32>());
        let target = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
            accel(world.clone()),
        ))));
        target
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::from([1.0, -2.0, 0.5]),
                1.0,
            )))
            .await;

        let (_field, _cam_id, cam_mech) = make_camera(world, target.clone(), 10.0, 40.0).await;

        let v_target = target.centroid_velocity().await.unwrap().linear();
        let v_cam = cam_mech.centroid_velocity().await.unwrap().linear();
        assert!(
            approx_pt(v_cam, v_target, 1e-12),
            "cam {v_cam:?} vs target {v_target:?}"
        );
    }

    #[tokio::test]
    async fn add_camera_stores_weak_target_that_upgrades() {
        let world = Arc::new(World::builder().usual::<f32>());

        let target = target_mech_with_body(world.clone(), Vector3::ZERO, 1.0).await;
        let (field, cam_id, _cam_mech) = make_camera(world, target.clone(), 10.0, 40.0).await;

        let guard = field.cameras.read().await;
        let cam = guard.get(&cam_id).unwrap();
        let guard = cam.target.read().await;
        let weak = guard.as_ref().expect("target was set in add_camera");
        assert!(
            weak.upgrade().is_some(),
            "weak ref must upgrade while Arc<target> lives"
        );
    }

    #[tokio::test]
    async fn add_camera_step_does_not_panic() {
        let world = Arc::new(World::builder().usual::<f32>());

        let target = target_mech_with_body(world.clone(), Vector3::ZERO, 1.0).await;
        let (_field, _cam_id, cam_mech) = make_camera(world, target, 10.0, 40.0).await;
        cam_mech.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
    }

    #[tokio::test]
    async fn deregister_removes_entry() {
        let world = Arc::new(World::builder().usual::<f32>());

        let target = target_mech_with_body(world.clone(), Vector3::ZERO, 1.0).await;
        let (field, cam_id, _cam_mech) = make_camera(world, target, 10.0, 40.0).await;
        assert_eq!(field.cameras.read().await.len(), 1);
        field.deregister(cam_id).await;
        assert!(field.cameras.read().await.is_empty());
    }

    #[tokio::test]
    async fn update_camera_can_clear_target_and_set_distance() {
        let world = Arc::new(World::builder().usual::<f32>());

        let target = target_mech_with_body(world.clone(), Vector3::ZERO, 1.0).await;
        let (field, cam_id, _cam_mech) = make_camera(world, target, 10.0, 40.0).await;

        field
            .update_camera(cam_id, async |c| {
                c.retarget(None).await;
                c.set_distance(42.0);
            })
            .await;

        let guard = field.cameras.read().await;
        let cam = guard.get(&cam_id).unwrap();
        assert!(
            cam.target.read().await.is_none(),
            "retarget(None) must clear target"
        );
        assert!(
            approx(cam.r_object, 42.0, 1e-12),
            "r_object = {}",
            cam.r_object
        );
    }

    // ========================================================================
    // Family B — long-run stability, mirroring `examples/simple.rs`.
    // ========================================================================

    /// Mirrors `examples/simple.rs`: a single stationary target body, the camera
    /// tracking it through its own mechanism. With a stationary target the camera
    /// must reach a steady stand-off and STAY bounded — it must not blow up.
    /// Regression for the "flies away after a while" report.
    /// Regression for the COM-anchoring frame bug: a target FAR from the global
    /// origin makes the camera island's anchor non-zero, so the tracker must take
    /// `target_centroid − anchor` in S. With the old (absolute-into-local) tracker
    /// the rig sprung toward `anchor + target` → it slowly rotated about the origin
    /// and retargeting failed. Here the eye must converge near the (far) target and
    /// then follow a retarget to a second far target.
    // Heavy (6000 stiff Newton steps) — runs on the fast CPU backend only.
    // Ignored: under f32 the stiff network with r_object=2 does not converge within the allotted
    // iterations → every step subdivides dt to the limit → the test does not finish. Waiting on
    // work on Newton solver convergence under f32 (not the GPU path — that one is fine).
    #[ignore]
    #[tokio::test]
    async fn tracks_far_target_and_retargets() {
        let world = Arc::new(World::builder().usual::<f32>());

        let far = Vector3::from([50.0f32, 0.0, 0.0]);
        let target = target_mech_with_body(world.clone(), far, 1.0).await;
        let far2 = Vector3::from([50.0f32, 40.0, 0.0]);
        let target2 = target_mech_with_body(world.clone(), far2, 1.0).await;

        // Gravity off (G=0) so this isolates the kinematic target-tracking frame.
        let prop = Arc::new(GravityPropagator::new(0.0));
        let field = Arc::new(CameraField::new(world, prop.clone()));
        let (cam_id, cam_mech) = field.clone().add_camera(target.clone(), 2.0, 1.0).await;

        let dt = 1.0 / 60.0;
        let dist_to = async |c: &Arc<CameraField<f32, f32>>, t: Vector3<f32>| {
            let p = c.inspect_camera(cam_id, Camera::position).await.unwrap();
            ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2) + (p[2] - t[2]).powi(2)).sqrt()
        };

        for step in 0..3000 {
            cam_mech.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            prop.advance_epoch();
            let d = dist_to(&field, far).await;
            assert!(
                d.is_finite() && d < 20.0,
                "camera ran away from far target at step {step}: dist = {d}"
            );
        }
        // Settled within a rig's reach of the far target (≈ r_object scale), NOT
        // stuck near the global origin.
        assert!(
            dist_to(&field, far).await < 8.0,
            "eye did not converge to far target: dist = {}",
            dist_to(&field, far).await
        );

        // Retarget to the second far target; the eye must move toward it.
        field
            .clone()
            .update_camera(cam_id, async |c| c.retarget(Some(target2.clone())).await)
            .await;
        let before = dist_to(&field, far2).await;
        for _ in 0..3000 {
            cam_mech.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            prop.advance_epoch();
        }
        let after = dist_to(&field, far2).await;
        assert!(
            after < before && after < 8.0,
            "retarget failed to pull the rig: {before} → {after}"
        );
    }

    // Heavy soak/diagnostic — CPU backend only (too slow under Lua).
    #[tokio::test]
    #[ignore]
    async fn tracks_stationary_target_without_diverging() {
        // Target: one inert body at the origin (mass like the example).
        let world = Arc::new(World::builder().usual::<f32>());
        let mech = Arc::new(Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
            Newton::new(accel(world.clone())),
        )));

        let body_id = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([0.0f32, 0.0, 0.0]),
                &Vector3::ZERO,
                1.0,
            )))
            .await;
        let prop = Arc::new(GravityPropagator::new(0.1));
        prop.register(body_id);
        mech.add_field(Box::new(prop.clone())).await;

        let field = Arc::new(CameraField::new(world, prop.clone()));
        let (cam_id, cam_mech) = field.clone().add_camera(mech.clone(), 2.0, 1.0).await;

        let dt = 1.0 / 60.0;
        let mut max_r = 0.0_f32;
        for step in 0..20_000 {
            let epoch = Epoch::standalone(dt, 1.0);
            mech.step(&epoch).await.unwrap();
            cam_mech.step(&epoch).await.unwrap();
            prop.advance_epoch();

            let p = field
                .inspect_camera(cam_id, Camera::position)
                .await
                .unwrap();
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            max_r = max_r.max(r);
            assert!(
                r.is_finite() && r < 100.0,
                "camera diverged at step {step} (t = {:.1}s): pos = {p:?}, |r| = {r}",
                step as f32 * dt
            );
        }
    }

    /// The rig starts at rest at its own equilibrium and every joint in it is
    /// dissipative, so nothing here has a source of energy: the eye's speed must
    /// stay at the noise floor. Growth means the solver is feeding a body forces
    /// that are not its own.
    ///
    /// This is the ONLY test in the workspace with a kinematic body that is not
    /// last in its island — the camera's despun tracker is merged in by the
    /// `intermediate → target` springs, which are declared before the anchor's.
    /// The fused gather/POST stage indexes its per-body inputs by the SOLVER's
    /// order, which omits kinematic bodies, so an island ordering that disagrees
    /// gave the anchor the tracker's external wrench and the tracker's incident
    /// connections. The rig then self-excited from machine noise and blew up in
    /// about eight seconds — see `newton::integrator::implicit::cache::bake_lanes`.
    #[tokio::test]
    async fn rig_at_equilibrium_does_not_self_excite() {
        let world = Arc::new(World::builder().usual::<f32>());
        let mech = Arc::new(Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
            Newton::new(accel(world.clone())),
        )));
        let body_id = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::ZERO,
                1.0,
            )))
            .await;
        let prop = Arc::new(GravityPropagator::new(0.1));
        prop.register(body_id);
        mech.add_field(Box::new(prop.clone())).await;
        let field = Arc::new(CameraField::new(world, prop.clone()));
        let (cam_id, cam_mech) = field.clone().add_camera(mech.clone(), 2.0, 1.0).await;
        let (real, anchor) = {
            let g = field.cameras.read().await;
            let c = g.get(&cam_id).unwrap();
            (c.real, c.anchor)
        };
        let dt = 1.0 / 60.0;
        for step in 0..600 {
            let epoch = Epoch::standalone(dt, 1.0);
            mech.step(&epoch).await.unwrap();
            cam_mech.step(&epoch).await.unwrap();
            prop.advance_epoch();

            // The rig starts at rest at its own equilibrium and every joint in it
            // is dissipative, so the only thing that can grow is an instability.
            // Fail as soon as the eye's speed leaves the noise floor — that is the
            // whole signal, and it makes this cheap enough to bisect on.
            let v = cam_mech
                .inspect_body(real, async |b| b.velocity().linear())
                .await
                .unwrap();
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                speed.is_finite() && speed < 1.0e-3,
                "rig self-excites: |v| = {speed:.3e} at step {step} (t = {:.2}s)",
                step as f32 * dt
            );

            if step % 20 == 0 {
                let pr = cam_mech
                    .inspect_body(real, async |b| b.position())
                    .await
                    .unwrap();
                let vr = cam_mech
                    .inspect_body(real, async |b| b.velocity().linear())
                    .await
                    .unwrap();
                let wr = cam_mech
                    .inspect_body(real, async |b| b.velocity().angular())
                    .await
                    .unwrap();
                let pa = cam_mech
                    .inspect_body(anchor, async |b| b.position())
                    .await
                    .unwrap();
                let r = (pr[0] * pr[0] + pr[1] * pr[1] + pr[2] * pr[2]).sqrt();
                let vmag = (vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]).sqrt();
                let wmag = (wr[0] * wr[0] + wr[1] * wr[1] + wr[2] * wr[2]).sqrt();
                eprintln!(
                    "t={:6.1} |r|={r:7.3} real={pr:?} |v|={vmag:.3e} |w|={wmag:.3e} anchor={pa:?}",
                    step as f32 * dt
                );
            }
        }
    }

    /// Sweep dt to reproduce the example's blow-up (example uses variable
    /// wall-clock dt capped at 0.1, i.e. up to 6× the 1/60 of the regression).
    // Heavy soak/diagnostic — CPU backend only (too slow under Lua).
    #[tokio::test]
    #[ignore]
    async fn diag_dt_sweep() {
        let world = Arc::new(World::builder().usual::<f32>());

        for &dt in &[1.0f32 / 60.0, 1.0 / 30.0, 0.05, 0.1] {
            let mech = Arc::new(Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
                Newton::new(accel(world.clone())),
            )));
            let body_id = mech
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::ZERO,
                    &Vector3::ZERO,
                    10.0,
                )))
                .await;
            let prop = Arc::new(GravityPropagator::new(0.1));
            prop.register(body_id);
            mech.add_field(Box::new(prop.clone())).await;
            let field = Arc::new(CameraField::new(world.clone(), prop.clone()));
            let (cam_id, cam_mech) = field.clone().add_camera(mech.clone(), 2.0, 1.0).await;

            let mut diverged_at = None;
            let mut max_r = 0.0_f32;
            for step in 0..40_000 {
                let epoch = Epoch::standalone(dt, 1.0);
                mech.step(&epoch).await.unwrap();
                cam_mech.step(&epoch).await.unwrap();
                prop.advance_epoch();
                let p = field
                    .inspect_camera(cam_id, Camera::position)
                    .await
                    .unwrap();
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                max_r = max_r.max(r);
                if !(r.is_finite() && r < 1000.0) {
                    diverged_at = Some((step, step as f32 * dt));
                    break;
                }
            }
            match diverged_at {
                Some((s, t)) => eprintln!("dt={dt:.4}: DIVERGED at step {s} (t={t:.1}s)"),
                None => eprintln!("dt={dt:.4}: stable, max_r={max_r:.3}"),
            }
        }
    }

    #[tokio::test]
    async fn two_cameras_coexist() {
        let world = Arc::new(World::builder().usual::<f32>());

        let target_a = target_mech_with_body(world.clone(), Vector3::ZERO, 1.0).await;
        let target_b =
            target_mech_with_body(world.clone(), Vector3::from([100.0, 0.0, 0.0]), 1.0).await;

        let gravity = Arc::new(GravityPropagator::new(0.0));
        let field = Arc::new(CameraField::new(world, gravity));

        let (id_a, mech_a) = field.clone().add_camera(target_a, 5.0, 40.0).await;
        let (id_b, mech_b) = field.clone().add_camera(target_b, 7.0, 40.0).await;

        assert_ne!(id_a, id_b);
        assert_eq!(field.cameras.read().await.len(), 2);

        mech_a.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
        mech_b.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
    }

    /// Locates the example's blow-up. Refutes the "anchor grazes the gravitating
    /// centre" hypothesis (G·m/r² spike): with G=0 the rig diverges at the SAME
    /// dt thresholds and times, max|F|=0. The real cause is Newton instability
    /// of the stiff spring/damper net at dt≥1/30 — stable at 1/60 — triggered by
    /// the target-switch transient (the despun tracker teleports across the A↔B
    /// gap, the springs see a huge stretch). The example's variable dt capped at
    /// 0.1 lets a hitching frame enter the unstable regime.
    // Heavy soak/diagnostic — CPU backend only (too slow under Lua).
    #[tokio::test]
    #[ignore]
    async fn diag_blowup_cause() {
        const SWITCH: f32 = 30.0;
        for &(g, dt) in &[
            (0.1, 1.0 / 60.0),
            (0.1, 1.0 / 30.0),
            (0.1, 0.05),
            (0.1, 0.1),
            (0.0, 0.1), // gravity off — isolates the spring/dt instability
            (0.0, 0.05),
            (0.0, 1.0 / 30.0),
        ] {
            let world = Arc::new(World::builder().usual::<f32>());
            let mech = Arc::new(Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
                Newton::new(accel(world.clone())),
            )));
            let body_a = mech
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::ZERO,
                    &Vector3::ZERO,
                    5.0,
                )))
                .await;
            let prop = Arc::new(GravityPropagator::new(g));
            prop.register(body_a);
            mech.add_field(Box::new(prop.clone())).await;

            let mech_b = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
                accel(world.clone()),
            ))));
            mech_b
                .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &Vector3::from([10.0, 0.0, 0.0]),
                    &Vector3::ZERO,
                    1.0,
                )))
                .await;

            let field = Arc::new(CameraField::new(world, prop.clone()));
            let (cam_id, cam_mech) = field.clone().add_camera(mech.clone(), 2.0, 1.0).await;
            let anchor = field.cameras.read().await.get(&cam_id).unwrap().anchor;

            let mut targeting_b = false;
            let mut elapsed = 0.0_f32;
            let mut min_anchor_dist = f32::INFINITY;
            let mut max_force = 0.0_f32;
            let mut diverged_at = None;

            let steps = (6.0 * 2.0 * SWITCH / dt) as usize;
            for step in 0..steps {
                let epoch = Epoch::standalone(dt, 1.0);
                mech.step(&epoch).await.unwrap();
                mech_b.step(&epoch).await.unwrap();
                cam_mech.step(&epoch).await.unwrap();
                prop.advance_epoch();

                elapsed += dt;
                let want_b = ((elapsed / SWITCH) as u64) % 2 == 1;
                if want_b != targeting_b {
                    targeting_b = want_b;
                    let tgt = if want_b { &mech_b } else { &mech }.clone();
                    field
                        .update_camera(cam_id, async |c| c.retarget(Some(tgt.clone())).await)
                        .await;
                }

                let pa = cam_mech
                    .inspect_body(anchor, async |b| b.position())
                    .await
                    .unwrap();
                let d = (pa[0] * pa[0] + pa[1] * pa[1] + pa[2] * pa[2]).sqrt(); // |anchor − A|
                let fa = cam_mech
                    .inspect_body(anchor, async |b| {
                        prop.force_on_probe(&prop.charge_of_body(b, &Vector3::ZERO))
                            .force()
                    })
                    .await
                    .unwrap();
                let fmag = (fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]).sqrt();
                min_anchor_dist = min_anchor_dist.min(d);
                max_force = max_force.max(fmag);

                let p = field
                    .inspect_camera(cam_id, Camera::position)
                    .await
                    .unwrap();
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                if !(r.is_finite() && r < 1.0e4) {
                    diverged_at = Some((step, step as f32 * dt));
                    break;
                }
            }
            match diverged_at {
                Some((s, t)) => eprintln!(
                    "G={g:.2} dt={dt:.4}: DIVERGED at step {s} (t={t:.1}s)  min|anchor−A|={min_anchor_dist:.4}  max|F|={max_force:.3e}"
                ),
                None => eprintln!(
                    "G={g:.2} dt={dt:.4}: stable  min|anchor−A|={min_anchor_dist:.4}  max|F|={max_force:.3e}"
                ),
            }
        }
    }

    /// Snapshot of the eye at the end of a settle window (just before a switch).
    #[derive(Clone, Copy)]
    struct WindowEnd {
        targeting_b: bool,
        standoff: f32, // |eye − active-target centroid|
        v: f32,        // eye linear speed
        w: f32,        // eye angular speed
    }

    fn vmag(a: Vector3<f32>) -> f32 {
        a.dot(a).sqrt()
    }

    /// Mirrors `examples/simple.rs`: a gravitating target A and a static,
    /// non-gravitating target B, one camera, target switched every `SWITCH`
    /// sim-seconds. Both targets are stationary, so once the rig settles the
    /// eye's residual speed must decay toward zero — and, crucially, must NOT
    /// grow cycle-over-cycle. Each retarget kinematically teleports the despun
    /// tracker across the A↔B gap and the springs yank the rig after it; this
    /// guards against that transient pumping energy that never dissipates.
    ///
    /// `#[ignore]`: ~26s soak (21k stiff 4-body steps). Run with `--ignored`.
    /// Stable only at dt≤1/60 — see [`diag_blowup_cause`] for the dt limit.
    // Heavy soak/diagnostic — CPU backend only (too slow under Lua).
    #[tokio::test]
    #[ignore]
    async fn mirrors_example_no_secular_accumulation() {
        const SWITCH: f32 = 30.0; // sim-seconds per target (example's SWITCH_PERIOD)
        const DT: f32 = 1.0 / 60.0; // virtual frame step
        const CYCLES: usize = 6; // full A→B→A… cycles
        const STEPS: usize = (CYCLES as f32 * 2.0 * SWITCH / DT) as usize;

        let world = Arc::new(World::builder().usual::<f32>());

        // Target A: gravitating (registered with the propagator), like the example.
        let mech = Arc::new(Mechanism::<f32, f32>::new(ImplicitIntegrator::Newton(
            Newton::new(accel(world.clone())),
        )));
        let body_a = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::ZERO,
                5.0,
            )))
            .await;
        let prop = Arc::new(GravityPropagator::new(0.1));
        prop.register(body_a);
        mech.add_field(Box::new(prop.clone())).await;

        // Target B: static, non-gravitating, 10 units away.
        let mech_b = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
            accel(world.clone()),
        ))));
        mech_b
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([10.0, 0.0, 0.0]),
                &Vector3::ZERO,
                1.0,
            )))
            .await;

        let field = Arc::new(CameraField::new(world, prop.clone()));
        let (cam_id, cam_mech) = field.clone().add_camera(mech.clone(), 2.0, 1.0).await;
        let real = field.cameras.read().await.get(&cam_id).unwrap().real;

        let mut targeting_b = false;
        let mut elapsed = 0.0_f32;
        let mut ends: Vec<WindowEnd> = Vec::new();
        // Worst transient anywhere in the run — must stay bounded/finite.
        let mut peak_v = 0.0_f32;
        let mut peak_w = 0.0_f32;
        let mut peak_r = 0.0_f32;

        let snapshot = async |targeting_b: bool| -> WindowEnd {
            let p = field
                .inspect_camera(cam_id, Camera::position)
                .await
                .unwrap();
            let active = if targeting_b { &mech_b } else { &mech };
            let c = active.centroid().await.unwrap().coords();
            let v = cam_mech
                .inspect_body(real, async |b| b.velocity().linear())
                .await
                .unwrap();
            let w = cam_mech
                .inspect_body(real, async |b| b.velocity().angular())
                .await
                .unwrap();
            WindowEnd {
                targeting_b,
                standoff: vmag(p - c),
                v: vmag(v),
                w: vmag(w),
            }
        };

        for step in 0..STEPS {
            let epoch = Epoch::standalone(DT, 1.0);
            mech.step(&epoch).await.unwrap();
            mech_b.step(&epoch).await.unwrap();
            cam_mech.step(&epoch).await.unwrap();
            prop.advance_epoch();

            elapsed += DT;
            let want_b = ((elapsed / SWITCH) as u64) % 2 == 1;
            if want_b != targeting_b {
                ends.push(snapshot(targeting_b).await); // record the window that just ended
                targeting_b = want_b;
                let tgt = if want_b { &mech_b } else { &mech }.clone();
                field
                    .update_camera(cam_id, async |c| c.retarget(Some(tgt.clone())).await)
                    .await;
            }

            let p = field
                .inspect_camera(cam_id, Camera::position)
                .await
                .unwrap();
            let v = cam_mech
                .inspect_body(real, async |b| b.velocity().linear())
                .await
                .unwrap();
            let w = cam_mech
                .inspect_body(real, async |b| b.velocity().angular())
                .await
                .unwrap();
            assert!(
                p.split().iter().all(|x| x.is_finite()),
                "eye position non-finite at step {step}: {p:?}"
            );
            peak_r = peak_r.max(vmag(p));
            peak_v = peak_v.max(vmag(v));
            peak_w = peak_w.max(vmag(w));
        }

        for (i, e) in ends.iter().enumerate() {
            eprintln!(
                "window {i:2} target={} standoff={:8.4} |v|={:.3e} |w|={:.3e}",
                if e.targeting_b { "B" } else { "A" },
                e.standoff,
                e.v,
                e.w,
            );
        }
        eprintln!("peak over run: |r|={peak_r:.4} |v|={peak_v:.3e} |w|={peak_w:.3e}");

        // Split the settled window-ends into early vs late halves; secular
        // accumulation would make the late half systematically larger.
        let a_ends: Vec<&WindowEnd> = ends.iter().filter(|e| !e.targeting_b).collect();
        let b_ends: Vec<&WindowEnd> = ends.iter().filter(|e| e.targeting_b).collect();
        let max = |xs: &[&WindowEnd], f: fn(&WindowEnd) -> f32| {
            xs.iter().map(|e| f(e)).fold(0.0_f32, f32::max)
        };

        for (label, es) in [("A", &a_ends), ("B", &b_ends)] {
            assert!(es.len() >= 4, "need ≥4 {label}-windows to judge a trend");
            let mid = es.len() / 2;
            let (early, late) = es.split_at(mid);

            // Residual rates: late half must not exceed early half by much.
            let v_early = max(early, |e| e.v).max(1e-9);
            let v_late = max(late, |e| e.v);
            assert!(
                v_late <= 4.0 * v_early,
                "{label}: residual |v| growing — early {v_early:.3e} → late {v_late:.3e}"
            );
            let w_early = max(early, |e| e.w).max(1e-9);
            let w_late = max(late, |e| e.w);
            assert!(
                w_late <= 4.0 * w_early,
                "{label}: residual |w| growing — early {w_early:.3e} → late {w_late:.3e}"
            );

            // Geometry: stand-off to the same target must not drift across visits.
            let s_first = es.first().unwrap().standoff;
            let s_last = es.last().unwrap().standoff;
            assert!(
                (s_last - s_first).abs() <= 0.05 * s_first.max(1e-9) + 1e-3,
                "{label}: stand-off drifting — first {s_first:.4} → last {s_last:.4}"
            );
        }

        assert!(
            peak_r.is_finite() && peak_r < 100.0,
            "eye left a sane region: peak |r| = {peak_r}"
        );
    }
}
