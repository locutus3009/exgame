// SPDX-License-Identifier: MIT

use aristotle::{World, WorldId};
use bytemuck::Pod;
use clifford::Lift;
use joints::{
    AxialSpringDamper, CriticallyDampedWarped, PerpendicularDamperWarped, TorsionalDamperWarped,
};
use newton::Mechanism;
use peano::prelude::*;
use std::sync::Arc;

/// A joint endpoint: the body it hangs off, and the anchor's offset in that
/// body's local frame.
pub(crate) type Anchor<T> = (WorldId, Vector3<T>);
/// A longitudinal spring, as the pair of endpoints it derives its world line
/// from.
pub(crate) type SpringLine<T> = (Anchor<T>, Anchor<T>);

/// Kind of a debug marker returned by [`crate::Camera::markers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    /// The camera body (the eye / `real`).
    Eye,
    Intermediate,
    Anchor,
    TargetTracker,
    /// A spring attachment point (offset from a body's centre of mass).
    Attachment,
}

/// Visualization-only rig layout, accumulated as a by-product of building the
/// links (see [`RigBuilder`]): body ids plus every spring attachment point,
/// turned into world markers/segments by [`Camera`].
#[derive(Debug)]
pub(crate) struct RigDebug<T: Scalar> {
    pub(crate) intermediate: WorldId,
    pub(crate) target_tracker: WorldId,
    /// Every joint anchor `(body, local)` seen while building (both endpoints of
    /// each spring). Drives the attachment markers and the frame spokes.
    pub(crate) attachments: Vec<Anchor<T>>,
    /// Longitudinal spring lines as world-deriving endpoint pairs.
    pub(crate) springs: Vec<SpringLine<T>>,
}

/// Wraps `Mechanism::connect`, recording each link into a [`RigDebug`] as it is
/// added. Spring anchors feed the spring lines and (when offset from the COM)
/// the frame spokes; pure dampers act at the COM and add no frame geometry.
pub(crate) struct RigBuilder<
    'm,
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    world: Arc<World>,
    mech: &'m Mechanism<T, S>,
    linear: T,
    /// Softening length ε for the distance-using joints (axial spring + transverse
    /// damper): keeps `1/d` and its AD gradient finite through anchor coincidence.
    /// Scene-scaled by the caller (a small fraction of the object radius).
    softening: T,
    attachments: Vec<Anchor<T>>,
    springs: Vec<SpringLine<T>>,
}

impl<
    'm,
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T> + 'static + Send + Sync,
    S: Scalar + From<T> + Into<T>,
> RigBuilder<'m, T, S>
{
    pub(crate) fn new(
        world: Arc<World>,
        mech: &'m Mechanism<T, S>,
        linear: T,
        softening: T,
    ) -> Self {
        Self {
            world,
            mech,
            linear,
            softening,
            attachments: Vec::new(),
            springs: Vec::new(),
        }
    }

    /// Critically-damped axial spring between offset anchors.
    pub(crate) async fn spring(
        &mut self,
        a: WorldId,
        a_local: Vector3<T>,
        b: WorldId,
        b_local: Vector3<T>,
        rest: T,
    ) {
        let joint = AxialSpringDamper::builder(self.world.clone(), CriticallyDampedWarped)
            .a(a_local)
            .b(b_local)
            .rest(rest)
            .stiffness(self.linear)
            .softening(self.softening)
            .build();
        self.mech.connect(a, vec![(joint, b)]).await;
        self.attachments.push((a, a_local));
        self.attachments.push((b, b_local));
        self.springs.push(((a, a_local), (b, b_local)));
    }

    /// Torsional (spin) rate damper at the centres of mass.
    pub(crate) async fn torsional(&mut self, a: WorldId, b: WorldId) {
        self.mech
            .connect(
                a,
                vec![(
                    TorsionalDamperWarped::new_joint(self.world.clone(), self.linear),
                    b,
                )],
            )
            .await;
    }

    /// Transverse (orbital-swing) rate damper at the centres of mass.
    pub(crate) async fn transverse(&mut self, a: WorldId, b: WorldId) {
        self.mech
            .connect(
                a,
                vec![(
                    PerpendicularDamperWarped::new_joint(
                        self.world.clone(),
                        self.linear,
                        self.softening,
                    ),
                    b,
                )],
            )
            .await;
    }

    pub(crate) fn into_debug(self, intermediate: WorldId, target_tracker: WorldId) -> RigDebug<T> {
        RigDebug {
            intermediate,
            target_tracker,
            attachments: self.attachments,
            springs: self.springs,
        }
    }
}
