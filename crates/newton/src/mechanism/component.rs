// SPDX-License-Identifier: MIT

use crate::RigidBody;
use aristotle::{Epoch, WorldId};
use async_trait::async_trait;
use bytemuck::Pod;
use peano::prelude::*;

// ============================================================================
// BEHAVIOR — portable body behavior (travels with the body on migration)
// ============================================================================

/// Logic and consumable state of a specific body: "a stage behaves like this",
/// "a battery behaves like this". Called in the pre-step phase BEFORE forces are computed — this is
/// where state is updated (fuel, inertia, engine mode) so that forces
/// are computed from the already up-to-date state.
///
/// Lives INSIDE an `Entity`, so on detach/attach it moves together with the body.
/// `RigidBody` meanwhile stays a pure Copy-POD.
#[async_trait]
pub trait Component<T: Scalar + Pod, S: Ring = T>: Send + Sync {
    /// Update the body's state before the step. `anchor` is the island origin in S (this is how
    /// absolute positions are ALWAYS expressed): the body holds its pose LOCALLY to it,
    /// so a behavior referring to an EXTERNAL absolute point (camera→target)
    /// takes the difference IN S (`external_abs_S − anchor`) and only then drops the small
    /// remainder into T. Default is nothing (an inert body).
    async fn pre_step(&mut self, _epoch: &Epoch<T>, _anchor: &Vector3<S>) {}
    fn body(&self) -> &RigidBody<T>;
    fn body_mut(&mut self) -> &mut RigidBody<T>;
    fn id(&self) -> WorldId;
}

/// Default inert behavior (a body with no logic of its own).
#[derive(Debug)]
pub struct Inert<T: Scalar + Pod> {
    // pub(crate): the duplicate_world_id_panics test in mod.rs forces a duplicate id
    // by assigning this field; the test used to live in the same file as Inert.
    pub(crate) id: WorldId,
    body: RigidBody<T>,
}

impl<T: Scalar + Pod, S: Ring> Component<T, S> for Inert<T> {
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

impl<T: Scalar + Pod> Inert<T> {
    pub fn new(body: RigidBody<T>) -> Box<Self> {
        Box::new(Self {
            id: WorldId::get(),
            body,
        })
    }
}

/// A body plus its portable behavior. The unit that migrates between mechanisms.
pub struct Entity<T, F>
where
    T: Scalar + Pod,
    F: FnMut(&mut RigidBody<T>, &Epoch<T>) + 'static,
{
    id: WorldId,
    body: RigidBody<T>,
    behavior: Box<F>,
}

impl<T: Scalar + Pod, F> Entity<T, F>
where
    F: FnMut(&mut RigidBody<T>, &Epoch<T>) + 'static,
{
    #[inline]
    pub fn new(body: RigidBody<T>, behavior: F) -> Box<Self> {
        Box::new(Self {
            id: WorldId::get(),
            body,
            behavior: Box::new(behavior),
        })
    }
}

#[async_trait]
impl<T: Scalar + Pod, S: Ring, F> Component<T, S> for Entity<T, F>
where
    F: FnMut(&mut RigidBody<T>, &Epoch<T>) + 'static + Send + Sync,
{
    // The private behavior does not see the island anchor (its job is the consumable
    // state of the body, not absolute geometry); the anchor is ignored.
    async fn pre_step(&mut self, epoch: &Epoch<T>, _anchor: &Vector3<S>) {
        let Self { body, behavior, .. } = self;
        (behavior)(body, epoch);
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
