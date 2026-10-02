// SPDX-License-Identifier: MIT

use crate::{Accelerator, EvalError, Mechanism, StructureError};
use aristotle::{Epoch, WorldId};
use bytemuck::Pod;
use clifford::Lift;
use peano::prelude::*;
use std::sync::Arc;

/// Steps every mechanism of an epoch concurrently on ONE accelerator
/// (ACCELERATOR.md Part II). Batch breadth comes from here: while one
/// mechanism is parked on a dispatch, the others submit theirs, and the
/// accelerator flushes once all of them are parked (quiescence) or a batch is
/// full.
///
/// For the length of [`Driver::step`] every registered mechanism is frozen:
/// a structural change to any of them is refused with
/// [`StructureError::EpochInProgress`]. Together with single ownership of
/// bodies (`Mechanism::try_add_body`), that is the epoch invariant which makes
/// the interleaving order-independent — a mechanism's step reads nothing
/// another one writes mid-epoch, so stepping together is bit-exact with
/// stepping alone.
pub struct Driver<
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
> {
    accelerator: Arc<Accelerator<T>>,
    mechanisms: Vec<Arc<Mechanism<T, S>>>,
}

impl<T, S> Driver<T, S>
where
    T: Scalar + StandardPart + PartialOrd + Pod + Lift<T>,
    S: Scalar + From<T> + Into<T>,
{
    /// An empty driver over the accelerator its mechanisms must share.
    pub fn new(accelerator: Arc<Accelerator<T>>) -> Self {
        Self {
            accelerator,
            mechanisms: Vec::new(),
        }
    }

    /// Register a mechanism. Refused if it runs on another accelerator or is
    /// registered already. Registration needs `&mut self`, so it can never
    /// happen during [`Self::step`].
    pub async fn add(&mut self, mechanism: Arc<Mechanism<T, S>>) -> Result<(), StructureError> {
        let id = mechanism.id();
        if self.mechanisms.iter().any(|m| m.id() == id) {
            return Err(StructureError::DuplicateMechanism(id));
        }
        if !Arc::ptr_eq(&mechanism.accelerator().await, &self.accelerator) {
            return Err(StructureError::ForeignAccelerator(id));
        }
        self.mechanisms.push(mechanism);
        Ok(())
    }

    /// Unregister a mechanism by id, handing it back.
    pub fn remove(&mut self, id: WorldId) -> Option<Arc<Mechanism<T, S>>> {
        let at = self.mechanisms.iter().position(|m| m.id() == id)?;
        Some(self.mechanisms.remove(at))
    }

    /// The registered mechanisms, in registration order.
    pub fn mechanisms(&self) -> &[Arc<Mechanism<T, S>>] {
        &self.mechanisms
    }

    /// Step every registered mechanism through one epoch, concurrently.
    ///
    /// All of them are frozen before the first is polled and thawed after the
    /// last completes, so no structural change lands anywhere mid-epoch. Every
    /// mechanism runs to the end of its step even if another fails; the first
    /// failure in registration order is returned.
    pub async fn step(&self, epoch: &Epoch<T>) -> Result<(), EvalError> {
        let _frozen: Vec<_> = self.mechanisms.iter().map(|m| m.freeze()).collect();
        let steps = self.mechanisms.iter().map(|m| m.step(epoch)).collect();
        self.accelerator.epoch(steps).await.into_iter().collect()
    }
}
