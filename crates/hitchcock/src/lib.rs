// SPDX-License-Identifier: MIT

//! Camera as a world object: a self-orienting rig in its own `newton`
//! `Mechanism`, with the viewport derived from the eye body's pose.
//!
//! The rig is four bodies (eye/`real`, intermediate, anchor, despun
//! target_tracker) wired by axial springs and torsional/transverse dampers.
//! It reads gravity through the anchor as a sensor and never perturbs the
//! simulation. WIP — geometry and stiffnesses are still being tuned.

mod camera;
mod rig;

pub use camera::{Camera, CameraField};
pub use rig::MarkerKind;
