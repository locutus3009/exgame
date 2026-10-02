// SPDX-License-Identifier: MIT

//! The incidence ROW: the baked, index-only description of one unit of
//! accelerator work. A row lives in World storage like any other datum, so it is
//! already device-visible and `setup` never has to serialise anything — it
//! writes the row's slot index into the batch table and stops.
//!
//! A row is a flat `u32` array; each stage overlays its own word layout,
//! documented in that stage's module under `rows/`. Scalars ride as
//! `f32::to_bits` and come back through GLSL's `uintBitsToFloat`, so the storage
//! type stays purely indexical and one type serves every stage.
//!
//! Rows are baked on topology change, not per step. The cost that motivates this
//! is not the index arithmetic — that is a few integer ops per term — but the
//! `WorldKey` dereference: a key is an `Arc<WorldKeyInner<T>>` and the inners are
//! scattered, so resolving a term list per dispatch is a likely cache miss per
//! term, against arithmetic the GPU finishes in well under a nanosecond.
//!
//! `ROW` is a CEILING, not a tuning knob: a reduction that needs more terms than
//! fit is baked as several rows and dispatched as several rounds, never as a
//! wider row.

/// Words per row.
pub(crate) const ROW: usize = 128;

/// One baked row.
pub(crate) type Row = [u32; ROW];

/// Pack a scalar into a row word. The kernel reads it back with
/// `uintBitsToFloat`; nothing in between interprets it as a number.
#[allow(dead_code)]
pub(crate) fn bits<T: bytemuck::Pod + 'static>(x: T) -> u32 {
    // `super::`, not `crate::accelerator::` — the path survives the module
    // moving, which is the convention in this tree.
    f32::to_bits(super::shaders::t_to_f32(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aristotle::World;
    use std::sync::Arc;

    #[test]
    fn a_row_round_trips_through_world_storage() {
        let world = Arc::new(World::builder().usual::<f32>());
        let key = {
            let mut map = world.write::<Row>();
            let mut r: Row = [0; ROW];
            r[0] = 7;
            r[1] = f32::to_bits(0.5);
            map.add(r)
        };
        let got = key.read();
        assert_eq!(got[0], 7);
        assert_eq!(f32::from_bits(got[1]), 0.5);
    }

    /// `aristotle::usual` registers the row storage with a LITERAL width,
    /// because it cannot depend on newton. If `ROW` ever changes, this fails
    /// loudly instead of the storage lookup panicking at runtime.
    #[test]
    fn row_width_matches_the_registered_storage() {
        assert_eq!(ROW, 128);
    }
}
