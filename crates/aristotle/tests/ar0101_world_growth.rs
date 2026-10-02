// SPDX-License-Identifier: MIT

//! World storage grows past its initial capacity instead of writing past the
//! end of its buffer.

use aristotle::{World, WorldKey};
use std::any::TypeId;
use std::sync::Arc;

const INITIAL: usize = 8;

fn world() -> Arc<World> {
    Arc::new(
        World::builder()
            .capacity(INITIAL)
            .with_storage::<u32>()
            .with_storage::<[f32; 3]>()
            .build(),
    )
}

#[test]
fn insert_past_capacity_reads_every_value_back() {
    let w = world();
    let n = INITIAL * 5 + 3;
    let keys: Vec<WorldKey<u32>> = {
        let mut map = w.write::<u32>();
        (0..n as u32).map(|i| map.add(i * 3 + 1)).collect()
    };
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(k.raw_index(), i, "growth must keep raw indices");
        assert_eq!(k.read(), i as u32 * 3 + 1);
    }
    let view = w.read::<u32>();
    assert!(view.capacity() >= n);
    assert!(view.len() == n);
    assert!(view.generation() > 0);
    assert_eq!(view.generation(), w.generation(TypeId::of::<u32>()));
    // The buffer the accelerator would bind holds the same values.
    let data = view.get_map().read().unwrap();
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(data[k.raw_index()], i as u32 * 3 + 1);
    }
}

#[test]
fn writes_before_and_after_growth_survive() {
    let w = world();
    let early: Vec<_> = {
        let mut map = w.write::<[f32; 3]>();
        (0..INITIAL).map(|i| map.add([i as f32; 3])).collect()
    };
    early[0].write([-1.0; 3]);
    let generation = w.generation(TypeId::of::<[f32; 3]>());
    let late: Vec<_> = {
        let mut map = w.write::<[f32; 3]>();
        (0..INITIAL * 2)
            .map(|i| map.add([100.0 + i as f32; 3]))
            .collect()
    };
    assert!(w.generation(TypeId::of::<[f32; 3]>()) > generation);
    // Only the grown type's generation moved.
    assert_eq!(w.generation(TypeId::of::<u32>()), 0);
    late[3].write([7.0; 3]);
    assert_eq!(early[0].read(), [-1.0; 3]);
    for (i, k) in early.iter().enumerate().skip(1) {
        assert_eq!(k.read(), [i as f32; 3]);
    }
    assert_eq!(late[3].read(), [7.0; 3]);
    assert_eq!(late[4].read(), [104.0; 3]);
}

#[test]
fn freed_slots_are_reused_before_growing() {
    let w = world();
    let keys: Vec<_> = {
        let mut map = w.write::<u32>();
        (0..INITIAL as u32).map(|i| map.add(i)).collect()
    };
    let generation = w.generation(TypeId::of::<u32>());
    // Dropped outside any guard: the last handle takes `World::write`.
    drop(keys);
    let again: Vec<_> = {
        let mut map = w.write::<u32>();
        (0..INITIAL as u32).map(|i| map.add(i + 50)).collect()
    };
    assert_eq!(w.generation(TypeId::of::<u32>()), generation);
    assert_eq!(w.read::<u32>().capacity(), INITIAL);
    for (i, k) in again.iter().enumerate() {
        assert_eq!(k.read(), i as u32 + 50);
    }
}

#[test]
#[should_panic(expected = "capacity must be at least 1")]
fn zero_capacity_is_refused() {
    let _ = World::builder().capacity(0);
}
