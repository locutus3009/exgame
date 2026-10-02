// SPDX-License-Identifier: MIT

use clifford::pga3::{Motor, Twist, Wrench};
use peano::prelude::*;
use rembrandt::{
    AnyVec, GpuAccelerator,
    bytemuck::{Pod, Zeroable},
    vulkano::buffer::{BufferMemory, Subbuffer},
};
use std::any::TypeId;
use std::collections::{HashMap, hash_map::Entry};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
#[bytemuck(crate = "rembrandt::bytemuck")]
struct RawMapKey {
    is_value: u32,
    next: u32,
    generation: u32,
}

impl RawMapKey {
    fn value(generation: u32) -> Self {
        Self {
            is_value: 1,
            next: 0,
            generation,
        }
    }

    fn free(next: u32, generation: u32) -> Self {
        Self {
            is_value: 0,
            next,
            generation,
        }
    }

    fn get_value(&self) -> Option<u32> {
        if self.is_value == 1 {
            Some(self.generation)
        } else {
            None
        }
    }

    fn get_free(&self) -> Option<(u32, u32)> {
        if self.is_value == 0 {
            Some((self.next, self.generation))
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct MapKey<T: Pod> {
    value: u32,
    generation: u32,
    _marker: PhantomData<T>,
}

struct RawMap {
    // Flat payload storage: a type-erased `Subbuffer<[T]>` of `capacity`
    // elements in host-visible device memory.
    //
    // The STRUCTURAL hazard is a reallocation (`grow`) or slot reuse changing what
    // the indices the accelerator holds point at. Growth copies every slot to the
    // same position, so indices survive it; what does NOT survive is the buffer
    // handle, which is why growth bumps `generation`. Structural mutation takes
    // `&mut self`, i.e. this map's write guard.
    //
    // There are two doors to the elements, and they are not merged:
    //   * the HOST door, `WorldKey::read`/`write`: `read_at`/`write_at` below,
    //     through the cached mapping or the guarded `Subbuffer` accessors. A
    //     read takes this map's read guard, a write its WRITE guard, so host
    //     writes are serialized against every other host access and are safe
    //     under arbitrary concurrency;
    //   * the DEVICE door: a descriptor set binds this `Subbuffer` itself
    //     (`ReadView::get_map`), and the kernels of one flush read and write its
    //     slots in place. `Shaders::dispatch` holds `World::write_guard` on every
    //     storage it binds from submit to fence, so the host door is shut while
    //     the device is in, and a `grow` cannot swap the buffer under it.
    // What the guard cannot order is the device against ITSELF: invocations of
    // one flush run unordered, so the door is sound only while each slot has at
    // most one writer per flush. Newton checks that
    // before submitting (`accelerator/shaders.rs`, `Ledger`) and refuses a
    // flush whose rows name one output slot twice.
    values: Box<dyn AnyVec>,
    /// Elements `values` holds. Every slot in `map` is below it — `insert` grows
    /// the buffer before it hands out a slot past the end — and both access paths
    /// check it, so no read or write lands past the end of the buffer.
    capacity: usize,
    /// Bumped by every `grow`, which replaces `values` with a new buffer. Anyone
    /// holding a clone of the old buffer — a descriptor set, above all — compares
    /// against it to see that it must rebind. Shared with `World`, so it can be
    /// read without this map's lock; only `grow`, under the write guard, moves it.
    generation: Arc<AtomicU64>,
    /// Where `grow` allocates the replacement buffer.
    gpu: Arc<GpuAccelerator>,
    /// The host mapping of `values`, taken once per buffer: at construction and
    /// again by every `grow`.
    ///
    /// `Subbuffer::read`/`write` are not free. Each one takes a mutex on the
    /// buffer's state tracker, checks its range against an interval tree of
    /// in-flight device accesses, records the host lock, and unrecords it when the
    /// guard drops. Measured on a 66-unknown cloth that machinery was 27% of the
    /// step's host time — `Buffer::state` 17.8%, the interval tree 4.6%, the
    /// guard's drop 5.2% — and it is a check we already make one level up:
    /// `Shaders::dispatch` holds `World::write_guard` over every storage it binds
    /// for the whole submission, so host and device never touch this memory at
    /// once. The per-slot tracking re-derives that per access.
    ///
    /// `None` in two cases, and then every access goes back through the guarded
    /// path:
    ///
    /// - the memory turned out NON-COHERENT, so a host read must invalidate the
    ///   range and a host write must flush it. The guarded path does that and a
    ///   raw pointer would silently skip it. A fallback rather than a refusal,
    ///   because which memory type a driver hands out is not ours to pick;
    /// - the `tracked-access` feature is on. What the mapping gives up is
    ///   vulkano's per-access check against in-flight device work, which turns a
    ///   host touch racing the GPU into a panic AT THE OFFENDING ACCESS instead of
    ///   a quietly corrupted slot. That check has earned its keep before, so it
    ///   stays one flag away rather than being deleted.
    base: Option<NonNull<u8>>,
    /// The `T` this map was built for. The guarded path checks it by downcast;
    /// the mapped path has no such check of its own, and getting it wrong would
    /// reinterpret the storage rather than panic — so the check moves here, where
    /// both paths pay for it. One integer compare against what `Subbuffer::read`
    /// used to cost is not a trade worth thinking about.
    ty: TypeId,
    map: Vec<RawMapKey>,
    first_free: u32,
}

impl RawMap {
    fn new<T: Pod + Send + Sync>(gpu: &Arc<GpuAccelerator>, capacity: usize) -> Self {
        // A zero-length buffer is not a valid Vulkan allocation; one slot is the
        // smallest storage that can grow geometrically.
        let capacity = capacity.max(1);
        let values = gpu.allocate_buffer::<T>(capacity);
        let base = Self::host_base(&values);
        Self {
            values: Box::new(values),
            capacity,
            generation: Arc::new(AtomicU64::new(0)),
            gpu: gpu.clone(),
            base,
            ty: TypeId::of::<T>(),
            map: Vec::with_capacity(capacity),
            first_free: 0,
        }
    }

    /// The cached host mapping of `values`, or `None` for the guarded path.
    ///
    /// Coherent memory only, and only if the mapping is actually there and
    /// aligned for `T`. Anything else — including the collision detector being
    /// switched on — keeps the guarded path.
    fn host_base<T: Pod>(values: &Subbuffer<[T]>) -> Option<NonNull<u8>> {
        let coherent = !cfg!(feature = "tracked-access")
            && match values.buffer().memory() {
                BufferMemory::Normal(mem) => mem.atom_size().is_none(),
                _ => false,
            };
        coherent
            .then(|| values.mapped_slice().ok())
            .flatten()
            .map(|s| s.as_ptr().cast::<u8>())
            .filter(|p| p.addr() % align_of::<T>() == 0)
            .and_then(NonNull::new)
    }

    /// Reallocate the payload to hold at least `min` elements: geometric growth,
    /// every live slot copied to the SAME position, the host mapping re-taken
    /// and `generation` bumped.
    ///
    /// Takes `&mut self`, i.e. the structural write guard. That guard is also
    /// what `Shaders::dispatch` holds over every storage it binds from submit to
    /// fence, so no device access to the old buffer is in flight while it is
    /// copied. Raw indices are unchanged, so a queued message that holds one
    /// stays valid; only a descriptor set that bound the old buffer is stale,
    /// and the new generation is how its owner learns that.
    fn grow<T: Pod + Send + Sync>(&mut self, min: usize) {
        self.assert_type::<T>();
        let capacity = min.max(self.capacity.saturating_mul(2));
        let fresh = self.gpu.allocate_buffer::<T>(capacity);
        let len = self.map.len();
        {
            // The guarded accessors on both ends, not the cached pointer: they
            // invalidate and flush when the memory is non-coherent, and growth
            // is rare enough that their bookkeeping does not matter.
            let old = self
                .values::<T>()
                .read()
                .expect("world storage grown while the device still uses it");
            let mut new = fresh
                .write()
                .expect("freshly allocated world storage is not host-writable");
            new[..len].copy_from_slice(&old[..len]);
        }
        self.base = Self::host_base(&fresh);
        self.values = Box::new(fresh);
        self.capacity = capacity;
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Element `position` of this map, through the cached mapping when there is
    /// one.
    ///
    /// # Safety
    /// `position` must be in bounds of the buffer, and the caller must uphold the
    /// storage's own discipline: at most one writer per slot, and no device access
    /// to this storage in flight (which `World::write_guard` is what enforces).
    /// `T` must be the type this map was built with; `assert_type` is what both
    /// callers check that with.
    #[inline]
    unsafe fn at<T: Pod>(base: NonNull<u8>, position: usize) -> *mut T {
        // SAFETY: the caller guarantees `position` is within the buffer `base`
        // maps, so the offset stays inside that one allocation.
        unsafe { base.as_ptr().cast::<T>().add(position) }
    }

    /// Downcast of the payload storage back to the typed `Subbuffer<[T]>`. A real
    /// `Any` check instead of a manual `type_id` + `debug_assert`: `expect` fires
    /// if a `T` other than the one stored in this map was requested.
    pub fn values<T: Pod>(&self) -> &Subbuffer<[T]> {
        // Reached under the map's guard; `&self` rules out a concurrent `grow`
        // replacing the buffer (which requires `&mut self`).
        self.values
            .as_any()
            .downcast_ref::<Subbuffer<[T]>>()
            .expect("Attempt to access an element of different type")
    }

    fn payload_len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    fn assert_type<T: Pod + 'static>(&self) {
        assert!(
            self.ty == TypeId::of::<T>(),
            "Attempt to access an element of different type"
        );
    }

    /// The bound both access paths share. `insert` keeps every live slot below
    /// `capacity`, so this never fires through the public API; it is here so
    /// that the raw-pointer path cannot write past the mapping even if that
    /// invariant is ever broken, and so the guarded path fails with a message
    /// rather than a bare slice index.
    #[inline]
    fn assert_in_bounds(&self, position: usize) {
        assert!(
            position < self.capacity,
            "world storage slot {position} is past the buffer's capacity {}",
            self.capacity
        );
    }

    #[inline]
    fn read_at<T: Pod + Send + Sync>(&self, position: usize) -> T {
        self.assert_type::<T>();
        self.assert_in_bounds(position);
        match self.base {
            // SAFETY: `assert_in_bounds` holds `position` below `capacity`, the
            // length of the buffer `base` maps; `assert_type` makes `T` the type
            // it was built for, and `host_base` only hands out a coherent mapping
            // aligned for `T`. The caller holds this map's read or write guard,
            // which `Shaders::dispatch`'s write guard excludes, so the device is
            // not writing the storage meanwhile; a host writer would need the
            // write guard too, so none runs concurrently either.
            Some(base) => unsafe { Self::at::<T>(base, position).read() },
            None => self.values::<T>().read().unwrap()[position],
        }
    }

    /// Takes `&self` on purpose: a slot write is not a structural change, and the
    /// storage's invariant is one writer per SLOT rather than one per map.
    #[inline]
    fn write_at<T: Pod + Send + Sync>(&self, position: usize, value: T) {
        self.assert_type::<T>();
        self.assert_in_bounds(position);
        match self.base {
            // SAFETY: bounds, type and mapping as in `read_at`. Every caller holds
            // this map exclusively — `set` through a `WriteView`, `insert` through
            // `&mut self` — so no other host access and no device access (which
            // needs the write guard `dispatch` takes) overlaps this store.
            Some(base) => unsafe { Self::at::<T>(base, position).write(value) },
            None => self.values::<T>().write().unwrap()[position] = value,
        }
    }
}

impl RawMap {
    fn insert<T: Pod + Send + Sync>(&mut self, value: T) -> MapKey<T> {
        let position = self.first_free;

        if position as usize >= self.map.len() {
            // fresh slot at the end — grow first if the buffer is full
            if position as usize >= self.capacity {
                self.grow::<T>(position as usize + 1);
            }
            self.first_free = position + 1;
            self.map.push(RawMapKey::value(0));
            self.write_at::<T>(position as usize, value);
            MapKey {
                value: position,
                generation: 0,
                _marker: PhantomData,
            }
        } else {
            // reuse a free slot — take its gen
            let (next, generation) = self.map[position as usize]
                .get_free()
                .expect("Free list points to a non-free element!");
            self.first_free = next;
            self.map[position as usize] = RawMapKey::value(generation);
            self.write_at::<T>(position as usize, value);
            MapKey {
                value: position,
                generation,
                _marker: PhantomData,
            }
        }
    }

    fn get<T: Pod + Send + Sync>(&self, key: &MapKey<T>) -> Option<T> {
        let position = key.value as usize;
        if let Some(g) = self.map.get(position)?.get_value()
            && g == key.generation
        {
        } else {
            return None; // free, stale gen, or a foreign key out of range
        }
        Some(self.read_at::<T>(position))
    }

    fn set<T: Pod + Send + Sync>(&self, key: &MapKey<T>, value: T) {
        let position = key.value as usize;
        if let Some(g) = self.map.get(position).unwrap().get_value()
            && g == key.generation
        {
            self.write_at::<T>(position, value);
        }
    }

    fn remove<T: Pod + Send + Sync>(&mut self, key: &MapKey<T>) -> Option<T> {
        let out = self.get(key)?; // gen check inside get: None for stale/free/foreign

        let position = key.value;
        let mut it = self.first_free;
        let mut prev: Option<u32> = None;
        loop {
            if position < it {
                // insert position before it, bump its generation
                let new_gen = key.generation.wrapping_add(1);
                self.map[position as usize] = RawMapKey::free(it, new_gen);
                match prev {
                    None => self.first_free = position,
                    Some(i) => {
                        // relink prev -> position, PRESERVING node prev's own gen
                        let (_, g) = self.map[i as usize]
                            .get_free()
                            .expect("prev is a free-list node");
                        self.map[i as usize] = RawMapKey::free(position, g);
                    }
                }
                break;
            }
            prev = Some(it);
            (it, _) = self.map[it as usize]
                .get_free()
                .expect("Non-free element in a free list!");
        }

        Some(out)
    }
}

// SAFETY: the one field that is not `Sync` is `base`, the raw pointer into the
// host mapping of `values` (`NonNull` opts out of both auto traits); everything
// else is `Sync` already. What `&RawMap` allows through `base` is `read_at` — a
// plain read of a `Pod` slot, harmless in parallel — and `write_at`, whose every
// caller holds the map exclusively (see there). Mutating `base` itself needs
// `&mut self`. Shared access from several threads therefore never races a store.
unsafe impl Sync for RawMap {}

// SAFETY: `base` points into the host mapping of the buffer this map owns, which
// lives as long as that buffer and is not tied to the creating thread; `grow`
// replaces the two together. Everything else in `RawMap` is already `Send`.
unsafe impl Send for RawMap {}

struct WorldInner {
    gpu: Arc<GpuAccelerator>,
    map: HashMap<TypeId, RwLock<RawMap>>,
    /// Each map's `generation`, reachable without its lock.
    generations: HashMap<TypeId, Arc<AtomicU64>>,
}

pub struct World {
    inner: WorldInner,
}

impl std::fmt::Debug for World {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        Ok(())
    }
}

pub struct ReadView<'a, T: Pod> {
    inner: RwLockReadGuard<'a, RawMap>,
    _marker: PhantomData<T>,
}

impl<'a, T: Pod> ReadView<'a, T> {
    pub fn len(&self) -> usize {
        self.inner.payload_len()
    }

    pub fn get_map(&self) -> &Subbuffer<[T]> {
        self.inner.values()
    }

    /// Elements the buffer `get_map` returns can hold.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// The generation of the buffer `get_map` returns. Read under the same guard,
    /// so the pair is consistent: a clone of that buffer is current exactly while
    /// `World::generation` still reports this number.
    pub fn generation(&self) -> u64 {
        self.inner.generation.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone)]
pub struct WorldKey<T: Pod + Send + Sync> {
    inner: Arc<WorldKeyInner<T>>,
}

#[derive(Debug)]
struct WorldKeyInner<T: Pod + Send + Sync> {
    world: Arc<World>,
    key: MapKey<T>,
}

impl<T: Pod + Send + Sync> Drop for WorldKeyInner<T> {
    fn drop(&mut self) {
        let mut map = self.world.write::<T>();
        map.inner.remove(&self.key);
    }
}

impl<T: Pod + Send + Sync> WorldKey<T> {
    pub fn write(&self, value: T) {
        let map = self.inner.world.write::<T>();
        map.inner.set(&self.inner.key, value);
    }

    pub fn read(&self) -> T {
        let map = self.inner.world.read::<T>();
        map.inner.get(&self.inner.key).unwrap()
    }

    /// The `World` this key allocates in — lets a holder allocate a sibling key
    /// in the same world without threading a separate `Arc<World>` handle.
    pub fn world(&self) -> Arc<World> {
        self.inner.world.clone()
    }

    /// Position of the value in its type's dense vector — the address a baked
    /// accelerator row carries for the device to index the bound buffer with.
    ///
    /// There is no generation check here: the index is valid while the key is alive, and the key owns
    /// the slot (releases it in `Drop`). Batched access is obliged to hold
    /// live keys anyway — it holds them as clones inside the message itself.
    pub fn raw_index(&self) -> usize {
        self.inner.key.value as usize
    }
}

pub struct WriteView<'a, T: Pod> {
    inner: RwLockWriteGuard<'a, RawMap>,
    world: Arc<World>,
    _marker: PhantomData<T>,
}

pub struct WriteGuard<'a> {
    _inner: RwLockWriteGuard<'a, RawMap>,
}

impl<'a, T: Pod + Send + Sync> WriteView<'a, T> {
    pub fn len(&self) -> usize {
        self.inner.payload_len()
    }

    pub fn add(&mut self, value: T) -> WorldKey<T> {
        WorldKey {
            inner: Arc::new(WorldKeyInner {
                world: self.world.clone(),
                key: self.inner.insert(value),
            }),
        }
    }
}

pub struct WorldBuilder {
    capacity: usize,
    inner: WorldInner,
}

impl WorldBuilder {
    /// Initial capacity, in elements, of every storage registered AFTER this
    /// call. Storage grows past it on demand, so this is a sizing hint rather
    /// than a limit; tests lower it to exercise growth without allocating
    /// `World::DEFAULT_CAPACITY` elements first.
    pub fn capacity(mut self, capacity: usize) -> Self {
        assert!(capacity > 0, "world storage capacity must be at least 1");
        self.capacity = capacity;
        self
    }

    pub fn with_storage<T: Pod + Send + Sync>(mut self) -> Self {
        match self.inner.map.entry(TypeId::of::<T>()) {
            Entry::Vacant(entry) => {
                let raw = RawMap::new::<T>(&self.inner.gpu, self.capacity);
                self.inner
                    .generations
                    .insert(TypeId::of::<T>(), raw.generation.clone());
                entry.insert(RwLock::new(raw));
            }
            Entry::Occupied(_) => {
                panic!(
                    "storage for type {} already added",
                    std::any::type_name::<T>()
                );
            }
        }
        self
    }

    pub fn usual<T: Pod + Sync + Send + Scalar>(self) -> World {
        self.with_storage::<T>()
            .with_storage::<[T; 3]>()
            .with_storage::<[[T; 3]; 3]>()
            .with_storage::<Vector3<T>>()
            .with_storage::<Wrench<T>>()
            .with_storage::<Motor<T>>()
            .with_storage::<Twist<T>>()
            // Per-connection accelerator outputs: the value pair [wrench on a, on b]
            // and the 24-column Jacobian block. See newton's `Accelerator`.
            .with_storage::<[Wrench<T>; 2]>()
            .with_storage::<[[Wrench<T>; 24]; 2]>()
            // One 6×6 block of the implicit solver's system matrix. The matrix is
            // block-structured by construction — row block = body, column block =
            // body — so the Newton–Schulz iteration addresses it a block at a time.
            .with_storage::<[[T; 6]; 6]>()
            // Accelerator incidence rows: the baked, index-only description of
            // one unit of work. Deliberately NOT parameterised by `T` — a row
            // holds slot indices, and the scalars it carries ride as `f32` bit
            // patterns. Width must match `newton::accelerator::row::ROW`, which
            // asserts it (aristotle cannot depend on newton).
            .with_storage::<[u32; 128]>()
            .build()
    }

    pub fn build(self) -> World {
        World { inner: self.inner }
    }
}

impl World {
    pub const DEFAULT_CAPACITY: usize = 256 * 1024;
    pub fn builder() -> WorldBuilder {
        let gpu = Arc::new(GpuAccelerator::new());
        WorldBuilder {
            capacity: Self::DEFAULT_CAPACITY,
            inner: WorldInner {
                gpu,
                map: HashMap::new(),
                generations: HashMap::new(),
            },
        }
    }

    pub fn gpu(&self) -> Arc<GpuAccelerator> {
        self.inner.gpu.clone()
    }

    pub fn read<'a, T: Pod>(&'a self) -> ReadView<'a, T> {
        // A plain borrow: the map of storages is fixed once `build` returns, so
        // the `RwLock` lives exactly as long as `self`.
        let lock: &'a RwLock<RawMap> =
            self.inner.map.get(&TypeId::of::<T>()).unwrap_or_else(|| {
                panic!(
                    "Cannot find a corresponding map in a world for {}",
                    std::any::type_name::<T>()
                )
            });
        ReadView {
            inner: lock.read().unwrap(),
            _marker: PhantomData,
        }
    }

    pub fn write<'a, T: Pod>(self: &'a Arc<Self>) -> WriteView<'a, T> {
        let lock: &'a RwLock<RawMap> =
            self.inner.map.get(&TypeId::of::<T>()).unwrap_or_else(|| {
                panic!(
                    "Cannot find a corresponding map in a world for {}",
                    std::any::type_name::<T>()
                )
            });
        WriteView {
            inner: lock.write().unwrap(),
            world: self.clone(),
            _marker: PhantomData,
        }
    }

    /// The current generation of the storage for type `t`, without taking its
    /// lock. It changes only when the storage is reallocated by growth, so a
    /// holder of a clone of the buffer (`ReadView::get_map`) taken at generation
    /// `g` holds the live buffer exactly while this still returns `g`.
    pub fn generation(&self, t: TypeId) -> u64 {
        self.inner
            .generations
            .get(&t)
            .unwrap_or_else(|| panic!("Cannot find a corresponding map in a world for {:?}", t))
            .load(Ordering::Acquire)
    }

    pub fn write_guard<'a>(self: &'a Arc<Self>, t: TypeId) -> WriteGuard<'a> {
        let raw =
            self.inner.map.get(&t).unwrap_or_else(|| {
                panic!("Cannot find a corresponding map in a world for {:?}", t)
            });
        WriteGuard {
            _inner: raw.write().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let w = Arc::new(World::builder().with_storage::<usize>().build());

        let mut guard = w.write::<usize>();
        let key = guard.add(42);
        let key2 = guard.add(27);
        drop(guard);
        assert_eq!(key.read(), 42);
        assert_eq!(key2.read(), 27);
    }

    #[test]
    fn view() {
        let w = Arc::new(World::builder().with_storage::<usize>().build());

        let mut guard = w.write::<usize>();
        let key = guard.add(42);
        let key2 = guard.add(27);
        drop(guard);

        {
            assert_eq!(key.read(), 42);
            assert_eq!(key2.read(), 27);
        }

        {
            key.write(11);
            key2.write(110);
            assert_eq!(key.read(), 11);
            assert_eq!(key2.read(), 110);
        }
    }

    fn accel() -> Arc<GpuAccelerator> {
        Arc::new(GpuAccelerator::new())
    }

    #[test]
    fn raw_map() {
        let mut map = RawMap::new::<u32>(&accel(), 100);
        let key = map.insert(42u32);
        let key2 = map.insert(27u32);
        assert_eq!(map.get(&key).unwrap(), 42);
        assert_eq!(map.get(&key2).unwrap(), 27);
        assert_eq!(map.get(&key).unwrap(), 42);

        map.set(&key, 117);
        assert_eq!(map.get(&key).unwrap(), 117);
        assert_eq!(map.remove(&key).unwrap(), 117);
        assert_eq!(map.get(&key2).unwrap(), 27);
        assert_eq!(map.remove(&key2).unwrap(), 27);
    }

    // A non-scalar Pod type: checks that erasure/downcast works not only
    // on primitives and that as_bytes yields the dense layout of the struct.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    #[bytemuck(crate = "rembrandt::bytemuck")]
    struct Vec3 {
        x: f32,
        y: f32,
        z: f32,
    }

    // Vec<T>s of different types in one World do not overlap: the HashMap keyed by TypeId routes
    // them to different RawMaps, and the downcast recovers the correct T.
    #[test]
    fn multiple_types_do_not_collide() {
        let w = Arc::new(
            World::builder()
                .with_storage::<u32>()
                .with_storage::<Vec3>()
                .build(),
        );

        let mut ints = w.write::<u32>();
        let mut vecs = w.write::<Vec3>();

        let a = ints.add(7);
        let b = ints.add(8);
        let v = vecs.add(Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        });
        drop(ints);
        drop(vecs);

        assert_eq!(a.read(), 7);
        assert_eq!(b.read(), 8);
        assert_eq!(
            v.read(),
            Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
    }

    // The downcast is now a real type check: a foreign T in a RawMap is caught in release
    // too (previously it was a debug_assert).
    #[test]
    #[should_panic(expected = "different type")]
    fn wrong_type_panics_on_downcast() {
        let mut map = RawMap::new::<u32>(&accel(), 100);
        // insert::<f32> into a map holding Vec<u32> requires Vec<u32> -> Vec<f32>.
        let _ = map.insert(1.0f32);
    }

    // Generation layer: a stale key cannot be read after remove, and the slot
    // is reused with the same position but a different generation.
    #[test]
    fn stale_key_returns_none_and_slot_is_reused() {
        let mut map = RawMap::new::<u32>(&accel(), 100);
        let key = map.insert(100u32);
        assert_eq!(map.remove(&key).unwrap(), 100);
        assert!(map.get(&key).is_none()); // key is Copy, but the slot is already free

        let key2 = map.insert(200u32);
        assert_eq!(key2.value, key.value); // the same physical slot
        assert_ne!(key2.generation, key.generation); // but a new generation
        assert!(map.get(&key).is_none()); // the old key is still stale
        assert_eq!(map.get(&key2).unwrap(), 200);
    }

    #[test]
    fn remove_twice_is_none() {
        let mut map = RawMap::new::<u32>(&accel(), 100);
        let key = map.insert(5u32);
        assert_eq!(map.remove(&key).unwrap(), 5);
        assert!(map.remove(&key).is_none());
    }

    // The free-list is kept sorted in ascending order: insert always hands out
    // the lowest free slot, regardless of the order of removes.
    #[test]
    fn free_list_reuses_lowest_slot_first() {
        let mut map = RawMap::new::<u32>(&accel(), 100);
        let k0 = map.insert(0u32);
        let k1 = map.insert(1u32);
        let k2 = map.insert(2u32);
        let k3 = map.insert(3u32);

        // free them out of order
        map.remove(&k1);
        map.remove(&k3);
        map.remove(&k0);

        // reuse proceeds in ascending position: 0, then 1, then 3
        assert_eq!(map.insert(10u32).value, k0.value);
        assert_eq!(map.insert(11u32).value, k1.value);
        assert_eq!(map.insert(13u32).value, k3.value);

        // the untouched slot 2 is still valid
        assert_eq!(map.get(&k2).unwrap(), 2);
        // no free slots left — a fresh insert extends past the high-water mark
        assert_eq!(map.insert(14u32).value, 4);
    }

    // Invariant map.len() == values.len(): the payload and the slot table grow strictly
    // in lockstep, and remove truncates neither of them (the hole remains as garbage).
    #[test]
    fn payload_tracks_slots_as_high_water_mark() {
        let mut map = RawMap::new::<u32>(&accel(), 100);
        let k0 = map.insert(11u32);
        map.insert(22u32);
        assert_eq!(map.payload_len(), 2);

        map.remove(&k0);
        assert_eq!(map.payload_len(), 2); // len is a high-water mark, it does not shrink

        map.insert(33u32); // reuses slot 0, does not grow
        assert_eq!(map.payload_len(), 2);
    }

    #[test]
    fn worldref_get_mut_persists_and_remove_invalidates() {
        let w = Arc::new(World::builder().with_storage::<u32>().build());
        let mut g = w.write::<u32>();
        let key = g.add(1);
        drop(g);

        key.write(99);
        assert_eq!(key.read(), 99);
    }

    // Different types mean different RwLock<RawMap>s: a write view of one and a read view of the other
    // are held simultaneously and do not deadlock.
    #[test]
    fn concurrent_views_of_distinct_types() {
        let w = Arc::new(
            World::builder()
                .with_storage::<u32>()
                .with_storage::<u64>()
                .build(),
        );
        let ka = w.write::<u32>().add(3);
        let kb = w.write::<u64>().add(9);

        ka.write(30);
        assert_eq!(ka.read(), 30);
        assert_eq!(kb.read(), 9);
    }

    // Inserting past the capacity grows the buffer: every slot keeps its index
    // and its value, the host mapping follows the new buffer, and the generation
    // moves exactly once per reallocation.
    #[test]
    fn raw_map_grows_and_keeps_indices() {
        let mut map = RawMap::new::<u32>(&accel(), 2);
        let keys: Vec<_> = (0..9u32).map(|i| map.insert(i * 10)).collect();
        assert!(map.capacity >= 9);
        // 2 -> 4 -> 8 -> 16
        assert_eq!(map.generation.load(Ordering::Acquire), 3);
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(k.value as usize, i);
            assert_eq!(map.get(k).unwrap(), i as u32 * 10);
        }
        // writes after growth land in the new buffer
        map.set(&keys[8], 7);
        assert_eq!(map.get(&keys[8]).unwrap(), 7);
        assert_eq!(map.values::<u32>().read().unwrap()[8], 7);
    }

    // Reusing a freed slot never grows: only a fresh slot past the end does.
    #[test]
    fn raw_map_reuse_does_not_grow() {
        let mut map = RawMap::new::<u32>(&accel(), 2);
        let k0 = map.insert(1u32);
        map.insert(2u32);
        map.remove(&k0);
        map.insert(3u32);
        assert_eq!(map.capacity, 2);
        assert_eq!(map.generation.load(Ordering::Acquire), 0);
    }

    // Negative case: a slot past the buffer's end is refused on the access path
    // itself, whichever of the two (mapped or guarded) this device takes.
    #[test]
    #[should_panic(expected = "past the buffer's capacity")]
    fn write_past_capacity_panics() {
        let map = RawMap::new::<u32>(&accel(), 4);
        map.write_at::<u32>(4, 1);
    }

    #[test]
    #[should_panic(expected = "past the buffer's capacity")]
    fn read_past_capacity_panics() {
        let map = RawMap::new::<u32>(&accel(), 4);
        let _ = map.read_at::<u32>(4);
    }
}
