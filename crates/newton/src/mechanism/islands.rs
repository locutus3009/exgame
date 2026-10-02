// SPDX-License-Identifier: MIT

use super::component::Component;
use super::island::Island;
use aristotle::WorldId;
use bytemuck::Pod;
use indexmap::IndexMap;
use joints::{Joint, JointEdge};
use peano::prelude::*;
use std::collections::{HashMap, HashSet};

/// The topological layer of the mechanism. Owns the body/joint graph, organized into
/// islands, and encapsulates the invariant "each `Island` is exactly one
/// component". Only invariant-preserving primitives are exposed; it knows nothing
/// about the integrator, fields or the step (topology ⊘ physics). All the boolean algebra
/// of mechanisms reduces to two private primitives — `union` (gluing when
/// an edge is added) and `repartition` (rebuilding when bodies/edges are removed).
pub(crate) struct Islands<T: Scalar + Pod, S: Scalar + From<T> + Into<T>>(Vec<Island<T, S>>);

impl<T: Scalar + StandardPart + Pod, S: Scalar + From<T> + Into<T>> Islands<T, S> {
    pub(crate) fn new() -> Self {
        Islands(Vec::new())
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Index of the island containing body `id` (linear scan — there are few islands,
    /// this is not a hot path).
    pub(crate) fn island_of(&self, id: WorldId) -> Option<usize> {
        self.0.iter().position(|isl| isl.contains_body(id))
    }

    pub(crate) fn get_body(&self, id: WorldId) -> Option<&dyn Component<T, S>> {
        self.0.iter().find_map(|isl| isl.get_body(id))
    }

    pub(crate) fn get_body_mut(
        &mut self,
        id: WorldId,
    ) -> Option<&mut (dyn Component<T, S> + 'static)> {
        self.0
            .iter_mut()
            .find_map(|isl| isl.get_body_mut(id))
            .map(|b| &mut **b)
    }

    pub(crate) fn all_bodies(&self) -> impl Iterator<Item = &dyn Component<T, S>> {
        self.0
            .iter()
            .flat_map(|isl| isl.bodies_iter())
            .map(|b| &**b)
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut Island<T, S>> {
        self.0.iter_mut()
    }

    #[allow(dead_code)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Island<T, S>> {
        self.0.iter()
    }

    pub(crate) fn origin_of(&self, idx: usize) -> Vector3<S> {
        self.0[idx].origin()
    }

    /// Per-island `(origin, total mass)` for the "centroid of centroids".
    pub(crate) fn island_anchors(&self) -> impl Iterator<Item = (Vector3<S>, T)> + '_ {
        self.0.iter().map(|isl| {
            let m = isl
                .bodies_iter()
                .fold(T::ZERO, |a, e| a + e.body().inertia.mass());
            (isl.origin(), m)
        })
    }

    /// Partition into islands as lists of world ids (for diagnostics and `split`).
    /// This is now just a storage read — no recomputation.
    pub(crate) fn components(&self) -> Vec<Vec<WorldId>> {
        self.0
            .iter()
            .map(|isl| isl.body_ids().copied().collect())
            .collect()
    }

    // ── Primitive 1: union (adding an edge glues islands) ────────────────────

    /// Ensure that `a` and `b` are in one island, merging two different ones. The edge itself
    /// is inserted by the CALLER after union. Panics on an unknown id. Connecting
    /// two components by an edge yields a component again: you cannot "forget to glue", an edge
    /// across a boundary must merge the islands, otherwise the structure is invalid on the spot.
    fn union(&mut self, a: WorldId, b: WorldId) {
        let ia = self.island_of(a).expect("union: unknown world id a");
        let ib = self.island_of(b).expect("union: unknown world id b");
        if ia == ib {
            return;
        }
        let (lo, hi) = (ia.min(ib), ia.max(ib));
        let hi_isl = self.0.remove(hi); // lo < hi → index lo does not shift
        self.0[lo].merge_from(hi_isl);
        self.0[lo].recompute_centroid(); // merged set of bodies → new COM
        self.0[lo].recompute_incidence(); // the joints of the two islands merged
    }

    // ── Primitive 2: repartition (removal may split an island) ───────────────

    /// Rebuild island `idx` into its connected components: replace it with the first one,
    /// append the rest. An emptied island is removed. The order of islands is not
    /// specified, so the cheap swap_remove is used.
    fn repartition(&mut self, idx: usize) {
        let isl = self.0.swap_remove(idx);
        if isl.is_empty() {
            return;
        }
        let (comps, comp_of) = connected_components(isl.bodies(), isl.joints());
        self.push_partitioned(isl, &comps, &comp_of);
    }

    /// Distribute the bodies/joints of `isl` into `comps` islands and append them at the end. With
    /// a single component — return the island as is (no reshuffling).
    fn push_partitioned(
        &mut self,
        isl: Island<T, S>,
        comps: &[Vec<WorldId>],
        comp_of: &HashMap<WorldId, usize>,
    ) {
        if comps.len() <= 1 {
            // The set of bodies/joints may have changed (removal/extract) → re-anchor and
            // rebuild the incidence.
            let mut isl = isl;
            isl.recompute_centroid();
            isl.recompute_incidence();
            self.0.push(isl);
            return;
        }
        // Fragments inherit the parent's origin: body poses are given relative to it,
        // and must be interpreted correctly until each fragment is re-centred.
        let parent_origin = isl.origin();
        let (bodies, joints) = isl.into_parts();
        let mut built: Vec<Island<T, S>> = comps
            .iter()
            .map(|_| Island::empty_at(parent_origin))
            .collect();
        for (id, b) in bodies {
            built[comp_of[&id]].insert_body(id, b);
        }
        for (jid, e) in joints {
            built[comp_of[&e.a()]].insert_joint(jid, e);
        }
        for isl in &mut built {
            isl.recompute_centroid(); // each fragment → its own COM
            isl.recompute_incidence(); // and its own set of joints → its own gather contributions
        }
        self.0.extend(built);
    }

    // ── Public operations (everything via the two primitives) ────────────────

    /// Add a body as a fresh singleton island. Panics on a duplicate id.
    pub(crate) fn add_body(&mut self, entity: Box<dyn Component<T, S>>) -> WorldId {
        let id = entity.id();
        assert!(self.island_of(id).is_none(), "duplicate world id {id:?}");
        self.0.push(Island::from_body(entity));
        id
    }

    /// Insert joints `a`—`b`, merging islands as needed. A self-loop or
    /// an unknown id panics.
    pub(crate) fn connect(&mut self, a: WorldId, links: Vec<(Joint<T>, WorldId)>) {
        for (joint, b) in links {
            assert!(a != b, "self-loop forbidden (world id {a:?})");
            self.union(a, b);
            let edge = JointEdge::new(a, b, joint);
            let i = self
                .island_of(a)
                .expect("connect: body a present after union");
            self.0[i].insert_joint(edge.id(), edge);
            self.0[i].recompute_incidence(); // new edge → new gather contribution for a and b
        }
    }

    /// Take out body `id` and its incident joints; rebuild its island. `None`
    /// if the id is unknown (a lookup miss, not an invariant violation).
    pub(crate) fn detach(&mut self, id: WorldId) -> Option<Box<dyn Component<T, S>>> {
        let i = self.island_of(id)?;
        let isl = &mut self.0[i];
        isl.retain_joints_incident(id);
        let out = isl.remove_body(id);
        self.repartition(i);
        out
    }

    /// Extract bodies `ids` (an arbitrary subset) into a NEW `Islands`.
    /// Internal edges (both ends in the set) travel with the bodies; boundary ones are cut;
    /// external ones stay. The affected islands of `self` are rebuilt. The new
    /// `Islands` is built from singletons + connect of the internal edges — `union` itself
    /// assembles its components, there is no separate connectivity code.
    pub(crate) fn extract(&mut self, ids: &[WorldId]) -> Islands<T, S> {
        let set: HashSet<WorldId> = ids.iter().copied().collect();
        let mut extracted: Vec<Box<dyn Component<T, S>>> = Vec::new();
        let mut internal: Vec<(WorldId, WorldId, Joint<T>)> = Vec::new();

        // Take all islands; what remains is returned to self already rebuilt.
        for mut isl in std::mem::take(&mut self.0) {
            let mut kept: IndexMap<WorldId, JointEdge<T>> = IndexMap::new();
            for (jid, e) in isl.drain_joints() {
                match (set.contains(&e.a()), set.contains(&e.b())) {
                    (true, true) => {
                        let (a, b, j) = e.split();
                        internal.push((a, b, j));
                    }
                    (true, false) | (false, true) => { /* boundary → cut */ }
                    (false, false) => {
                        kept.insert(jid, e);
                    }
                }
            }
            isl.set_joints(kept);

            let here: Vec<WorldId> = isl
                .body_ids()
                .copied()
                .filter(|k| set.contains(k))
                .collect();
            for k in here {
                extracted.push(isl.remove_body(k).expect("key came from keys()"));
            }

            if !isl.is_empty() {
                let (comps, comp_of) = connected_components(isl.bodies(), isl.joints());
                self.push_partitioned(isl, &comps, &comp_of);
            }
        }

        let mut out = Islands::new();
        for b in extracted {
            out.add_body(b);
        }
        for (a, b, j) in internal {
            out.connect(a, vec![(j, b)]);
        }
        out
    }

    /// Merge in the islands of `other` as they are. There can be no id collisions — `WorldId`
    /// is unique by construction. The linking edge is inserted by the caller (`connect`).
    pub(crate) fn absorb(&mut self, mut other: Islands<T, S>) {
        self.0.append(&mut other.0);
    }
}

/// Connected components of the `bodies`+`joints` graph by BFS: a list of world ids
/// per component and the inverse map body→component index. O(V+E),
/// the temporary adjacency is built on the spot (the crate keeps no adjacency lists —
/// the operation is rare, not a hot path).
fn connected_components<T: Scalar + StandardPart + Pod, S: Ring>(
    bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
    joints: &IndexMap<WorldId, JointEdge<T>>,
) -> (Vec<Vec<WorldId>>, HashMap<WorldId, usize>) {
    let mut adj: IndexMap<WorldId, Vec<WorldId>> = IndexMap::new();
    for k in bodies.keys() {
        adj.insert(*k, Vec::new());
    }
    for e in joints.values() {
        adj.get_mut(&e.a()).unwrap().push(e.b());
        adj.get_mut(&e.b()).unwrap().push(e.a());
    }

    let mut visited: HashSet<WorldId> = HashSet::new();
    let mut comps: Vec<Vec<WorldId>> = Vec::new();
    let mut comp_of: HashMap<WorldId, usize> = HashMap::new();
    for start in bodies.keys() {
        if visited.contains(start) {
            continue;
        }
        let ci = comps.len();
        let mut comp = vec![*start];
        let mut stack = vec![*start];
        visited.insert(*start);
        comp_of.insert(*start, ci);
        while let Some(k) = stack.pop() {
            for n in adj.get(&k).unwrap() {
                if visited.insert(*n) {
                    comp.push(*n);
                    comp_of.insert(*n, ci);
                    stack.push(*n);
                }
            }
        }
        comps.push(comp);
    }
    (comps, comp_of)
}
