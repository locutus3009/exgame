// SPDX-License-Identifier: MIT

use aristotle::{World, WorldKey};
use bytemuck::Pod;
use clifford::pga3::{Twist, Wrench};
use peano::prelude::*;
use std::sync::Arc;

// --- Small 3×3 linear algebra for the off-diagonal tensor ---

#[inline]
fn matvec3<T>(a: [[T; 3]; 3], v: [T; 3]) -> [T; 3]
where
    T: Scalar,
{
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

/// Solves `A·x = b` for 3×3 via adjugate/determinant. Assumes a
/// non-singular (for an inertia tensor — SPD) matrix.
#[inline]
fn solve3<T>(a: [[T; 3]; 3], b: [T; 3]) -> [T; 3]
where
    T: Scalar,
{
    // Cofactors (C_ij)
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let c10 = a[0][2] * a[2][1] - a[0][1] * a[2][2];
    let c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let c12 = a[0][1] * a[2][0] - a[0][0] * a[2][1];
    let c20 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    let c21 = a[0][2] * a[1][0] - a[0][0] * a[1][2];
    let c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];

    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;

    // x = adj(A)·b / det,  where adj = Cᵀ, i.e. x_i = (Σ_j C_ji · b_j) / det
    [
        (c00 * b[0] + c10 * b[1] + c20 * b[2]) / det,
        (c01 * b[0] + c11 * b[1] + c21 * b[2]) / det,
        (c02 * b[0] + c12 * b[1] + c22 * b[2]) / det,
    ]
}

// ============================================================================
// INERTIA — the velocity → momentum map  (Twist → Wrench),  P = I·V
// ============================================================================
//
// Physically, inertia maps a twist (velocity) to a wrench (momentum as a
// covector). Since Twist and Wrench are stored in the SAME basis (J is deferred to
// `power`), here it is simply a scaling of the linear and angular parts —
// no blades and no dual. The duality is checked by the energy test below.

/// Representation of rotational inertia. Public because it is a field of
/// the `Inertia::Rigid` variant; but consumers build inertia through the constructors
/// (`diagonal`/`isotropic`/`from_tensor`) rather than assembling the variant by hand.
#[derive(Debug)]
pub enum InertiaTensor<T: Pod + Send + Sync> {
    /// Principal axes: [Ixx, Iyy, Izz].
    Diagonal(Arc<World>, WorldKey<[T; 3]>),
    /// General symmetric 3×3 inertia tensor (arbitrarily oriented body).
    Full(Arc<World>, WorldKey<[[T; 3]; 3]>),
}

impl<T: Pod + Send + Sync> Clone for InertiaTensor<T> {
    fn clone(&self) -> Self {
        match self {
            InertiaTensor::Diagonal(world, key) => {
                let value = key.read();
                let mut map = world.write();
                let new_key = map.add(value);
                InertiaTensor::Diagonal(world.clone(), new_key)
            }
            InertiaTensor::Full(world, key) => {
                let value = key.read();
                let mut map = world.write();
                let new_key = map.add(value);
                InertiaTensor::Full(world.clone(), new_key)
            }
        }
    }
}

/// Lightweight handle to the inertia slots: the same `WorldKey`s, WITHOUT allocating new ones.
///
/// Needed because `Inertia::clone` deliberately copies VALUES into fresh slots
/// (a copied body must have its own inertia), while an accelerator task needs exactly
/// the opposite — to refer to the same slots and travel in a message that must
/// be `'static + Send`. Cloning a `WorldKey` is an `Arc` bump, so the handle
/// is copied for free.
#[derive(Debug, Clone)]
pub enum InertiaKeys<T: Pod + Send + Sync> {
    Rigid {
        mass: WorldKey<T>,
        angular: AngularKeys<T>,
    },
    Kinematic,
}

/// Rotational part of [`InertiaKeys`].
#[derive(Debug, Clone)]
pub enum AngularKeys<T: Pod + Send + Sync> {
    Diagonal(WorldKey<[T; 3]>),
    Full(WorldKey<[[T; 3]; 3]>),
}

impl<T: Scalar + Pod> InertiaKeys<T> {
    /// Same as `Inertia::apply`: momentum from velocity, in BODY coordinates.
    pub fn apply(&self, twist: &Twist<T>) -> Wrench<T> {
        match self {
            Self::Rigid { mass, angular } => {
                let v = twist.linear();
                let w = twist.angular();
                let m = mass.read();
                match angular {
                    AngularKeys::Diagonal(key) => {
                        let [ix, iy, iz] = key.read();
                        Wrench::new(
                            &v.scale(m),
                            &Vector3::from([ix * w[0], iy * w[1], iz * w[2]]),
                        )
                    }
                    AngularKeys::Full(tensor) => {
                        let l = matvec3(tensor.read(), w.into());
                        Wrench::new(&v.scale(m), &Vector3::from(l))
                    }
                }
            }
            Self::Kinematic => Wrench::new(&twist.linear(), &twist.angular()),
        }
    }
}

/// Rigid-body inertia: isotropic mass + rotational tensor (`Rigid`),
/// or `Kinematic` — a prescribed body: not a solved-for unknown,
/// its velocity is stored in `momentum` identically ("I am attached here"). The centre of mass
/// is at the body origin; an offset CoM is a future extension.
#[derive(Debug)]
pub enum Inertia<T: Pod + Send + Sync> {
    Rigid {
        world: Arc<World>,
        mass: WorldKey<T>,
        angular: InertiaTensor<T>,
    },
    Kinematic,
}

impl<T: Pod + Send + Sync> Clone for Inertia<T> {
    fn clone(&self) -> Self {
        match self {
            Inertia::Rigid {
                world,
                mass,
                angular,
            } => {
                let value = mass.read();
                let mut map = world.write();
                let new_key = map.add(value);
                Inertia::Rigid {
                    world: world.clone(),
                    mass: new_key,
                    angular: angular.clone(),
                }
            }
            Inertia::Kinematic => Inertia::Kinematic,
        }
    }
}

impl<T> Inertia<T>
where
    T: Copy + Pod + Send + Sync,
{
    /// Diagonal tensor: mass (kg) + principal moments [Ixx, Iyy, Izz] (kg·m²).
    #[inline]
    pub fn diagonal(world: Arc<World>, mass: T, principal_moments: [T; 3]) -> Self {
        let binding = world.clone();
        let mut map_plain = binding.write();
        let mut map_moments = binding.write();
        Self::Rigid {
            world: world.clone(),
            mass: map_plain.add(mass),
            angular: InertiaTensor::Diagonal(world, map_moments.add(principal_moments)),
        }
    }

    /// Isotropic body: mass + a single moment about all axes.
    #[inline]
    pub fn isotropic(world: Arc<World>, mass: T, moment: T) -> Self {
        Self::diagonal(world, mass, [moment, moment, moment])
    }

    /// General inertia tensor (arbitrarily oriented body). Expects a
    /// symmetric positive-definite matrix; checking it is a TODO.
    #[inline]
    pub fn from_tensor(world: Arc<World>, mass: T, tensor: [[T; 3]; 3]) -> Self {
        let binding = world.clone();
        let mut map_plain = binding.write();
        let mut map_tensor = binding.write();
        Self::Rigid {
            world: world.clone(),
            mass: map_plain.add(mass),
            angular: InertiaTensor::Full(world, map_tensor.add(tensor)),
        }
    }

    /// Whether the body is prescribed (velocity is prescribed, excluded from the solver's unknowns).
    #[inline]
    pub fn is_kinematic(&self) -> bool {
        matches!(self, Self::Kinematic)
    }

    /// Principal moments [Ixx, Iyy, Izz]. Meaningless for `Kinematic` (zeros).
    #[inline]
    pub fn principal_moments(&self) -> [T; 3]
    where
        T: Scalar,
    {
        match self {
            Self::Rigid {
                angular: InertiaTensor::Diagonal(_, m),
                ..
            } => m.read(),
            // The principal moments of an off-diagonal tensor are its eigenvalues;
            // they require diagonalization, not just the diagonal.
            Self::Rigid {
                angular: InertiaTensor::Full(_, _),
                ..
            } => todo!("principal moments of a Full tensor (eigen)"),
            Self::Kinematic => [T::ZERO; 3],
        }
    }
}

impl<T> Inertia<T>
where
    T: Scalar + Pod,
{
    /// Body mass. Zero for `Kinematic` (the body does not gravitate and has no mass scale).
    #[inline]
    pub fn mass(&self) -> T {
        match self {
            Self::Rigid { mass, .. } => mass.read(),
            Self::Kinematic => T::ZERO,
        }
    }

    /// Handle to ITS OWN slots — without copying values (unlike `clone`).
    pub fn keys(&self) -> InertiaKeys<T> {
        match self {
            Self::Rigid { mass, angular, .. } => InertiaKeys::Rigid {
                mass: mass.clone(),
                angular: match angular {
                    InertiaTensor::Diagonal(_, k) => AngularKeys::Diagonal(k.clone()),
                    InertiaTensor::Full(_, k) => AngularKeys::Full(k.clone()),
                },
            },
            Self::Kinematic => InertiaKeys::Kinematic,
        }
    }

    /// Momentum from velocity: `P = I·V`. Linear momentum `m·v`, angular `I_ω·ω`.
    /// For `Kinematic` — identity (velocity is put into momentum 1:1).
    #[inline]
    pub fn apply(&self, twist: &Twist<T>) -> Wrench<T> {
        match self {
            Self::Rigid { mass, angular, .. } => {
                let v = twist.linear();
                let w = twist.angular();
                let m = mass.read();
                match angular {
                    InertiaTensor::Diagonal(_, key) => {
                        let [ix, iy, iz] = key.read();
                        Wrench::new(
                            &v.scale(m),
                            &Vector3::from([ix * w[0], iy * w[1], iz * w[2]]),
                        )
                    }
                    // L = I·ω (3×3 tensor times vector); the linear part is the same.
                    InertiaTensor::Full(_, tensor) => {
                        let l = matvec3(tensor.read(), w.into());
                        Wrench::new(&v.scale(m), &Vector3::from(l))
                    }
                }
            }
            Self::Kinematic => Wrench::new(&twist.linear(), &twist.angular()),
        }
    }

    /// Velocity from momentum: `V = I⁻¹·P`. Assumes non-singular inertia
    /// (mass and moments ≠ 0) — for a physical body this is a constructor invariant.
    /// For `Kinematic` — identity, the inverse of `apply`.
    #[inline]
    pub fn apply_inverse(&self, momentum: &Wrench<T>) -> Twist<T> {
        match self {
            Self::Rigid { mass, angular, .. } => {
                let f = momentum.force();
                let t = momentum.torque();
                let m = mass.read();
                match angular {
                    InertiaTensor::Diagonal(_, key) => {
                        let [ix, iy, iz] = key.read();
                        Twist::new(
                            &f.scale(T::ONE / m),
                            &Vector3::from([t[0] / ix, t[1] / iy, t[2] / iz]),
                        )
                    }
                    // ω = I⁻¹·L — solving a 3×3 system (not component-wise division).
                    InertiaTensor::Full(_, tensor) => {
                        let omega = solve3(tensor.read(), t.into());
                        Twist::new(&f.scale(T::ONE / m), &Vector3::from(omega))
                    }
                }
            }
            Self::Kinematic => Twist::new(&momentum.force(), &momentum.torque()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn twist_approx(a: &Twist<f32>, b: &Twist<f32>, eps: f32) -> bool {
        let (al, bl) = (a.linear(), b.linear());
        let (aa, ba) = (a.angular(), b.angular());
        let mut i = 0;
        while i < 3 {
            if (al[i] - bl[i]).abs() > eps || (aa[i] - ba[i]).abs() > eps {
                return false;
            }
            i += 1;
        }
        true
    }

    #[test]
    fn momentum_scales_components() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 2.0f32, [3.0, 4.0, 5.0]);
        let v = Twist::new(
            &Vector3::from([1.0f32, 2.0, 3.0]),
            &Vector3::from([10.0, 20.0, 30.0]),
        );
        let p = inertia.apply(&v);
        assert_eq!(p.force().split(), [2.0, 4.0, 6.0]); // m·v
        assert_eq!(p.torque().split(), [30.0, 80.0, 150.0]); // I·ω
    }

    #[test]
    fn inverse_recovers_twist() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 2.0, [3.0, 4.0, 5.0]);
        let v = Twist::new(
            &Vector3::from([1.0, -2.0, 3.0]),
            &Vector3::from([0.5, -1.5, 2.5]),
        );
        let recovered = inertia.apply_inverse(&inertia.apply(&v));
        assert!(twist_approx(&recovered, &v, 1e-12));
    }

    #[test]
    fn kinematic_is_flagged_and_rigid_is_not() {
        let world = Arc::new(World::builder().usual::<f32>());

        let k = Inertia::<f32>::Kinematic;
        let r = Inertia::isotropic(world.clone(), 2.0f32, 1.0);
        assert!(k.is_kinematic());
        assert!(!r.is_kinematic());
    }

    #[test]
    fn kinematic_apply_is_identity_roundtrip() {
        // A kinematic inertia stores velocity in the momentum slot 1:1, so
        // apply then apply_inverse must return the original twist unchanged.
        let k = Inertia::<f32>::Kinematic;
        let v = Twist::new(
            &Vector3::from([1.0, -2.0, 3.0]),
            &Vector3::from([0.5, -1.5, 2.5]),
        );
        let recovered = k.apply_inverse(&k.apply(&v));
        assert!(twist_approx(&recovered, &v, 1e-15));
    }

    #[test]
    fn kinematic_mass_is_zero() {
        assert_eq!(Inertia::<f32>::Kinematic.mass(), 0.0);
    }

    /// THE PAYOFF FOR HONEST DUALITY: kinetic energy E = ½⟨P, V⟩, where the
    /// pairing is via `power` (the dual J). The linear part m|v|² is NOT lost on the
    /// degenerate axis — a naive metric product would give only the
    /// rotational energy. This is the basis for future energy-conservation tests.
    #[test]
    fn kinetic_energy_includes_linear_and_angular() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 2.0, [3.0, 4.0, 5.0]);
        let v = Twist::new(
            &Vector3::from([1.0, 2.0, 3.0]),
            &Vector3::from([0.5, 1.5, 2.5]),
        );

        let momentum = inertia.apply(&v);
        let ke = 0.5 * momentum.power(&v);

        // m|v|² = 2·14 = 28 ; ω·I·ω = 3·0.25 + 4·2.25 + 5·6.25 = 41 ; E = ½·69
        assert!(approx(ke, 34.5, 1e-9), "E = {ke}, expected 34.5");

        // Control: if the linear part were lost, only ½·41 = 20.5 would remain.
        assert!(ke > 30.0, "linear energy must not vanish");
    }

    // --- Off-diagonal (Full) tensor ---

    // Symmetric SPD tensor: leading minors 4 > 0, 11 > 0, det = 51 > 0.
    const FULL: [[f32; 3]; 3] = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]];

    #[test]
    fn full_inverse_recovers_twist() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::from_tensor(world.clone(), 2.0, FULL);
        let v = Twist::new(
            &Vector3::from([1.0, -2.0, 3.0]),
            &Vector3::from([0.5, -1.5, 2.5]),
        );
        let recovered = inertia.apply_inverse(&inertia.apply(&v));
        assert!(twist_approx(&recovered, &v, 1e-9));
    }

    /// Energy for an off-diagonal tensor: E = ½(m|v|² + ωᵀ·I·ω). Couples
    /// `power` (the dual) with the tensor matvec. An asymmetric inverse would break the test.
    #[test]
    fn full_kinetic_energy() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::from_tensor(world.clone(), 2.0, FULL);
        let v = Twist::new(
            &Vector3::from([1.0, 2.0, 3.0]),
            &Vector3::from([0.5, 1.5, 2.5]),
        );

        let ke = 0.5 * inertia.apply(&v).power(&v);

        // m|v|² = 2·14 = 28.
        // I·ω = [3.5, 7.5, 14.0];  ωᵀ·I·ω = 0.5·3.5 + 1.5·7.5 + 2.5·14 = 48.
        // E = ½·(28 + 48) = 38.
        assert!(approx(ke, 38.0, 1e-9), "E = {ke}, expected 38.0");
    }
}
