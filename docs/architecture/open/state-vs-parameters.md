# State vs parameters discipline — OPEN

Only integrated quantities (position, velocity, the driving
acceleration) belong in the shared scalar state. Spacecraft mass,
`Isp`, `Δv`, propellant, thrust curves — and **GM/`grav` and the
`influence` cutoff radius** — stay *local* per-body parameters with
their own representation freedom, entering the state only through the
acceleration they produce. A spacecraft is a gravitational test
particle; its mass never enters the force sum.

This is still load-bearing, now exercised by the current code
(`newton::Mechanism` keeps per-body parameters separate from the
integrator-visible state value; the gravity propagator computes
charges from parameters and feeds wrenches into the
integrator-visible state).

## Cross-references

- [determinism-boundary](./determinism-boundary.md) — corollary: only
  what is integrated is deterministic state; parameters and outputs
  are not.
- [fixed-point-vs-f64](./fixed-point-vs-f64.md) — when revisited, the
  parameter side has independent representation freedom (it doesn't
  need to share the integrated-state scalar type).
- [crates/newton](../crates/newton.md) — `ForceField` /
  `GravityPropagator` honour this split.

*Original CLAUDE.md deliberation: 5.*
