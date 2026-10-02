// SPDX-License-Identifier: MIT

//! Hanging cloth grids: NUM_CURTAINS uniform COLS×ROWS mass-spring curtains
//! (each COLS = N wide, ROWS = 2N tall, square cells), each an independent
//! Mechanism sharing one Accelerator and stepped together via join_all. Pinned
//! along a kinematic top row, hanging under a uniform gravity field, kicked by a
//! one-off +Y "wind gust". Newton implicit integrator on the Fix+f32 substrate.
//! Plain orbit camera (drag = orbit, scroll = zoom). Run:
//!   cargo run -p newton --release --example cloth_grid

use aristotle::{Epoch, World, WorldId};
use async_trait::async_trait;
use clifford::pga3::Twist;
use futures::future::join_all;
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use melies::winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use melies::{
    CircleInstance, CircleRenderer, Config, Example, Frame, Gpu, LineInstance, LineRenderer,
};
use newton::{
    Accelerator, Inert, Inertia, Mechanism, RigidBody,
    field::UniformField,
    integrator::{ImplicitIntegrator, Newton},
};
use peano::fixed::{Fix, types::I96F32};
use peano::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

type Fx = Fix<I96F32>;

// ── Tunable knobs ───────────────────────────────────────────────────────────
const N: usize = 4; // curtain WIDTH in nodes (columns); the grid is COLS × ROWS
const COLS: usize = N; // columns (width)
const ROWS: usize = 2 * N; // rows (height) — twice the width; cells stay square (uniform h)
const G: f32 = 9.81; // gravity magnitude (applied along world −Z)
const WIND: f32 = 1.0; // steady wind accel ⟂ to the curtain plane (world +Y); < G
const K: f32 = 100.0; // spring stiffness (N/m)
const ZETA: f32 = 0.7; // spring damping ratio (1.0 = critical); C is derived from it in `init`
const MASS_TOTAL: f32 = 1.0; // total cloth mass (kg); per dynamic body = MASS_TOTAL / (COLS·ROWS)
const KICK_COUNT: usize = 5; // how many dynamic nodes get an initial velocity kick
const KICK_SPEED: f32 = 0.5; // magnitude of each kick velocity (m/s)
const SEED: u64 = 0x5EED_1234_ABCD_0001;
const MAX_DT: f32 = 0.10; // fixed max sim step per rendered frame

// ── Scene layout ─────────────────────────────────────────────────────────────
const NUM_CURTAINS: usize = 2; // each its own Mechanism, stepped together via join_all
const WIDTH: f32 = 0.5; // physical width along X; uniform cell size h = WIDTH/(COLS-1)
const GAP: f32 = 1.0; // clear distance between neighbouring curtains along X

// ── Deterministic PRNG (splitmix64) — reproducible, no crate dependency ──────
#[derive(Clone)]
struct Rng(Arc<tokio::sync::Mutex<u64>>);
impl Rng {
    async fn next_u64(&self) -> u64 {
        let mut guard = self.0.lock().await;
        *guard = guard.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *guard;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    async fn unit01(&self) -> f32 {
        (self.next_u64().await >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in [-1, 1).
    async fn signed(&self) -> f32 {
        self.unit01().await * 2.0 - 1.0
    }
    /// Uniform integer in [0, n).
    async fn below(&self, n: usize) -> usize {
        (self.next_u64().await % n as u64) as usize
    }
}

// ── Orbit camera (throwaway example glue, adapted from hitchcock/simple.rs) ───
struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    dist: f32,
    focal: f32,
    center: Vector3<f32>,
}

impl OrbitCamera {
    fn new() -> Self {
        // Fit the row of curtains: X span across them, Z span = grid height.
        let h = WIDTH / (COLS as f32 - 1.0);
        let height = (ROWS as f32 - 1.0) * h;
        let ext_x = (NUM_CURTAINS as f32 - 1.0) * (WIDTH + GAP) + WIDTH;
        let fit = if ext_x > height { ext_x } else { height };
        Self {
            yaw: 0.4,    // slight 3/4 azimuth (face-on to the XZ curtains is yaw = 0)
            pitch: 0.25, // a touch above the horizon
            dist: fit * 1.6,
            focal: 1.0 / (30.0_f32.to_radians().tan()), // ~60° vertical FOV
            center: Vector3::from([ext_x * 0.5, 0.0, height * 0.45]), // scene centre (sags in Z)
        }
    }
    fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        let lim = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = (self.pitch + dpitch).clamp(-lim, lim);
    }
    fn zoom(&mut self, factor: f32) {
        self.dist = (self.dist * factor).clamp(0.3, 20.0);
    }
    fn eye(&self) -> Vector3<f32> {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        // Z-up orbit: azimuth `yaw` sweeps the XY plane, elevation `pitch` lifts
        // toward +Z. (World up is Z here — the curtain hangs along Z.)
        let off = Vector3::from([self.dist * cp * sy, self.dist * cp * cy, self.dist * sp]);
        self.center + off
    }
    /// Freeze the current orbit into a per-frame projector.
    fn projector(&self) -> Projector {
        let eye = self.eye();
        let fwd = normalize(self.center - eye);
        // World up is +Z (the curtain's vertical) → screen-up tracks Z.
        let right = normalize(fwd.cross(Vector3::from([0.0, 0.0, 1.0])));
        let up = right.cross(fwd);
        Projector {
            eye,
            right,
            up,
            fwd,
            focal: self.focal,
        }
    }
}

struct Projector {
    eye: Vector3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
    fwd: Vector3<f32>,
    focal: f32,
}
impl Projector {
    /// World point → (NDC centre, depth-in-front). `None` if behind / non-finite.
    fn project_point(&self, p: Vector3<f32>) -> Option<(Vector2<f32>, f32)> {
        let rel = p - self.eye;
        let cz = rel.dot(self.fwd);
        pinhole(
            Vector2::from([rel.dot(self.right), rel.dot(self.up)]),
            cz,
            self.focal,
        )
    }
}

fn pinhole(ru: Vector2<f32>, cz: f32, focal: f32) -> Option<(Vector2<f32>, f32)> {
    // Spelled through `partial_cmp` rather than `cz <= 0.02`: a NaN depth is
    // unordered against 0.02 and must be rejected, not carried into the sort.
    if !matches!(cz.partial_cmp(&0.02), Some(Ordering::Greater)) {
        return None;
    }
    let center = ru.scale(1.0 / cz).scale(focal);
    (center[0].is_finite() && center[1].is_finite() && cz.is_finite()).then_some((center, cz))
}

fn normalize(v: Vector3<f32>) -> Vector3<f32> {
    let n = v.dot(v).sqrt();
    if n > 1e-9 { v.scale(1.0 / n) } else { v }
}

// ── Demo state ───────────────────────────────────────────────────────────────
/// One curtain = its own `Mechanism` plus the id bookkeeping to draw it.
struct Curtain {
    mech: Mechanism<f32, Fx>,
    nodes: Vec<(WorldId, bool)>, // (id, is_kinematic) — for drawing/colour
    springs: Vec<(WorldId, WorldId)>, // endpoint id pairs — for drawing
}

struct Cloth {
    max_dt: f32,
    last: Instant,
    curtains: Vec<Curtain>,
    circles: CircleRenderer,
    lines: LineRenderer,
    orbit: OrbitCamera,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
}

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

/// Build one curtain as an independent mechanism sharing `accel`. A uniform
/// COLS × ROWS grid (square cells, spacing `h`) in the XZ plane, left edge at
/// `x_offset`; the top row (z = (ROWS−1)·h) is kinematic. A few random dynamic
/// nodes get a "wind gust" kick — all velocities fall in the +Y half-space, as
/// if one puff blew, then stopped. `rng` is threaded so successive curtains draw
/// distinct kicks.
async fn build_curtain(
    world: &Arc<World>,
    accel: &Arc<Accelerator<f32>>,
    x_offset: f32,
    h: f32,
    m: f32,
    c: f32,
    rng: &Rng,
) -> Curtain {
    let mech: Mechanism<f32, Fx> =
        Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel.clone())));
    //    let mech: Mechanism<f32, Fx> = Mechanism::new(ImplicitIntegrator::LieEuler(accel.clone()));

    // Pick KICK_COUNT distinct dynamic nodes (rows 0..ROWS-1; top row is kinematic).
    let dynamic_count = COLS * (ROWS - 1);
    let mut kicked: Vec<usize> = Vec::new();
    while kicked.len() < KICK_COUNT.min(dynamic_count) {
        let lin = rng.below(dynamic_count).await;
        if !kicked.contains(&lin) {
            kicked.push(lin);
        }
    }

    // Nodes. ids[row*COLS + col] = WorldId of that node.
    let mut ids: Vec<WorldId> = Vec::with_capacity(COLS * ROWS);
    for row in 0..ROWS {
        for col in 0..COLS {
            let pos = Vector3::from([x_offset + col as f32 * h, 0.0, row as f32 * h]);
            let kinematic = row == ROWS - 1; // top row is pinned
            let id = if kinematic {
                let b = RigidBody::new(world.clone(), Inertia::Kinematic);
                b.pose.write(Twist::new(&pos, &Vector3::ZERO).exp(1.0));
                mech.add_body(Inert::new(b)).await
            } else {
                let lin = row * COLS + col;
                let vel = if kicked.contains(&lin) {
                    // Wind gust: random spread but Y folded into the +Y half-space
                    // so every kick blows the same way.
                    let dir = Vector3::from([
                        rng.signed().await,
                        rng.signed().await.abs(),
                        rng.signed().await,
                    ]);
                    let n = dir.dot(dir).sqrt();
                    if n > 1e-6 {
                        dir.scale(KICK_SPEED / n)
                    } else {
                        Vector3::from([0.0, KICK_SPEED, 0.0])
                    }
                } else {
                    Vector3::ZERO
                };
                mech.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &pos,
                    &vel,
                    m,
                )))
                .await
            };
            ids.push(id);
        }
    }

    // 4-neighbour structural springs (uniform rest h): right (col+1) and up (row+1).
    let mut springs: Vec<(WorldId, WorldId)> = Vec::new();
    for row in 0..ROWS {
        for col in 0..COLS {
            let a = ids[row * COLS + col];
            if col + 1 < COLS {
                let b = ids[row * COLS + col + 1];
                mech.connect(a, vec![(spring(world.clone(), h, K, c), b)])
                    .await;
                springs.push((a, b));
            }
            if row + 1 < ROWS {
                let b = ids[(row + 1) * COLS + col];
                mech.connect(a, vec![(spring(world.clone(), h, K, c), b)])
                    .await;
                springs.push((a, b));
            }
        }
    }

    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, 0.0, -G]))))
        .await;
    // Steady breeze ⟂ to the XZ plane (world +Y), gentler than gravity.
    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, WIND, 0.0]))))
        .await;

    let nodes: Vec<(WorldId, bool)> = ids
        .iter()
        .enumerate()
        .map(|(lin, id)| (*id, lin / COLS == ROWS - 1))
        .collect();

    Curtain {
        mech,
        nodes,
        springs,
    }
}

impl Cloth {
    pub fn tick(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(self.max_dt);
        self.last = now;
        dt
    }
}

#[async_trait]
impl Example for Cloth {
    async fn init(gpu: &Gpu) -> Self {
        // ONE accelerator shared by every curtain — its worker batches the kernels
        // of all mechanisms stepped concurrently (that's where the parallelism is).
        let world = Arc::new(World::builder().usual::<f32>());
        let accel = Arc::new(Accelerator::builder(world.clone()).build());

        let h = WIDTH / (COLS as f32 - 1.0); // uniform cell size (square cells)
        let m = MASS_TOTAL / (COLS as f32 * ROWS as f32); // per dynamic body mass
        // Spring damping for the target ratio ZETA: c = zeta · 2·√(k·μ), reduced
        // mass μ = m/2 for two equal masses on a spring (1.0 = critical damping).
        let c = ZETA * 2.0 * (K * (m * 0.5)).sqrt();

        // One shared RNG stream so the curtains get distinct-but-reproducible kicks.
        let rng = Rng(Arc::new(tokio::sync::Mutex::new(SEED)));
        let tmp: Vec<_> = (0..NUM_CURTAINS)
            .map(|i| {
                let world_cloned = world.clone();
                let accel_cloned = accel.clone();
                let rng_cloned = rng.clone();
                async move {
                    let x_offset = i as f32 * (WIDTH + GAP); // side by side, GAP clear between
                    build_curtain(&world_cloned, &accel_cloned, x_offset, h, m, c, &rng_cloned)
                        .await
                }
            })
            .collect();

        let mut curtains: Vec<Curtain> = Vec::new();
        for t in tmp {
            curtains.push(t.await);
        }

        // Instance buffers sized for ALL curtains: COLS·ROWS nodes + all springs each.
        let per_nodes = COLS * ROWS;
        let per_springs = ROWS * (COLS - 1) + COLS * (ROWS - 1);
        let circle_cap = NUM_CURTAINS * per_nodes;
        let line_cap = NUM_CURTAINS * per_springs;

        let last = Instant::now();

        Cloth {
            max_dt: MAX_DT,
            last,
            curtains,
            circles: CircleRenderer::with_capacity(gpu.device(), gpu.surface_format(), circle_cap),
            lines: LineRenderer::with_capacity(gpu.device(), gpu.surface_format(), line_cap),
            orbit: OrbitCamera::new(),
            dragging: false,
            last_cursor: None,
        }
    }

    fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::MouseInput { state, button, .. } if *button == MouseButton::Left => {
                self.dragging = *state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x, position.y);
                if self.dragging
                    && let Some((lx, ly)) = self.last_cursor
                {
                    let sens = 0.005_f32;
                    self.orbit
                        .rotate((p.0 - lx) as f32 * sens, -(p.1 - ly) as f32 * sens);
                }
                self.last_cursor = Some(p);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                self.orbit.zoom(0.9_f32.powf(steps)); // scroll up → zoom in
            }
            _ => {}
        }
    }

    async fn render(&mut self, frame: &mut Frame<'_>) {
        // --- One fixed sim step, ALL curtains together. Each is an independent
        // mechanism; join_all keeps both dispatches in flight so the shared
        // accelerator batches their kernels in parallel (cooperative on this
        // thread, parallel in the accelerator's rayon pool). ---
        let dt = self.tick();

        let epoch = Epoch::standalone(dt, 1.0);
        let futs: Vec<_> = self
            .curtains
            .iter()
            .map(|cu| cu.mech.step(&epoch))
            .collect();
        for r in join_all(futs).await {
            r.unwrap();
        }

        let proj = self.orbit.projector();

        // --- Nodes → circles (far-to-near for overdraw). ---
        let mut drawn: Vec<(f32, CircleInstance)> = Vec::new();
        for cu in &self.curtains {
            for (id, kinematic) in &cu.nodes {
                let p = cu.mech.body_absolute_position(*id).await.unwrap();
                if let Some((center, depth)) = proj.project_point(p) {
                    let (rw, color) = if *kinematic {
                        (0.014, [0.98, 0.62, 0.20, 1.0]) // pinned top row — orange
                    } else {
                        (0.008, [0.45, 0.75, 0.95, 1.0]) // cloth node — blue
                    };
                    let half = rw / depth * proj.focal;
                    drawn.push((
                        depth,
                        CircleInstance {
                            center: center.split(),
                            half: [half, half],
                            color,
                        },
                    ));
                }
            }
        }
        drawn.sort_by(|a, b| b.0.total_cmp(&a.0));
        let circles: Vec<CircleInstance> = drawn.into_iter().map(|(_, c)| c).collect();

        // --- Springs → lines. ---
        let mut segments: Vec<LineInstance> = Vec::new();
        for cu in &self.curtains {
            for (a, b) in &cu.springs {
                let pa = cu.mech.body_absolute_position(*a).await.unwrap();
                let pb = cu.mech.body_absolute_position(*b).await.unwrap();
                if let (Some((ca, _)), Some((cb, _))) =
                    (proj.project_point(pa), proj.project_point(pb))
                {
                    segments.push(LineInstance {
                        a: ca.split(),
                        b: cb.split(),
                        half: 0.0018,
                        color: [0.50, 0.55, 0.65, 0.7],
                    });
                }
            }
        }

        frame.clear(melies::wgpu::Color::BLACK);
        self.lines.draw(frame, &segments);
        self.circles.draw(frame, &circles);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    melies::run::<Cloth>(
        Config::builder()
            .title("Cloth grids — two mechanisms (drag = orbit, scroll = zoom)")
            .build(),
    )?;
    Ok(())
}
