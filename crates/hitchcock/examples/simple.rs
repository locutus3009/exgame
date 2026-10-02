// SPDX-License-Identifier: MIT

use aristotle::{Epoch, World, WorldId};
use async_trait::async_trait;
use clifford::pga3::{Motor, Point};
use hitchcock::{Camera, CameraField, MarkerKind};
use melies::winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use melies::winit::keyboard::Key;
use melies::{
    CircleInstance, CircleRenderer, Config, Example, Frame, Gpu, LineInstance, LineRenderer,
};
use newton::{
    Accelerator, Inert, Mechanism, RigidBody, gravity::GravityPropagator,
    integrator::ImplicitIntegrator,
};
use peano::fixed::{Fix, types::I96F32};
use peano::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

type Fx = Fix<I96F32>;

// ============================================================================
// Orbit camera — DEMO-ONLY perspective projection on the CPU.
// ============================================================================
//
// Pinhole projection good enough to inspect the rig in 3D. Not the engine
// renderer (that's planned raw-Vulkan/PGA); this stays throwaway example glue.

struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    dist: f32,
    /// Focal length = 1/tan(fov/2).
    focal: f32,
    /// Orbit centre / look-at point, in world space. Tracked onto `real`
    /// (the rig eye, local 0,0,0) each frame so the view stays centred on it.
    /// Axes stay inertial (world-aligned) — only the pivot follows `real`.
    center: Vector3<f32>,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.5,
            dist: 8.0,
            focal: 1.0 / (30.0_f32.to_radians().tan()), // ~60° vertical FOV
            center: Vector3::ZERO,
        }
    }

    /// Move the orbit pivot (and look-at) onto `c` — `real`'s world position.
    fn set_center(&mut self, c: Vector3<f32>) {
        self.center = c;
    }

    fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        let lim = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = (self.pitch + dpitch).clamp(-lim, lim);
    }

    fn zoom(&mut self, factor: f32) {
        self.dist = (self.dist * factor).clamp(1.0, 200.0);
    }

    /// Eye position, orbiting `center` (the look-at point sits there).
    fn eye(&self) -> Vector3<f32> {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let off = Vector3::from([self.dist * cp * sy, self.dist * sp, self.dist * cp * cy]);
        self.center + off
    }

    /// Snapshot the current orbit pose into a [`Projector`]: external eye
    /// looking at `center`, world-aligned (inertial) basis.
    fn projector(&self) -> Projector {
        let eye = self.eye();
        // Look at `center` (= real); right-handed orthonormal view basis.
        let fwd = normalize(self.center - eye);
        let right = normalize(fwd.cross(Vector3::from([0.0, 1.0, 0.0])));
        let up = right.cross(fwd);
        Projector::Orbit {
            eye,
            right,
            up,
            fwd,
            focal: self.focal,
        }
    }
}

/// Which view the example renders. Toggle with `m`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CamMode {
    /// External orbit around `real` (drag/scroll), inertial axes.
    Orbit,
    /// Hard-fixed game view: from `real`, along its local −z, with the
    /// rig body's local x/y as screen right/up.
    Eye,
}

/// A frozen projection used for one frame. Both variants reduce a world point
/// to camera-space (right, up, depth) and share the same pinhole. melies applies
/// the aspect correction downstream.
enum Projector {
    Orbit {
        eye: Vector3<f32>,
        right: Vector3<f32>,
        up: Vector3<f32>,
        fwd: Vector3<f32>,
        focal: f32,
    },
    /// `view` is `real`'s pose inverse (world → body-local), built via PGA.
    Eye { view: Motor<f32>, focal: f32 },
}

impl Projector {
    fn focal(&self) -> f32 {
        match self {
            Projector::Orbit { focal, .. } | Projector::Eye { focal, .. } => *focal,
        }
    }

    /// World point → pre-scale NDC centre + depth in front. `None` if behind.
    fn project_point(&self, p: Vector3<f32>) -> Option<(Vector2<f32>, f32)> {
        match self {
            Projector::Orbit {
                eye,
                right,
                up,
                fwd,
                focal,
            } => {
                let pf: Vector3<f32> = Vector3::from([p[0], p[1], p[2]]);
                let rel: Vector3<f32> = pf - *eye;
                let cz = rel.dot(*fwd); // depth in front of the eye
                pinhole(Vector2::from([rel.dot(*right), rel.dot(*up)]), cz, *focal)
            }
            Projector::Eye { view, focal } => {
                // World → real's local frame, honestly via the body's Motor.
                let local = view.conjugate(&Point::new(p)).coords();
                // Camera looks along local −z; local +x → right, +y → up.
                pinhole(Vector2::from([local[0], local[1]]), -local[2], *focal)
            }
        }
    }
}

/// Shared pinhole tail: divide the (right, up) offsets by depth and scale by
/// focal length. `None` for points behind the eye OR any non-finite result —
/// `coords()` can divide by a near-zero homogeneous weight when a body pose
/// degenerates, and a NaN depth would later panic the far-to-near sort.
fn pinhole(ru: Vector2<f32>, cz: f32, focal: f32) -> Option<(Vector2<f32>, f32)> {
    // Spelled through `partial_cmp` rather than `cz <= 0.05`: a NaN depth is
    // unordered against 0.05, so it compares neither greater nor less, and it
    // must be rejected here before it can panic the far-to-near sort.
    if !matches!(cz.partial_cmp(&0.05), Some(Ordering::Greater)) {
        return None;
    }
    let center = ru.scale(1.0 / cz).scale(focal);
    (center[0].is_finite() && center[1].is_finite() && cz.is_finite()).then_some((center, cz))
}

fn normalize(v: Vector3<f32>) -> Vector3<f32> {
    let n = v.dot(v).sqrt();
    if n > 1e-9 { v.scale(1.0 / n) } else { v }
}

// World radius + RGBA per marker kind. Secondary refs are deliberately small.
fn style(kind: MarkerKind) -> (f32, [f32; 4]) {
    match kind {
        MarkerKind::Eye => (0.16, [0.30, 0.95, 0.95, 1.0]), // camera eye — cyan
        MarkerKind::Intermediate => (0.10, [0.40, 0.95, 0.40, 1.0]), // green
        MarkerKind::Anchor => (0.11, [0.98, 0.62, 0.20, 1.0]), // orange
        MarkerKind::TargetTracker => (0.08, [0.85, 0.45, 0.95, 1.0]), // purple
        MarkerKind::Attachment => (0.045, [0.75, 0.78, 0.85, 0.9]), // small grey
    }
}

struct Simple {
    cameras: Arc<CameraField<f32, Fx>>,
    camera_id: WorldId,
    max_dt: f32,
    last: Instant,
    circles: CircleRenderer,
    lines: LineRenderer,
    prop: Arc<GravityPropagator<f32, Fx>>,
    mech: Arc<Mechanism<f32, Fx>>,
    camera_mech: Arc<Mechanism<f32, Fx>>,
    body_id: WorldId,

    // Second, non-gravitating body in its own mechanism. The camera tracks a
    // mechanism's centroid, so a separate mechanism is what lets us flip the
    // target between the two bodies.
    mech_b: Arc<Mechanism<f32, Fx>>,
    body_b_id: WorldId,
    elapsed: f32,
    targeting_b: bool,

    orbit: OrbitCamera,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    mode: CamMode,
}

/// Sim-seconds between target switches.
const SWITCH_PERIOD: f32 = 20.0;

impl Simple {
    pub fn tick(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(self.max_dt);
        self.last = now;
        dt
    }
}

#[async_trait]
impl Example for Simple {
    async fn init(gpu: &Gpu) -> Self {
        let world = Arc::new(World::builder().usual::<f32>());
        // One accelerator for the whole app; every mechanism shares it.
        let accelerator = Arc::new(Accelerator::builder(world.clone()).build());
        let mech = Arc::new(Mechanism::new(ImplicitIntegrator::LieEuler(
            accelerator.clone(),
        )));

        let body_id = mech
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::ZERO,
                5.0,
            )))
            .await;

        let prop = Arc::new(GravityPropagator::new(0.1));
        prop.register(body_id);
        mech.add_field(Box::new(prop.clone())).await;

        // Second body: non-gravitating (NOT registered with the propagator), in
        // its own mechanism, 5 units away. Static — it just sits there.
        let mech_b = Arc::new(Mechanism::new(ImplicitIntegrator::LieEuler(
            accelerator.clone(),
        )));
        let body_b_id = mech_b
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([10.0, 0.0, 0.0]),
                &Vector3::ZERO,
                1.0,
            )))
            .await;

        let cameras = Arc::new(CameraField::new(world.clone(), prop.clone()));
        let (camera_id, camera_mech) = cameras.clone().add_camera(mech.clone(), 2.0, 1.0).await;

        let last = Instant::now();
        let circles = CircleRenderer::new(gpu.device(), gpu.surface_format());
        let lines = LineRenderer::new(gpu.device(), gpu.surface_format());

        Simple {
            cameras,
            camera_id,
            max_dt: 0.1,
            last,
            circles,
            lines,
            prop,
            mech,
            camera_mech,
            body_id,
            mech_b,
            body_b_id,
            elapsed: 0.0,
            targeting_b: false,
            orbit: OrbitCamera::new(),
            dragging: false,
            last_cursor: None,
            mode: CamMode::Orbit,
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
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed && !event.repeat;
                if pressed
                    && matches!(&event.logical_key, Key::Character(c) if c.eq_ignore_ascii_case("m"))
                {
                    self.mode = match self.mode {
                        CamMode::Orbit => CamMode::Eye,
                        CamMode::Eye => CamMode::Orbit,
                    };
                    let label = match self.mode {
                        CamMode::Orbit => "orbit (drag/scroll)",
                        CamMode::Eye => "eye: from real along local −z",
                    };
                    eprintln!("camera mode → {label}");
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                // scroll up → zoom in
                self.orbit.zoom(0.9_f32.powf(steps));
            }
            _ => {}
        }
    }

    async fn render(&mut self, frame: &mut Frame<'_>) {
        // --- Simulation ---
        let dt = self.tick();
        let epoch = Epoch::standalone(dt, 1.0);
        // `mech`, `mech_b` and the camera all step concurrently in one batch.
        // The camera OBSERVES the targets (its tracker reads target.centroid()
        // / centroid_velocity()), but those now read a PUBLISHED snapshot from
        // their own locks — not `inner` — so the camera no longer blocks on a
        // target's step-long write lock. It reads last-step's summary (lag-1),
        // which is exactly the observer semantics we want.
        for r in futures::future::join_all([
            self.mech.step(&epoch),
            self.mech_b.step(&epoch),
            self.camera_mech.step(&epoch),
        ])
        .await
        {
            r.unwrap();
        }
        self.prop.advance_epoch();

        // --- Switch target every SWITCH_PERIOD sim-seconds ---
        self.elapsed += dt;
        let want_b = ((self.elapsed / SWITCH_PERIOD) as u64) % 2 == 1;
        if want_b != self.targeting_b {
            self.targeting_b = want_b;
            let tgt = if want_b { &self.mech_b } else { &self.mech }.clone();
            self.cameras
                .update_camera(self.camera_id, async move |c| {
                    c.retarget(Some(tgt.clone())).await
                })
                .await;
        }

        // --- Collect world markers as (position, world-radius, color) ---
        // Both main bodies; the active target is drawn larger.
        let pa = self
            .mech
            .inspect_body(self.body_id, async |b| RigidBody::position(b))
            .await
            .unwrap();
        let pb = self
            .mech_b
            .inspect_body(self.body_b_id, async |b| RigidBody::position(b))
            .await
            .unwrap();
        let (ra, rb) = if self.targeting_b {
            (0.13, 0.22)
        } else {
            (0.22, 0.13)
        };
        let mut points: Vec<(Vector3<f32>, f32, [f32; 4])> = vec![
            (pa, ra, [0.95, 0.35, 0.25, 1.0]), // body A — red (gravitating)
            (pb, rb, [0.95, 0.85, 0.30, 1.0]), // body B — amber (non-gravitating)
        ];
        for (p, kind) in self
            .cameras
            .inspect_camera(self.camera_id, Camera::markers)
            .await
            .unwrap()
        {
            let (rw, color) = style(kind);
            points.push((p, rw, color));
        }

        // --- Centre the orbit on `real` (rig eye, local 0,0,0) so the view
        // stays locked on it; axes remain inertial. ---
        let real_pos = self
            .cameras
            .inspect_camera(self.camera_id, Camera::position)
            .await
            .unwrap();
        self.orbit.set_center(Vector3::from([
            real_pos[0] as f32,
            real_pos[1] as f32,
            real_pos[2] as f32,
        ]));

        // --- Pick the projection for this frame: orbit, or the hard-fixed game
        // view from `real` along its local −z (built from real's pose Motor). ---
        let projector = match self.mode {
            CamMode::Orbit => self.orbit.projector(),
            CamMode::Eye => match self.cameras.viewport(self.camera_id).await {
                Some(pose) => Projector::Eye {
                    view: pose.inverse(),
                    focal: self.orbit.focal,
                },
                None => self.orbit.projector(), // before the first step
            },
        };
        let focal = projector.focal();

        // --- Project (perspective) and build instances, far-to-near for overdraw ---
        let mut drawn: Vec<(f32, CircleInstance)> = Vec::new();
        for (p, rw, color) in &points {
            let (rw, color) = (*rw, *color);
            if let Some((center, depth)) = projector.project_point(*p) {
                let half = rw / depth * focal;
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
        drawn.sort_by(|a, b| b.0.total_cmp(&a.0)); // far first (total order, NaN-safe)
        let instances: Vec<CircleInstance> = drawn.into_iter().map(|(_, c)| c).collect();

        // --- Lines: rigid frames of real/intermediate (thick, body colour) and
        // the longitudinal springs (thin, distinct colour). Drawn under circles. ---
        const FRAME_HALF: f32 = 0.010;
        const SPRING_HALF: f32 = 0.004;
        let spring_color = [0.45, 0.60, 0.95, 0.85]; // light blue
        let mut segments: Vec<LineInstance> = Vec::new();

        let project_seg =
            |a: Vector3<f32>, b: Vector3<f32>| -> Option<(Vector2<f32>, Vector2<f32>)> {
                let (ca, _) = projector.project_point(a)?;
                let (cb, _) = projector.project_point(b)?;
                Some((ca, cb))
            };

        // Springs first (under the frames).
        for (a, b) in self
            .cameras
            .inspect_camera(self.camera_id, Camera::spring_segments)
            .await
            .unwrap()
        {
            if let Some((ca, cb)) = project_seg(a, b) {
                segments.push(LineInstance {
                    a: ca.split(),
                    b: cb.split(),
                    half: SPRING_HALF,
                    color: spring_color,
                });
            }
        }
        // Frames on top of springs.
        for (a, b, kind) in self
            .cameras
            .inspect_camera(self.camera_id, Camera::frame_segments)
            .await
            .unwrap()
        {
            if let Some((ca, cb)) = project_seg(a, b) {
                let color = style(kind).1;
                segments.push(LineInstance {
                    a: ca.split(),
                    b: cb.split(),
                    half: FRAME_HALF,
                    color,
                });
            }
        }

        frame.clear(melies::wgpu::Color::BLACK);
        self.lines.draw(frame, &segments);
        self.circles.draw(frame, &instances);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    melies::run::<Simple>(
        Config::builder()
            .title("Hitchcock demo: camera rig (drag = orbit, scroll = zoom)")
            .build(),
    )?;
    Ok(())
}
