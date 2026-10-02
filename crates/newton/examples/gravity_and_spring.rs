// SPDX-License-Identifier: MIT

//! Smallest possible melies consumer: open a window and clear it each frame.
//! Run with: cargo run -p newton --example gravity_and_spring

use aristotle::{Epoch, World, WorldId};
use async_trait::async_trait;
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use melies::{CircleInstance, CircleRenderer, Config, Example, Frame, Gpu};
use newton::{
    Accelerator, Inert, Mechanism, RigidBody, gravity::GravityPropagator,
    integrator::ImplicitIntegrator,
};
use peano::prelude::*;
use std::sync::Arc;
use std::time::Instant;

struct Simple {
    mech1: Mechanism<f32, f32>,
    mech2: Mechanism<f32, f32>,
    max_dt: f32,
    last: Instant,
    id1: WorldId,
    id2: WorldId,
    id3: WorldId,
    id4: WorldId,
    center: WorldId,
    circles: CircleRenderer,
    prop: Arc<GravityPropagator<f32>>,
}

const WORLD_TO_CLIP: f32 = 0.3;
const RADIUS: f32 = 0.08;

impl Simple {
    fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
        AxialSpringDamper::builder(world, SimpleSpringDamper)
            .rest(rest)
            .stiffness(k)
            .damping(c)
            .build()
    }

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
        // One accelerator for the whole app; both mechanisms share it.
        let world = Arc::new(World::builder().build());
        let accelerator = Arc::new(Accelerator::builder(world.clone()).build());
        let mech1 = Mechanism::new(ImplicitIntegrator::LieEuler(accelerator.clone()));
        let mech2 = Mechanism::new(ImplicitIntegrator::LieEuler(accelerator.clone()));

        let id1 = mech1
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([-3.0, 0.0, 0.0]),
                &Vector3::from([0.0, -1.0, 0.0]),
                1.0,
            )))
            .await;
        let id2 = mech1
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([-2.0, 0.0, 0.0]),
                &Vector3::from([0.0, -1.0, 0.0]),
                1.0,
            )))
            .await;
        let id3 = mech1
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([3.0, 0.0, 0.0]),
                &Vector3::from([1.0, 0.0, 0.0]),
                1.0,
            )))
            .await;
        let id4 = mech1
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::from([2.0, 0.0, 0.0]),
                &Vector3::from([-1.0, 0.0, 0.0]),
                1.0,
            )))
            .await;

        let damping = 0.5;
        let stifness = 5.0;

        mech1
            .connect(
                id1,
                vec![(Self::spring(world.clone(), 0.9, stifness, damping), id2)],
            )
            .await;
        mech1
            .connect(
                id3,
                vec![(
                    Self::spring(world.clone(), 0.9, stifness * 10.0, damping / 100.0),
                    id4,
                )],
            )
            .await;

        let center = mech2
            .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::ZERO,
                2200.0,
            )))
            .await;

        let prop = Arc::new(GravityPropagator::new(0.001));
        prop.register(id1);
        prop.register(id2);
        prop.register(center);
        mech1.add_field(Box::new(prop.clone())).await;
        mech2.add_field(Box::new(prop.clone())).await;

        let last = Instant::now();

        let circles = CircleRenderer::new(gpu.device(), gpu.surface_format());

        Simple {
            mech1,
            mech2,
            id1,
            id2,
            id3,
            id4,
            center,
            last,
            max_dt: 0.1,
            circles,
            prop,
        }
    }

    async fn render(&mut self, frame: &mut Frame<'_>) {
        frame.clear(melies::wgpu::Color::BLACK);
        let dt = self.tick();
        let epoch = Epoch::standalone(dt, 1.0);
        self.mech1.step(&epoch).await.unwrap();
        self.mech2.step(&epoch).await.unwrap();

        let p1 = self
            .mech1
            .inspect_body(self.id1, async |b| RigidBody::position(b))
            .await
            .unwrap();
        let p2 = self
            .mech1
            .inspect_body(self.id2, async |b| RigidBody::position(b))
            .await
            .unwrap();
        let p3 = self
            .mech1
            .inspect_body(self.id3, async |b| RigidBody::position(b))
            .await
            .unwrap();
        let p4 = self
            .mech1
            .inspect_body(self.id4, async |b| RigidBody::position(b))
            .await
            .unwrap();
        let pc = self
            .mech2
            .inspect_body(self.center, async |b| RigidBody::position(b))
            .await
            .unwrap();

        // Instance data is aspect-free world→clip; the shader's Viewport uniform
        // applies the height/width correction that keeps circles round.
        let circle = |p: Vector3<f32>, color: [f32; 4]| CircleInstance {
            center: [p[0] * WORLD_TO_CLIP, p[1] * WORLD_TO_CLIP],
            half: [RADIUS, RADIUS],
            color,
        };
        let instances = [
            circle(p1, [0.95, 0.35, 0.25, 1.0]),
            circle(p2, [0.30, 0.95, 0.95, 1.0]),
            circle(p3, [0.30, 0.55, 0.95, 1.0]),
            circle(p4, [0.30, 0.55, 0.35, 1.0]),
            circle(pc, [0.95, 0.55, 0.95, 1.0]),
        ];
        self.circles.draw(frame, &instances);
        self.prop.advance_epoch();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    melies::run::<Simple>(
        Config::builder()
            .title("Newton demo: gravity and spring")
            .build(),
    )?;
    Ok(())
}
