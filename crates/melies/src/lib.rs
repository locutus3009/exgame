// SPDX-License-Identifier: MIT

//! melies — standalone wgpu/winit example-host for newton/hitchcock examples.
//!
//! The example implements [`Example`]; [`run`] owns the window + wgpu and drives
//! the loop. Examples reach raw wgpu types through `melies::wgpu::*`.

use async_trait::async_trait;

pub use wgpu;
pub use winit;

mod app;
mod circle;
mod config;
mod frame;
mod gpu;
mod line;

pub use app::run;
pub use circle::{CircleInstance, CircleRenderer};
pub use config::{Config, ConfigBuilder};
pub use frame::Frame;
pub use gpu::Gpu;
pub use line::{LineInstance, LineRenderer};

/// What an example implements. The host calls these; it owns the window and the
/// acquire/submit/present cycle.
#[async_trait]
pub trait Example: Sized + Send + Sync {
    /// Build pipelines, buffers, bind groups — once, before the first frame.
    async fn init(gpu: &Gpu) -> Self;

    /// Draw one frame. Record into `frame.encoder()` against `frame.view()`.
    async fn render(&mut self, frame: &mut Frame<'_>);

    /// React to a resize. The surface is already reconfigured before this call.
    /// Default: no-op.
    fn resize(&mut self, gpu: &Gpu, size: winit::dpi::PhysicalSize<u32>) {
        let _ = (gpu, size);
    }

    /// See every raw window event (mouse, keyboard, …) before the host's own
    /// handling. The host still drives resize/redraw/close itself; this is for
    /// examples that need input (e.g. orbit-drag a 3D view). Default: no-op.
    fn window_event(&mut self, event: &winit::event::WindowEvent) {
        let _ = event;
    }
}
