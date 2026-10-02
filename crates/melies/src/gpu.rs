// SPDX-License-Identifier: MIT

use std::sync::Arc;

use winit::window::Window;

use crate::wgpu;

/// Ready GPU context handed to [`crate::Example::init`] and
/// [`crate::Example::resize`]. Owns the surface, device, queue and the live
/// surface configuration.
pub struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    /// Bring up wgpu against `window`. Blocks on the async init via pollster
    /// (no tokio/async-std — see project conventions).
    pub(crate) fn new(window: Arc<Window>, present_mode: wgpu::PresentMode) -> Self {
        pollster::block_on(Self::new_async(window, present_mode))
    }

    async fn new_async(window: Arc<Window>, present_mode: wgpu::PresentMode) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        // Arc<Window> gives wgpu a 'static surface target.
        let surface = instance
            .create_surface(window)
            .expect("create wgpu surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("request a wgpu adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("melies device"),
                ..Default::default()
            })
            .await
            .expect("request a wgpu device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        winit::dpi::PhysicalSize::new(self.config.width, self.config.height)
    }

    /// Reconfigure the surface to a new size (ignores zero-area sizes).
    pub(crate) fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Re-apply the current configuration (used to recover Lost/Outdated surfaces).
    pub(crate) fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Acquire the next swapchain frame. Returns wgpu's outcome enum; the caller
    /// branches on it (`Success`/`Suboptimal` carry the [`wgpu::SurfaceTexture`],
    /// `Outdated`/`Lost` ask for a reconfigure).
    pub(crate) fn acquire(&self) -> wgpu::CurrentSurfaceTexture {
        self.surface.get_current_texture()
    }
}
