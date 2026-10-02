// SPDX-License-Identifier: MIT

use std::error::Error;
use std::sync::Arc;
use tokio::runtime::Runtime;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::gpu::Gpu;
use crate::{Config, Example, Frame, wgpu};

/// Custom events dispatched from Tokio background tasks back to winit's main thread.
pub enum CustomEvent<E: Example> {
    Initialized { gpu: Gpu, example: E },
}

/// Internal winit driver running alongside a multi-threaded Tokio runtime.
struct App<E: Example + Send + Sync + 'static> {
    config: Config,
    proxy: EventLoopProxy<CustomEvent<E>>,
    rt: Arc<Runtime>,
    window: Option<Arc<Window>>,
    gpu: Option<Gpu>,
    example: Option<E>,
    is_initializing: bool,
}

impl<E: Example + Send + Sync + 'static> App<E> {
    fn new(config: Config, proxy: EventLoopProxy<CustomEvent<E>>, rt: Arc<Runtime>) -> Self {
        Self {
            config,
            proxy,
            rt,
            window: None,
            gpu: None,
            example: None,
            is_initializing: false,
        }
    }
}

impl<E: Example + Send + Sync + 'static> ApplicationHandler<CustomEvent<E>> for App<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() || self.is_initializing {
            return;
        }

        self.is_initializing = true;

        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());

        let present_mode = self.config.present_mode;
        let proxy = self.proxy.clone();
        let rt_handle = self.rt.handle().clone();

        // Spawn async initialization directly onto Tokio's multi-threaded task pool
        rt_handle.spawn(async move {
            let gpu = Gpu::new(window, present_mode);
            let example = E::init(&gpu).await;

            // Notify Winit on the main thread when initialization completes
            let _ = proxy.send_event(CustomEvent::Initialized { gpu, example });
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: CustomEvent<E>) {
        match event {
            CustomEvent::Initialized { gpu, example } => {
                self.gpu = Some(gpu);
                self.example = Some(example);
                self.is_initializing = false;

                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.example.is_some()
            && let Some(window) = self.window.as_ref()
        {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(example) = self.example.as_mut() {
            example.window_event(&event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let (Some(gpu), Some(example)) = (self.gpu.as_mut(), self.example.as_mut()) {
                    gpu.resize(size);
                    example.resize(gpu, size);
                }
            }

            WindowEvent::RedrawRequested => {
                let (Some(gpu), Some(example)) = (self.gpu.as_mut(), self.example.as_mut()) else {
                    return;
                };

                use wgpu::CurrentSurfaceTexture as Acquired;
                match gpu.acquire() {
                    Acquired::Success(surface_texture) | Acquired::Suboptimal(surface_texture) => {
                        let view = surface_texture
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            gpu.device()
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("melies frame"),
                                });
                        {
                            let mut frame = Frame::new(gpu, &view, &mut encoder);

                            // Execute async render inside Tokio's runtime block_on
                            self.rt.block_on(example.render(&mut frame));
                        }
                        gpu.queue().submit([encoder.finish()]);
                        surface_texture.present();
                    }
                    Acquired::Outdated | Acquired::Lost => gpu.reconfigure(),
                    Acquired::Timeout | Acquired::Occluded | Acquired::Validation => {}
                }
            }

            _ => {}
        }
    }
}

/// Start a multi-threaded Tokio runtime, bring up wgpu, run the event loop to completion,
/// and return the final example state once the window closes.
pub fn run<E: Example + Send + Sync + 'static>(config: Config) -> Result<E, Box<dyn Error>> {
    // 1. Build a Multi-Threaded Tokio Runtime
    let rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?,
    );

    // 2. Set up Winit Event Loop with Custom Events
    let event_loop: EventLoop<CustomEvent<E>> = EventLoop::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let proxy = event_loop.create_proxy();
    let mut app = App::<E>::new(config, proxy, rt);

    // 3. Run Winit loop on main thread
    event_loop.run_app(&mut app)?;

    app.example
        .ok_or_else(|| "melies: window closed before the example initialised".into())
}
