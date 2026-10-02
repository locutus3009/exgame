// SPDX-License-Identifier: MIT

use crate::gpu::Gpu;
use crate::wgpu;

/// Per-frame drawing handle handed to [`crate::Example::render`].
///
/// Wraps the acquired surface view and the frame's command encoder. The host
/// owns acquire/submit/present; the example only records work here.
pub struct Frame<'a> {
    gpu: &'a Gpu,
    view: &'a wgpu::TextureView,
    encoder: &'a mut wgpu::CommandEncoder,
}

impl<'a> Frame<'a> {
    pub(crate) fn new(
        gpu: &'a Gpu,
        view: &'a wgpu::TextureView,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> Self {
        Self { gpu, view, encoder }
    }

    pub fn gpu(&self) -> &Gpu {
        self.gpu
    }

    pub fn view(&self) -> &wgpu::TextureView {
        self.view
    }

    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder
    }

    /// Borrow the GPU context, target view, and encoder simultaneously.
    ///
    /// Recording a custom render pass needs `&mut encoder` *and* `&view` at the
    /// same time; the individual accessors each reborrow the whole `Frame`, so
    /// they can't be held together. This hands out the three disjoint fields.
    pub fn parts(&mut self) -> (&Gpu, &wgpu::TextureView, &mut wgpu::CommandEncoder) {
        (self.gpu, self.view, self.encoder)
    }

    /// Convenience: record a clear-only render pass over this frame's view.
    pub fn clear(&mut self, color: wgpu::Color) {
        let _pass = self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("melies clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: self.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
}
