// SPDX-License-Identifier: MIT

use crate::Frame;

// Instanced thick lines. Each instance carries two endpoints in the SAME
// pre-scale NDC space as `CircleInstance.center` (so lines and circles align),
// a half-width, and a colour. The vertex shader expands each segment into a quad
// in SCREEN space (after `vp.scale`), so the width is uniform on screen
// regardless of orientation or window aspect.
const SHADER_SRC: &str = r#"
struct Viewport { scale: vec2<f32> };
@group(0) @binding(0) var<uniform> vp: Viewport;

struct Inst {
    @location(0) a:     vec2<f32>,
    @location(1) b:     vec2<f32>,
    @location(2) half:  f32,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Inst) -> VsOut {
    // Quad as two triangles; e.x = t along the segment, e.y = side (±1).
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let e = quad[vi];

    // Work in screen space (after aspect scale) so width is uniform on screen.
    let sa = inst.a * vp.scale;
    let sb = inst.b * vp.scale;
    let d = sb - sa;
    let len = length(d);
    var dir = vec2<f32>(1.0, 0.0);
    if (len > 1e-6) { dir = d / len; }
    let perp = vec2<f32>(-dir.y, dir.x) * inst.half;

    let pos = mix(sa, sb, e.x) + perp * e.y;
    var out: VsOut;
    out.clip = vec4<f32>(pos, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineInstance {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub half: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Viewport {
    scale: [f32; 2],
    _pad: [f32; 2],
}

const CAPACITY: usize = 256;

pub struct LineRenderer {
    pipeline: wgpu::RenderPipeline,
    instances: wgpu::Buffer,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: usize,
}

impl LineRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self::with_capacity(device, format, CAPACITY)
    }

    /// Like `new`, but sizes the instance buffer for up to `capacity` line
    /// segments per frame — examples that draw dense meshes (e.g. an N×N spring
    /// grid) pass their own bound instead of the default `CAPACITY`.
    pub fn with_capacity(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        capacity: usize,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
        const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &ATTRS,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line.instances"),
            size: capacity as u64 * std::mem::size_of::<LineInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line.viewport"),
            size: std::mem::size_of::<Viewport>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("line.globals"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            instances,
            globals,
            bind_group,
            capacity,
        }
    }

    pub fn draw(&self, frame: &mut Frame<'_>, lines: &[LineInstance]) {
        assert!(lines.len() <= self.capacity);
        if lines.is_empty() {
            return;
        }

        let (gpu, view, encoder) = frame.parts();
        let size = gpu.size();
        let viewport = Viewport {
            scale: [size.height as f32 / size.width as f32, 1.0],
            _pad: [0.0, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&viewport));
        gpu.queue()
            .write_buffer(&self.instances, 0, bytemuck::cast_slice(lines));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("line.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // over the already-cleared frame
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..lines.len() as u32);
    }
}
