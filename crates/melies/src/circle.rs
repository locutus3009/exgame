// SPDX-License-Identifier: MIT

use crate::Frame;

const SHADER_SRC: &str = r#"
// Viewport scale applied to clip-space positions. `scale.x = height/width`
// keeps circles round on non-square windows; instance data stays aspect-free.
struct Viewport {
    scale: vec2<f32>,
};
@group(0) @binding(0) var<uniform> vp: Viewport;

struct Inst {
    @location(0) center: vec2<f32>,
    @location(1) half:   vec2<f32>,
    @location(2) color:  vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Inst) -> VsOut {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0,  1.0),
    );
    let local = corners[vi];
    var out: VsOut;
    out.clip  = vec4<f32>((inst.center + local * inst.half) * vp.scale, 0.0, 1.0);
    out.local = local;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    let aa = fwidth(d);
    let alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);
    if (alpha <= 0.0) { discard; }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleInstance {
    pub center: [f32; 2],
    pub half: [f32; 2],
    pub color: [f32; 4],
}

/// Mirror of the WGSL `Viewport` uniform. `_pad` rounds the struct up to the
/// 16-byte minimum uniform-buffer binding size.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Viewport {
    scale: [f32; 2],
    _pad: [f32; 2],
}

const CAPACITY: usize = 256;

pub struct CircleRenderer {
    pipeline: wgpu::RenderPipeline,
    instances: wgpu::Buffer,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: usize,
}

impl CircleRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self::with_capacity(device, format, CAPACITY)
    }

    /// Like `new`, but sizes the instance buffer for up to `capacity` circles per
    /// frame — examples that draw large point clouds (e.g. an N×N grid) pass their
    /// own bound instead of the default `CAPACITY`.
    pub fn with_capacity(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        capacity: usize,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("circle"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });
        const ATTRS: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            // No bind groups → let wgpu derive an empty layout from the shader.
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CircleInstance>() as u64,
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
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: capacity as u64 * std::mem::size_of::<CircleInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("circle.viewport"),
            size: std::mem::size_of::<Viewport>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // `layout: None` made wgpu derive the bind-group layout from the shader;
        // reuse it here rather than declaring it twice.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("circle.globals"),
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

    pub fn draw(&self, frame: &mut Frame<'_>, circles: &[CircleInstance]) {
        assert!(circles.len() <= self.capacity);

        // Disjoint borrows of the frame: queue (upload) + view & encoder (pass).
        let (gpu, view, encoder) = frame.parts();
        // Aspect correction lives in the shader; feed it the current viewport.
        let size = gpu.size();
        let viewport = Viewport {
            scale: [size.height as f32 / size.width as f32, 1.0],
            _pad: [0.0, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&viewport));
        gpu.queue()
            .write_buffer(&self.instances, 0, bytemuck::cast_slice(circles));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("circle.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // on top of the already-cleared frame!
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
        pass.draw(0..4, 0..circles.len() as u32);
    }
}
