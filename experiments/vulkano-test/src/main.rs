// SPDX-License-Identifier: MIT

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::DeviceSize;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};

const N: usize = 65536;
const ITERATIONS: usize = 3;

// === 1. PROPER ALIGNMENT ===
// For storage buffers using std430 (default), a single f32 is 4-byte aligned.
// If you use structs in shaders, use #[repr(C, align(16))] with bytemuck traits.
// Here we keep the shader as float[] but demonstrate the alignment pattern.
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C, align(16))]
struct AlignedData {
    value: f32,
    _pad: [f32; 3],
}

impl AlignedData {
    fn new(v: f32) -> Self {
        Self {
            value: v,
            _pad: [0.0; 3],
        }
    }
}

fn build_pipeline(
    device: &Arc<Device>,
    shader: &Arc<ShaderModule>,
    entry: &str,
) -> Arc<ComputePipeline> {
    let cs = shader.entry_point(entry).unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

fn main() {
    let library = vulkano::VulkanLibrary::new().expect("no Vulkan library");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let physical = instance
        .enumerate_physical_devices()
        .expect("no devices")
        .max_by_key(|p| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 4,
            PhysicalDeviceType::IntegratedGpu => 3,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 1,
            _ => 0,
        })
        .expect("no GPU found");

    let queue_family = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
        .expect("no compute queue") as u32;

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: queue_family,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // === REUSABLE BUFFER ===
    // HOST_RANDOM_ACCESS = coherent, persistently mapped memory
    let shared_buffer = Buffer::new_slice::<AlignedData>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        N as DeviceSize,
    )
    .expect("failed to create buffer");

    {
        let mut write = shared_buffer.write().unwrap();
        for (i, item) in write.iter_mut().enumerate() {
            *item = AlignedData::new(i as f32);
        }
    }

    // 5 compute shaders
    mod cs1 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 460
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                layout(set = 0, binding = 0) buffer Data {
                    float data[];
                } buf;
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] += 1.0;
                }
            "
        }
    }
    mod cs2 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 460
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                layout(set = 0, binding = 0) buffer Data {
                    float data[];
                } buf;
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] *= 2.0;
                }
            "
        }
    }
    mod cs3 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 460
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                layout(set = 0, binding = 0) buffer Data {
                    float data[];
                } buf;
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] -= 0.5;
                }
            "
        }
    }
    mod cs4 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 460
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                layout(set = 0, binding = 0) buffer Data {
                    float data[];
                } buf;
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] = sqrt(max(buf.data[idx], 0.0));
                }
            "
        }
    }
    mod cs5 {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 460
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                layout(set = 0, binding = 0) buffer Data {
                    float data[];
                } buf;
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    buf.data[idx] = -buf.data[idx];
                }
            "
        }
    }

    let shader1 = cs1::load(device.clone()).unwrap();
    let shader2 = cs2::load(device.clone()).unwrap();
    let shader3 = cs3::load(device.clone()).unwrap();
    let shader4 = cs4::load(device.clone()).unwrap();
    let shader5 = cs5::load(device.clone()).unwrap();

    let pipeline1 = build_pipeline(&device, &shader1, "main");
    let pipeline2 = build_pipeline(&device, &shader2, "main");
    let pipeline3 = build_pipeline(&device, &shader3, "main");
    let pipeline4 = build_pipeline(&device, &shader4, "main");
    let pipeline5 = build_pipeline(&device, &shader5, "main");

    // Descriptor sets — created ONCE and reused across loop iterations
    let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let set1 = DescriptorSet::new(
        ds_allocator.clone(),
        pipeline1.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, shared_buffer.clone())],
        [],
    )
    .unwrap();
    let set2 = DescriptorSet::new(
        ds_allocator.clone(),
        pipeline2.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, shared_buffer.clone())],
        [],
    )
    .unwrap();
    let set3 = DescriptorSet::new(
        ds_allocator.clone(),
        pipeline3.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, shared_buffer.clone())],
        [],
    )
    .unwrap();
    let set4 = DescriptorSet::new(
        ds_allocator.clone(),
        pipeline4.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, shared_buffer.clone())],
        [],
    )
    .unwrap();
    let set5 = DescriptorSet::new(
        ds_allocator.clone(),
        pipeline5.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, shared_buffer.clone())],
        [],
    )
    .unwrap();

    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let groups = [(N as u32).div_ceil(64), 1, 1];

    // === 2. LOOP: Reuse the same buffer and descriptor sets ===
    for iter in 0..ITERATIONS {
        println!("\n=== Iteration {} ===", iter);

        // Build a fresh command buffer each iteration
        let mut builder = AutoCommandBufferBuilder::primary(
            cmd_allocator.clone(),
            queue_family,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // === 3. BARRIERS ===
        // AutoCommandBufferBuilder automatically inserts pipeline barriers between
        // commands that read/write the same buffer. No manual barrier needed.
        unsafe {
            builder
                .bind_pipeline_compute(pipeline1.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline1.layout().clone(),
                    0,
                    set1.clone(),
                )
                .unwrap()
                .dispatch(groups)
                .unwrap()
                .bind_pipeline_compute(pipeline2.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline2.layout().clone(),
                    0,
                    set2.clone(),
                )
                .unwrap()
                .dispatch(groups)
                .unwrap()
                .bind_pipeline_compute(pipeline3.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline3.layout().clone(),
                    0,
                    set3.clone(),
                )
                .unwrap()
                .dispatch(groups)
                .unwrap()
                .bind_pipeline_compute(pipeline4.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline4.layout().clone(),
                    0,
                    set4.clone(),
                )
                .unwrap()
                .dispatch(groups)
                .unwrap()
                .bind_pipeline_compute(pipeline5.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline5.layout().clone(),
                    0,
                    set5.clone(),
                )
                .unwrap()
                .dispatch(groups)
                .unwrap();
        }

        let command_buffer = builder.build().unwrap();

        // Submit and fence
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // Wait for GPU to finish before next iteration
        future.wait(None).unwrap();

        // Read back (scoped guard!)
        {
            let content = shared_buffer.read().unwrap();
            println!(
                "  First 5: {:?}",
                content[..5].iter().map(|d| d.value).collect::<Vec<_>>()
            );
            println!(
                "  Last  5: {:?}",
                content[N - 5..].iter().map(|d| d.value).collect::<Vec<_>>()
            );
        }

        // Mutate on CPU between iterations to prove buffer reuse
        if iter == 0 {
            let mut write = shared_buffer.write().unwrap();
            write[0].value = 1000.0;
            println!("  Overwrote index 0 to 1000.0 for next iteration");
        }
    }

    println!("\nFinal readback:");
    {
        let content = shared_buffer.read().unwrap();
        println!("  Index 0 = {}", content[0].value);
    }
}
