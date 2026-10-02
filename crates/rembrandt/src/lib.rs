// SPDX-License-Identifier: MIT

pub use bytemuck;
pub use vulkano;
pub use vulkano_shaders;

use std::any::Any;
use std::sync::Arc;
use vulkano::{
    DeviceSize, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::ShaderModule,
};

/// A trait object hiding a `Vec<T>` of any Pod type. We erase `T` down to a
/// common type so that vectors of different types can sit in one HashMap, but in
/// a way that `T` can later be recovered through an `Any` downcast with a runtime check.
///
/// The `Any` supertrait is mandatory: without it `self` in `as_any` could not be coerced to
/// `&dyn Any`, and `downcast_ref` lives precisely on `dyn Any`, not on our trait.
pub trait AnyVec: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
    /// Companion to `len` — clippy rightly complains about a trait with only `len`.
    /// Default body: implementations have nothing to override.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// `Pod` does not have `Send`/`Sync` as supertraits (its chain is `Zeroable + Copy +
// 'static`), so the auto traits have to be required explicitly. For plain-byte
// Pod types this is always satisfiable, but now the invariant is checked by the compiler rather
// than by an unconditional `unsafe impl`.
impl<T: bytemuck::Pod + Send + Sync> AnyVec for Vec<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T: BufferContents> AnyVec for Subbuffer<[T]> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn len(&self) -> usize {
        Subbuffer::len(self) as usize
    }
}

pub struct GpuAccelerator {
    cmd_allocator: Arc<StandardCommandBufferAllocator>,
    ds_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    queue: Arc<Queue>,
    device: Arc<Device>,
}

impl Default for GpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuAccelerator {
    pub const LOCAL_SIZE_X: u32 = 64;

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn new() -> Self {
        let library = VulkanLibrary::new().expect("no Vulkan library");
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

        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        Self {
            device,
            queue,
            memory_allocator,
            ds_allocator,
            cmd_allocator,
        }
    }

    pub fn allocate_buffer<T>(&self, count: usize) -> Subbuffer<[T]>
    where
        T: BufferContents,
    {
        Buffer::new_slice::<T>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            count as DeviceSize,
        )
        .expect("failed to create buffer")
    }

    // TODO: determine the shader thread count properly
    fn gpu_in_flight_impl(&self) -> usize {
        let p = self.device.physical_device().properties();

        // AMD: VK_AMD_shader_core_properties
        let amd = (|| {
            Some(
                p.shader_engine_count? as usize
                    * p.shader_arrays_per_engine_count? as usize
                    * p.compute_units_per_shader_array? as usize
                    * p.simd_per_compute_unit? as usize
                    * p.wavefronts_per_simd? as usize
                    * p.wavefront_size? as usize,
            )
        })();

        // NVIDIA: VK_NV_shader_sm_builtins
        let nv = (|| {
            Some(
                p.shader_sm_count? as usize * p.shader_warps_per_sm? as usize * 32usize, // warp size
            )
        })();

        amd.or(nv).unwrap_or(1 << 16) // fallback ~64K if neither extension is available
    }

    pub fn gpu_in_flight(&self) -> usize {
        const HARD_CAP: usize = 1 << 20; // the table is never larger than 1M rows
        self.gpu_in_flight_impl().next_power_of_two().min(HARD_CAP)
    }

    pub fn build_pipeline(&self, shader: &Arc<ShaderModule>, entry: &str) -> Arc<ComputePipeline> {
        let cs = shader.entry_point(entry).unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();

        ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
    }

    pub fn ds_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.ds_allocator
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Command-buffer allocator for one-shot compute submissions. `rembrandt`
    /// supplies the allocator; assembling and submitting the command buffer is
    /// the caller's job.
    pub fn cmd_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.cmd_allocator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_accelerator_creation() {
        GpuAccelerator::new();
    }
}
