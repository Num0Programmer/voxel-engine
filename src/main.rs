use winit::{event_loop::EventLoop, window::WindowBuilder};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    image::ImageUsage,
    instance::{Instance, InstanceCreateInfo, InstanceCreateFlags},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendState, ColorBlendAttachmentState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            viewport::{Viewport, ViewportState},
            vertex_input::{Vertex, VertexDefinition},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState,
        GraphicsPipeline,
        PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanLibrary
};


pub fn main()
{
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new()
        .build(&event_loop)
        .unwrap_or_else(|err| panic!("Could not build window: {:?}", err))
    );

    let library = VulkanLibrary::new()
        .unwrap_or_else(|err| panic!("Could not load Vulkan Library: {:?}", err));
    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo
        {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        })
        .unwrap_or_else(|err| panic!("Could not create instance: {:?}", err));
    
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let dev_extensions = DeviceExtensions
    {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let physical_device = instance.enumerate_physical_devices()
        .unwrap_or_else(|err| panic!("Could not enumerate physical devices: {:?}", err))
        .next().expect("No physical device!");
    let queue_family_index = 0;  // will be set dynamically as I learn the API

    println!("Physical device:\n{} (type: {:?})\n",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo
        {
            enabled_extensions: dev_extensions,
            queue_create_infos: vec![QueueCreateInfo
            {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    ).unwrap();

    let queue = queues.next().unwrap();

    println!("Logical device: {:?}\n", device);
    println!("Queue family: {:?}\n", queue);

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo
            {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        ).unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    
    
    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex
    {
        #[format(R32G32B32_SFLOAT)]
        position: [f32; 2]
    }
    let vertices = [
        Vertex
        {
            position: [-0.5, -0.25]
        },
        Vertex
        {
            position: [0.0, 0.5]
        },
        Vertex
        {
            position: [0.25, -0.1]
        }
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo
        {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo
        {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices
    ).unwrap();

    mod vs
    {
        vulkano_shaders::shader!
        {
            ty: "vertex",
            src: r"
                #version 450

                layout (location = 0) in vec2 position;

                void main()
                {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            "
        }
    }
    mod fs
    {
        vulkano_shaders::shader!
        {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main()
                {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            "
        }
    }

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments:
        {
            color:
            {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap();

    let pipeline = {
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs)
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        ).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo
            {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default()
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            }
        ).unwrap()
    };

    let mut viewport = Viewport
    {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0
    };
}
