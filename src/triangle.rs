use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder
};
use std::{time::SystemTime, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator,
        auto::AutoCommandBufferBuilder,
        CommandBufferUsage,
        RenderPassBeginInfo,
        SubpassBeginInfo,
        SubpassContents
    },
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateInfo, InstanceCreateFlags},
    memory::allocator::{
        AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendState, ColorBlendAttachmentState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            viewport::{Viewport, ViewportState},
            vertex_input::{Vertex, VertexDefinition},
            GraphicsPipelineCreateInfo
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState,
        GraphicsPipeline,
        PipelineLayout,
        PipelineShaderStageCreateInfo
    },
    render_pass::{
        Framebuffer,
        FramebufferCreateInfo,
        RenderPass,
        Subpass
    },
    swapchain::{
        acquire_next_image,
        Surface,
        Swapchain,
        SwapchainCreateInfo,
        SwapchainPresentInfo
    },
    sync::{self, GpuFuture},
    Validated,
    VulkanError,
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
        .unwrap_or_else(|err|
        {
            panic!("Could not load Vulkan Library: {:?}", err)
        });
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
    
    let surface = Surface::from_window(instance.clone(), window.clone())
        .unwrap();

    let dev_extensions = DeviceExtensions
    {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let physical_device = instance.enumerate_physical_devices()
        .unwrap_or_else(|err|
        {
            panic!("Could not enumerate physical devices: {:?}", err)
        }).next().expect("No physical device!");
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

    let memory_allocator = Arc::new(
        StandardMemoryAllocator::new_default(device.clone())
    );
    
    
    #[derive(BufferContents, Clone, Vertex)]
    #[repr(C)]
    struct Vertex
    {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2]
    }
    let mut verticies = [
        Vertex
        {
            position: [0.0, -0.5]
        },
        Vertex
        {
            position: [0.5, 0.5]
        },
        Vertex
        {
            position: [-0.5, 0.5]
        }
    ];

    mod vs
    {
        vulkano_shaders::shader!
        {
            ty: "vertex",
            src: r"
                #version 450

                vec3 colors[3] = vec3[](
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 0.0, 1.0)
                );

                layout (location = 0) in vec2 position;
                layout (location = 0) out vec3 f_color;

                void main()
                {
                    gl_Position = vec4(position, 0.0, 1.0);
                    f_color = colors[gl_VertexIndex];
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

                layout (location = 0) in vec3 f_color;
                layout (location = 0) out vec4 o_color;

                void main()
                {
                    o_color = vec4(f_color, 1.0);
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

    let graphics_pipeline = {
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
                color_blend_state: Some(
                    ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default()
                    )
                ),
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

    let mut framebuffers = window_size_dependent_setup(
        &images, render_pass.clone(), &mut viewport
    );

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(), Default::default()
    );

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let start_time = SystemTime::now();
    let mut last_frame_time = start_time;
    event_loop.run(move |event, _, control_flow|
    {
        match event
        {
            Event::WindowEvent
            {
                event: WindowEvent::CloseRequested,
                ..
            } =>
            {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent
            {
                event: WindowEvent::Resized(_),
                ..
            } =>
            {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared =>
            {
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0)
                {
                    return;
                }

                let _now = SystemTime::now();
                let _time = _now
                    .duration_since(start_time)
                    .unwrap()
                    .as_secs_f32();
                let _dt = _now
                    .duration_since(last_frame_time)
                    .unwrap()
                    .as_secs_f32();
                last_frame_time = _now;

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain
                {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo
                        {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("Failed to recreate swapchain");

                    swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport
                    );

                    recreate_swapchain = false;
                }
            
                let vertex_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
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
                    verticies.clone()
                )
                .unwrap();

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None)
                        .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) =>
                        {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {e}")
                    };

                if suboptimal
                {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo
                        {
                            clear_values: vec![
                                Some([0.0, 0.0, 1.0, 1.0].into())
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone()
                            )
                        },
                        SubpassBeginInfo
                        {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(graphics_pipeline.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            swapchain.clone(), image_index
                        )
                    )
                    .then_signal_fence_and_flush();

                verticies.iter_mut().for_each(|vertex|
                {
                    vertex.position[0] += 2.0 * _dt;
                    vertex.position[1] += 2.0 * _dt;
                });

                match future.map_err(Validated::unwrap)
                {
                    Ok(future) =>
                    {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) =>
                    {
                        recreate_swapchain = true;
                        previous_frame_end = Some(
                            sync::now(device.clone()).boxed()
                        );
                    }
                    Err(e) =>
                    {
                        panic!("Failed to flush future: {e}");
                    }
                }
            }
            _ => ()
        }
    });
}


fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport
) -> Vec<Arc<Framebuffer>>
{
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image|
        {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo
                {
                    attachments: vec![view],
                    ..Default::default()
                }
            ).unwrap()
        }).collect::<Vec<_>>()
}
