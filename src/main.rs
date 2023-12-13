use winit::{event_loop::EventLoop, window::WindowBuilder};
use std::sync::Arc;
use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceCreateFlags},
    swapchain::Surface,
    VulkanLibrary
};


pub fn main()
{
    // define loop for winit events
    let event_loop = EventLoop::new();
    // setup vulkan instance and choose a physical device
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
    let physical_device = instance
        .enumerate_physical_devices()
        .unwrap_or_else(|err| panic!("Could not enumerate physical devices: {:?}", err))
        .next().expect("No physical device!");

    // setup window and surface onto which to draw
    let window = Arc::new(WindowBuilder::new()
        .build(&event_loop)
        .unwrap_or_else(|err| panic!("Could not build window: {:?}", err)));
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    println!("Created surface: {:?}", surface);
}
