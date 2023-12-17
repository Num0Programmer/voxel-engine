use winit::{event_loop::EventLoop, window::WindowBuilder};
use std::sync::Arc;
use vulkano::{
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    instance::{Instance, InstanceCreateInfo, InstanceCreateFlags},
    swapchain::Surface,
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
}
