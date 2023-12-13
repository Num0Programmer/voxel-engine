use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    device::physical::PhysicalDevice,
    VulkanLibrary
};


pub fn main()
{
    let library = VulkanLibrary::new()
        .unwrap_or_else(|err|
            panic!("Could not load Vulkan library: {:?}", err)
        );
    let extensions = InstanceExtensions
    {
        khr_surface: true,
        ext_metal_surface: true,
        ..InstanceExtensions::empty()
    };
    let instance = Instance::new(
        library.clone(),  // expensive - will not be included after setting up vulkan stage
        InstanceCreateInfo
        {
            enabled_extensions: extensions,
            ..Default::default()
        }
    ).unwrap_or_else(|err| panic!("Could not create Vulkan Instance: {:?}", err));

    // show user their vk library, extensions, and instance
    println!("Loaded library: {:?}\n", library);
    println!("Extensions: {:?}\n", extensions);
    println!("Instance: {:?}\n", instance);
    // show user available devices for VK
    println!("Physical devices found:");
    for physical_device in instance
        .enumerate_physical_devices()
        .unwrap_or_else(|err| panic!("Could not get physical device: {:?}", err))
    {
        println!("Available device: {:?}", physical_device.properties().device_name);
        println!("                  {:?}", physical_device.properties().device_id);
    }
}
