use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    device::{Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo},
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
    let physical_device = instance
        .enumerate_physical_devices()
        .unwrap_or_else(|err| panic!("Could not get physical device: {:?}", err))
        .next().expect("No physical device!");


    // choose first physical device found
    let device = {
        let dev_features = Features::empty();
        let dev_extensions = DeviceExtensions::empty();

        match Device::new(
            physical_device,
            DeviceCreateInfo
            {
                queue_create_infos: vec![QueueCreateInfo
                {
                    queue_family_index: 0,
                    ..Default::default()
                }],
                enabled_extensions: dev_extensions,
                enabled_features: dev_features,
                ..Default::default()
            },
        ) {
            Ok(d) => d,
            Err(err) => panic!("Couldn't build device: {:?}", err)
        }
    };

    // show user the extensions
    println!("Extensions: {:?}\n", extensions);
    // show user the created device
    println!("Logical device: {:?}", device.0);
}
