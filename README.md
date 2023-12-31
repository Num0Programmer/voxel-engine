# voxel-engine - tentative
Rusty graphics engine which utilizes voxels to display 3-dimensional geometry.

## Dependencies
Before running the following examples, please make sure you have the following
dependencies:

### Linux
Details to come

### MacOS
For Mac users, you will need MoltenVK to interact with Apple's Metal Framework.
The easiest way is to install the
[molten-vk](https://formulae.brew.sh/formula/molten-vk#default) package using
the Homebrew package manager:
```sh
brew install molten-vk
```

## Examples
Moving a triangle by modifying vertex positions on the CPU:
```sh
cargo run --example tri
```
Moving a triangle by modifying vertex positions using a compute shader:
```sh
cargo run --example tri_cs
```

## Software
- [Vulkano](https://crates.io/crates/vulkano)
- [winit](https://crates.io/crates/winit)
