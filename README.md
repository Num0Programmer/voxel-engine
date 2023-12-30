# voxel-engine - tentative
Rusty graphics engine which utilizes voxels to display 3-dimensional geometry.

## Software
- [Vulkano](https://crates.io/crates/vulkano)
- [winit](https://crates.io/crates/winit)

## Examples
Moving a triangle by modifying vertex positions on the CPU:
```sh
cargo run --example tri
```
Moving a triangle by modifying vertex positions using a compute shader:
```sh
cargo run --example tri_cs
```
