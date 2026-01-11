use wgpu::Instance;

fn main() {
    let instance = Instance::default();
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    println!("Available adapters:");
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("{}: {:#?}", i, info);
    }
}
