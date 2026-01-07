use burn::{
    module::{Module, ModuleDisplay},
    prelude::*,
};

pub mod lm;
pub mod seq;

pub trait ModelConfig {
    type Model<B: Backend>: Module<B> + ModuleDisplay;
    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B>;
}
