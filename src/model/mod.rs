use burn::{
    module::{Module, ModuleDisplay},
    prelude::Backend,
};

pub mod decoder;
pub mod lm;
pub mod rnn;

pub trait Config {
    type Model<B: Backend>: Module<B> + ModuleDisplay;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B>;
}
