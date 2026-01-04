use burn::{
    module::{Module, ModuleDisplay},
    prelude::*,
};

pub mod lm;
pub mod rnn;
pub mod seq;

pub trait Config {
    type Model<B: Backend>: Module<B> + ModuleDisplay;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B>;
}
