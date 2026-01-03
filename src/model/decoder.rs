use burn::{
    Tensor,
    module::{AutodiffModule, Module},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};

use crate::model::Config;

pub trait DecoderConfig: Config {
    fn model_dim(&self) -> usize;
}

pub trait Decoder<B: Backend>: Module<B> {
    type State;
    fn init_state(&self, batch_size: usize, device: &B::Device) -> Self::State;
    fn forward_sequence(&self, xs: Tensor<B, 3>, state: Self::State)
    -> (Tensor<B, 3>, Self::State);
}

pub trait AutodiffDecoder<B: AutodiffBackend>:
    Decoder<B> + AutodiffModule<B, InnerModule: Decoder<B::InnerBackend>>
{
}
impl<B: AutodiffBackend, R: Decoder<B> + AutodiffModule<B, InnerModule: Decoder<B::InnerBackend>>>
    AutodiffDecoder<B> for R
{
}

impl<B: Backend, M: Decoder<B>> Decoder<B> for Vec<M> {
    type State = Vec<M::State>;

    fn init_state(&self, batch_size: usize, device: &<B as Backend>::Device) -> Self::State {
        self.iter()
            .map(|m| m.init_state(batch_size, device))
            .collect()
    }

    fn forward_sequence(
        &self,
        xs: Tensor<B, 3>,
        states: Self::State,
    ) -> (Tensor<B, 3>, Self::State) {
        let mut hs = xs;
        let mut new_states = Vec::with_capacity(self.len());
        for (m, s) in self.iter().zip(states) {
            let (hs_new, s_new) = m.forward_sequence(hs, s);
            hs = hs_new;
            new_states.push(s_new);
        }
        (hs, new_states)
    }
}
