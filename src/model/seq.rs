use burn::{
    Tensor,
    module::{AutodiffModule, Module},
    prelude::*,
    tensor::{BasicOps, TensorKind, backend::AutodiffBackend},
};

use crate::model::Config;

pub trait SequenceModelConfig: Config {
    fn model_dim(&self) -> usize;
}

pub trait SequenceModel<B: Backend>: Module<B> {
    type State;
    fn init_state(&self, batch_size: usize, device: &B::Device) -> Self::State;
    fn forward_sequence(
        &self,
        xs: SequenceTensor<B, 3>,
        state: Self::State,
    ) -> (SequenceTensor<B, 3>, Self::State);
}

pub trait AutodiffSequenceModel<B: AutodiffBackend>:
    SequenceModel<B> + AutodiffModule<B, InnerModule: SequenceModel<B::InnerBackend>>
{
}
impl<
    B: AutodiffBackend,
    R: SequenceModel<B> + AutodiffModule<B, InnerModule: SequenceModel<B::InnerBackend>>,
> AutodiffSequenceModel<B> for R
{
}

impl<B: Backend, M: SequenceModel<B>> SequenceModel<B> for Vec<M> {
    type State = Vec<M::State>;

    fn init_state(&self, batch_size: usize, device: &<B as Backend>::Device) -> Self::State {
        self.iter()
            .map(|m| m.init_state(batch_size, device))
            .collect()
    }

    fn forward_sequence(
        &self,
        xs: SequenceTensor<B, 3>,
        states: Self::State,
    ) -> (SequenceTensor<B, 3>, Self::State) {
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

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SequenceTensor<B: Backend, const D: usize, K: TensorKind<B> = Float> {
    pub data: Tensor<B, D, K>,
    pub mask: Tensor<B, 2, Bool>,
}

impl<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>> SequenceTensor<B, D, K> {
    pub fn new(data: Tensor<B, D, K>, mask: Tensor<B, 2, Bool>) -> Self {
        assert_eq!(data.dims()[..2], mask.dims());
        Self { data, mask }
    }
}
