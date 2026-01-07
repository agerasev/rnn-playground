pub mod rnn;
pub mod some;

use burn::{
    module::{AutodiffModule, Module, ModuleDisplay},
    prelude::*,
    tensor::backend::AutodiffBackend,
};

use crate::{model::ModelConfig, util::SeqTensor};

pub trait SequenceModelConfig: ModelConfig {
    fn model_dim(&self) -> usize;
}

pub trait SequenceModel<B: Backend>: Module<B> + ModuleDisplay {
    type State;
    fn init_state(&self, batch_size: usize) -> Self::State;
    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        state: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State);
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

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.iter().map(|m| m.init_state(batch_size)).collect()
    }

    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        states: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State) {
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
