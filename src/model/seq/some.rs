use burn::{config::Config, prelude::*};

use crate::{
    model::{
        ModelConfig,
        seq::{
            SequenceModel, SequenceModelConfig,
            rnn::{SomeRnn, SomeRnnConfig, SomeRnnState},
        },
    },
    util::SeqTensor,
};

#[derive(Config, Debug)]
pub enum SeqModelConfig {
    Rnn(SomeRnnConfig),
}

#[derive(Module, Debug)]
pub enum SeqModel<B: Backend> {
    Rnn(SomeRnn<B>),
}

pub enum SeqModelState<B: Backend> {
    Rnn(SomeRnnState<B>),
}

impl ModelConfig for SeqModelConfig {
    type Model<B: Backend> = SeqModel<B>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        match self {
            Self::Rnn(c) => Self::Model::Rnn(c.init_model(device)),
        }
    }
}

impl SequenceModelConfig for SeqModelConfig {
    fn model_dim(&self) -> usize {
        match self {
            Self::Rnn(c) => c.model_dim(),
        }
    }
}

impl<B: Backend> SequenceModel<B> for SeqModel<B> {
    type State = SeqModelState<B>;
    fn init_state(&self, batch_size: usize) -> Self::State {
        match self {
            Self::Rnn(m) => Self::State::Rnn(m.init_state(batch_size)),
        }
    }
    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        state: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State) {
        match (self, state) {
            (Self::Rnn(m), Self::State::Rnn(s)) => {
                let (ys, s_new) = m.forward_sequence(xs, s);
                (ys, Self::State::Rnn(s_new))
            }
        }
    }
}
