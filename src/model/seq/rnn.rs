use burn::{
    nn::{
        Linear, LinearConfig, Lstm, LstmConfig, LstmState as LstmStateInner,
        gru::{Gru, GruConfig},
    },
    prelude::*,
};

use crate::{
    model::{
        ModelConfig,
        seq::{SequenceModel, SequenceModelConfig},
    },
    util::SeqTensor,
};

fn forward_multiple_steps<B: Backend, R: Rnn<B>>(
    rnn: &R,
    xs: SeqTensor<B, 3>,
    h_init: R::State,
) -> (SeqTensor<B, 3>, R::State) {
    let mut ys = Vec::with_capacity(xs.tensor().dims()[1]);
    let mut h = h_init;
    // FIXME: support masking
    for x in xs
        .tensor()
        .clone()
        .split(1, 1)
        .into_iter()
        .map(|x| x.squeeze_dim(1))
    {
        let (y, h_next) = rnn.forward_step(x, h);
        h = h_next;
        ys.push(y.unsqueeze_dim(1));
    }
    let ys = xs.replace(Tensor::cat(ys, 1));
    (ys, h)
}

pub trait Rnn<B: Backend>: SequenceModel<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State);
}

#[derive(Config, Debug)]
pub struct NaiveConfig {
    pub d_hidden: usize,
}

#[derive(Module, Debug)]
pub struct Naive<B: Backend> {
    pub hh: Linear<B>,
}

impl ModelConfig for NaiveConfig {
    type Model<B: Backend> = Naive<B>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        Naive {
            hh: LinearConfig::new(self.d_hidden, self.d_hidden).init(device),
        }
    }
}

impl<B: Backend> SequenceModel<B> for Naive<B> {
    type State = Tensor<B, 2>;

    fn init_state(&self, batch_size: usize) -> Self::State {
        Tensor::zeros(
            [batch_size, self.hh.weight.dims()[0]],
            &self.hh.weight.device(),
        )
    }

    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        state: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State) {
        forward_multiple_steps(self, xs, state)
    }
}

impl<B: Backend> Rnn<B> for Naive<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        let h_next = self.hh.forward(x + h).tanh();
        (h_next.clone(), h_next)
    }
}

#[derive(Clone, Debug)]
pub struct LstmState<B: Backend, const D: usize> {
    cell: Tensor<B, D>,
    hidden: Tensor<B, D>,
}

impl<B: Backend, const D: usize> From<LstmStateInner<B, D>> for LstmState<B, D> {
    fn from(inner: LstmStateInner<B, D>) -> Self {
        Self {
            cell: inner.cell,
            hidden: inner.hidden,
        }
    }
}

impl<B: Backend, const D: usize> From<LstmState<B, D>> for LstmStateInner<B, D> {
    fn from(inner: LstmState<B, D>) -> Self {
        Self {
            cell: inner.cell,
            hidden: inner.hidden,
        }
    }
}

impl ModelConfig for LstmConfig {
    type Model<B: Backend> = Lstm<B>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        self.init(device)
    }
}

impl<B: Backend> SequenceModel<B> for Lstm<B> {
    type State = LstmState<B, 2>;

    fn init_state(&self, batch_size: usize) -> Self::State {
        let device = self.input_gate.input_transform.weight.device();
        LstmState {
            cell: Tensor::zeros([batch_size, self.d_hidden], &device),
            hidden: Tensor::zeros([batch_size, self.d_hidden], &device),
        }
    }

    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        h_init: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State) {
        if xs.tensor().dims()[1] == 0 {
            return (xs, h_init);
        }
        let (ys, h_last) = self.forward(xs.tensor().clone(), Some(h_init.into()));
        // FIXME: select last state according to mask
        (xs.replace(ys), h_last.into())
    }
}

impl<B: Backend> Rnn<B> for Lstm<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        let (y, h_next) = self.forward(x.unsqueeze_dim(1), Some(h.into()));
        (y.squeeze_dim(1), h_next.into())
    }
}

impl ModelConfig for GruConfig {
    type Model<B: Backend> = Gru<B>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        self.init(device)
    }
}

impl<B: Backend> SequenceModel<B> for Gru<B> {
    type State = Tensor<B, 2>;

    fn init_state(&self, batch_size: usize) -> Self::State {
        Tensor::zeros(
            [batch_size, self.d_hidden],
            &self.new_gate.input_transform.weight.device(),
        )
    }

    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        h_init: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State) {
        if xs.tensor().dims()[1] == 0 {
            return (xs, h_init);
        }
        let h_all = self.forward(xs.tensor().clone(), Some(h_init.clone()));
        let h_last = Tensor::cat(vec![h_init.unsqueeze_dim(1), h_all.clone()], 1)
            .select(
                1,
                Tensor::from_data(xs.seq_lengths(), &xs.tensor().device()),
            )
            .squeeze_dim(1);
        (xs.replace(h_all), h_last)
    }
}

impl<B: Backend> Rnn<B> for Gru<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        let h_all = self.forward(x.unsqueeze_dim(1), Some(h));
        let h_next = h_all.squeeze_dim(1);
        (h_next.clone(), h_next)
    }
}

#[derive(Config, Debug)]
pub enum SomeRnnConfig {
    Naive(NaiveConfig),
    Lstm(LstmConfig),
    Gru(GruConfig),
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum SomeRnn<B: Backend> {
    Naive(Naive<B>),
    Lstm(Lstm<B>),
    Gru(Gru<B>),
}

#[derive(Clone, Debug)]
pub enum SomeRnnState<B: Backend> {
    Naive(Tensor<B, 2>),
    Lstm(LstmState<B, 2>),
    Gru(Tensor<B, 2>),
}

impl ModelConfig for SomeRnnConfig {
    type Model<B: Backend> = SomeRnn<B>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        match self {
            Self::Naive(c) => SomeRnn::Naive(c.init_model(device)),
            Self::Lstm(c) => SomeRnn::Lstm(c.init_model(device)),
            Self::Gru(c) => SomeRnn::Gru(c.init_model(device)),
        }
    }
}

impl SequenceModelConfig for SomeRnnConfig {
    fn model_dim(&self) -> usize {
        match self {
            Self::Naive(c) => c.d_hidden,
            Self::Lstm(c) => c.d_hidden,
            Self::Gru(c) => c.d_hidden,
        }
    }
}

impl<B: Backend> SequenceModel<B> for SomeRnn<B> {
    type State = SomeRnnState<B>;

    fn init_state(&self, batch_size: usize) -> Self::State {
        match self {
            Self::Naive(m) => Self::State::Naive(m.init_state(batch_size)),
            Self::Lstm(m) => Self::State::Lstm(m.init_state(batch_size)),
            Self::Gru(m) => Self::State::Gru(m.init_state(batch_size)),
        }
    }

    fn forward_sequence(
        &self,
        xs: SeqTensor<B, 3>,
        h_init: Self::State,
    ) -> (SeqTensor<B, 3>, Self::State) {
        match (self, h_init) {
            (Self::Naive(m), Self::State::Naive(h)) => {
                let (ys, h_new) = m.forward_sequence(xs, h);
                (ys, Self::State::Naive(h_new))
            }
            (Self::Lstm(m), Self::State::Lstm(h)) => {
                let (ys, h_new) = m.forward_sequence(xs, h);
                (ys, Self::State::Lstm(h_new))
            }
            (Self::Gru(m), Self::State::Gru(h)) => {
                let (ys, h_new) = m.forward_sequence(xs, h);
                (ys, Self::State::Gru(h_new))
            }
            _ => panic!("Model and state mismatch"),
        }
    }
}

impl<B: Backend> Rnn<B> for SomeRnn<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        match (self, h) {
            (Self::Naive(m), Self::State::Naive(h)) => {
                let (y, h_new) = m.forward_step(x, h);
                (y, Self::State::Naive(h_new))
            }
            (Self::Lstm(m), Self::State::Lstm(h)) => {
                let (y, h_new) = m.forward_step(x, h);
                (y, Self::State::Lstm(h_new))
            }
            (Self::Gru(m), Self::State::Gru(h)) => {
                let (y, h_new) = m.forward_step(x, h);
                (y, Self::State::Gru(h_new))
            }
            _ => panic!("Model and state mismatch"),
        }
    }
}
