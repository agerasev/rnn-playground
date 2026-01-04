use burn::{
    nn::{
        Linear, LinearConfig, Lstm, LstmConfig, LstmState,
        gru::{Gru, GruConfig},
    },
    prelude::*,
};

use crate::model::{
    Config,
    seq::{SequenceModel, SequenceTensor},
};

fn forward_multiple_steps<B: Backend, R: Rnn<B>>(
    rnn: &R,
    xs: SequenceTensor<B, 3>,
    h_init: R::State,
) -> (SequenceTensor<B, 3>, R::State) {
    let mut ys = Vec::with_capacity(xs.data.dims()[1]);
    let mut h = h_init;
    // FIXME: support masking
    for x in xs.data.split(1, 1).into_iter().map(|x| x.squeeze_dim(1)) {
        let (y, h_next) = rnn.forward_step(x, h);
        h = h_next;
        ys.push(y.unsqueeze_dim(1));
    }
    let ys = SequenceTensor::new(Tensor::cat(ys, 1), xs.mask);
    (ys, h)
}

pub trait Rnn<B: Backend>: SequenceModel<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State);
}

#[derive(Config, Debug)]
pub struct RnnConfig {
    pub d_hidden: usize,
}

#[derive(Module, Debug)]
pub struct Naive<B: Backend> {
    pub hh: Linear<B>,
}

impl Config for RnnConfig {
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
        xs: SequenceTensor<B, 3>,
        state: Self::State,
    ) -> (SequenceTensor<B, 3>, Self::State) {
        forward_multiple_steps(self, xs, state)
    }
}

impl<B: Backend> Rnn<B> for Naive<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        let h_next = self.hh.forward(x + h).tanh();
        (h_next.clone(), h_next)
    }
}

impl Config for LstmConfig {
    type Model<B: Backend> = Lstm<B>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        self.init(device)
    }
}

impl<B: Backend> SequenceModel<B> for Lstm<B> {
    type State = LstmState<B, 2>;

    fn init_state(&self, batch_size: usize) -> Self::State {
        let device = self.input_gate.input_transform.weight.device();
        LstmState::new(
            Tensor::zeros([batch_size, self.d_hidden], &device),
            Tensor::zeros([batch_size, self.d_hidden], &device),
        )
    }

    fn forward_sequence(
        &self,
        xs: SequenceTensor<B, 3>,
        h_init: Self::State,
    ) -> (SequenceTensor<B, 3>, Self::State) {
        if xs.data.dims()[1] == 0 {
            return (xs, h_init);
        }
        let (ys, h_last) = self.forward(xs.data, Some(h_init));
        // FIXME: select last state according to mask
        (SequenceTensor::new(ys, xs.mask), h_last)
    }
}

impl<B: Backend> Rnn<B> for Lstm<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        let (y, h_next) = self.forward(x.unsqueeze_dim(1), Some(h));
        (y.squeeze_dim(1), h_next)
    }
}

impl Config for GruConfig {
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
        xs: SequenceTensor<B, 3>,
        h_init: Self::State,
    ) -> (SequenceTensor<B, 3>, Self::State) {
        if xs.data.dims()[1] == 0 {
            return (xs, h_init);
        }
        let h_all = self.forward(xs.data, Some(h_init.clone()));
        // FIXME: support masks other than [1, ..., 1, 0, ..., 0]
        let h_last = Tensor::cat(vec![h_init.unsqueeze_dim(1), h_all.clone()], 1)
            .select(1, xs.mask.clone().int().sum_dim(1).squeeze_dim(1))
            .squeeze_dim(1);
        (SequenceTensor::new(h_all, xs.mask), h_last)
    }
}

impl<B: Backend> Rnn<B> for Gru<B> {
    fn forward_step(&self, x: Tensor<B, 2>, h: Self::State) -> (Tensor<B, 2>, Self::State) {
        let h_all = self.forward(x.unsqueeze_dim(1), Some(h));
        let h_next = h_all.squeeze_dim(1);
        (h_next.clone(), h_next)
    }
}
