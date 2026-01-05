use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{activation::log_softmax, backend::AutodiffBackend},
    train::{TrainOutput, TrainStep, ValidStep},
};

use crate::{
    model::{
        Config,
        seq::{AutodiffSequenceModel, SequenceModel, SequenceModelConfig},
    },
    util::SeqTensor,
};

#[derive(Debug)]
pub struct LmConfig<M: SequenceModelConfig> {
    pub vocab_size: usize,
    pub decoder: M,
}

#[derive(Module, Debug)]
pub struct Lm<B: Backend, M: Module<B>> {
    pub emb: Embedding<B>,
    pub decoder: M,
    pub lm_head: Linear<B>,
}

impl<M: SequenceModelConfig> Config for LmConfig<M> {
    type Model<B: Backend> = Lm<B, M::Model<B>>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        Lm {
            emb: EmbeddingConfig::new(self.vocab_size, self.decoder.model_dim()).init(device),
            decoder: self.decoder.init_model(device),
            lm_head: LinearConfig::new(self.decoder.model_dim(), self.vocab_size).init(device),
        }
    }
}

impl<B: Backend, M: SequenceModel<B>> Lm<B, M> {
    pub fn init_state(&self, batch_size: usize) -> M::State {
        self.decoder.init_state(batch_size)
    }

    pub fn forward(
        &self,
        tokens: SeqTensor<B, 2, Int>,
        state: M::State,
    ) -> (SeqTensor<B, 3>, M::State) {
        let xs = tokens.map(|tensor| self.emb.forward(tensor));
        let (hs, new_state) = self.decoder.forward_sequence(xs, state);
        let ys = hs.map(|tensor| self.lm_head.forward(tensor));
        (ys, new_state)
    }

    pub fn forward_autoregressive(&self, tokens: SeqTensor<B, 2, Int>) -> AutoregressionOutput<B> {
        let (logits, _) = self.forward(tokens.clone(), self.init_state(tokens.batch_size()));
        let loss = autoregressive_cross_entropy_loss(tokens.clone(), logits.clone());
        AutoregressionOutput {
            tokens,
            logits,
            loss,
        }
    }
}

fn autoregressive_cross_entropy_loss<B: Backend>(
    tokens: SeqTensor<B, 2, Int>,
    logits: SeqTensor<B, 3>,
) -> Tensor<B, 1> {
    assert_eq!(tokens.seq_lengths(), logits.seq_lengths());
    let max_seq_length = tokens.max_seq_length();

    let logits = logits.tensor().clone().narrow(1, 0, max_seq_length - 1);
    let targets = tokens.tensor().clone().narrow(1, 1, max_seq_length - 1);
    let mask = tokens.mask().clone().narrow(1, 1, max_seq_length - 1);

    let softmax = log_softmax(logits, 2);
    let cross_entropy = softmax
        .gather(2, targets.clone().unsqueeze_dim(2))
        .squeeze_dim(2);
    let elems_count: usize = tokens
        .seq_lengths()
        .iter()
        .map(|n| n.saturating_sub(1))
        .sum();
    cross_entropy
        .mask_fill(mask.bool_not(), 0.0)
        .sum()
        .div_scalar(elems_count as f32)
        .neg()
}

pub struct AutoregressionOutput<B: Backend> {
    pub tokens: SeqTensor<B, 2, Int>,
    pub logits: SeqTensor<B, 3>,
    pub loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend, M: AutodiffSequenceModel<B>>
    TrainStep<SeqTensor<B, 2, Int>, AutoregressionOutput<B>> for Lm<B, M>
{
    fn step(&self, tokens: SeqTensor<B, 2, Int>) -> TrainOutput<AutoregressionOutput<B>> {
        let output = self.forward_autoregressive(tokens);
        TrainOutput::new(self, output.loss.clone().backward(), output)
    }
}

impl<B: Backend, M: SequenceModel<B>> ValidStep<SeqTensor<B, 2, Int>, AutoregressionOutput<B>>
    for Lm<B, M>
{
    fn step(&self, tokens: SeqTensor<B, 2, Int>) -> AutoregressionOutput<B> {
        self.forward_autoregressive(tokens)
    }
}
