use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig},
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, ValidStep},
};

use crate::model::{
    Config,
    seq::{AutodiffSequenceModel, SequenceModel, SequenceModelConfig, SequenceTensor},
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
        tokens: SequenceTensor<B, 2, Int>,
        state: M::State,
    ) -> (SequenceTensor<B, 3>, M::State) {
        let x = SequenceTensor::new(self.emb.forward(tokens.data), tokens.mask);
        let (h, new_state) = self.decoder.forward_sequence(x, state);
        let y = SequenceTensor::new(self.lm_head.forward(h.data), h.mask);
        (y, new_state)
    }
}

impl<B: AutodiffBackend, M: AutodiffSequenceModel<B>>
    TrainStep<SequenceTensor<B, 2, Int>, SequenceTensor<B, 3>> for Lm<B, M>
{
    fn step(&self, batch: SequenceTensor<B, 2, Int>) -> TrainOutput<SequenceTensor<B, 3>> {
        let [batch_size, max_seq_len] = batch.mask.dims();
        let (output, _) = self.forward(batch.clone(), self.init_state(batch_size));
        let logits = output.data.narrow(1, 0, max_seq_len - 1);
        let target = batch.data.narrow(1, 1, max_seq_len - 1);
        let mask = batch.mask.narrow(1, 1, max_seq_len - 1);
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(unimplemented!("masked indices"))
            .init(&batch.data.device())
            .forward(logits.flatten(0, 1), target.flatten(0, 1));
        TrainOutput::new(self, loss.backward(), output)
    }
}
