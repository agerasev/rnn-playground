use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::model::{
    Config,
    decoder::{Decoder, DecoderConfig},
};

#[derive(Debug)]
pub struct LmConfig<M: DecoderConfig> {
    pub vocab_size: usize,
    pub decoder: M,
}

#[derive(Module, Debug)]
pub struct Lm<B: Backend, M: Module<B>> {
    pub emb: Embedding<B>,
    pub decoder: M,
    pub lm_head: Linear<B>,
}

impl<M: DecoderConfig> Config for LmConfig<M> {
    type Model<B: Backend> = Lm<B, M::Model<B>>;

    fn init_model<B: Backend>(&self, device: &B::Device) -> Self::Model<B> {
        Lm {
            emb: EmbeddingConfig::new(self.vocab_size, self.decoder.model_dim()).init(device),
            decoder: self.decoder.init_model(device),
            lm_head: LinearConfig::new(self.decoder.model_dim(), self.vocab_size).init(device),
        }
    }
}

impl<B: Backend, M: Decoder<B>> Lm<B, M> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, state: M::State) -> (Tensor<B, 3>, M::State) {
        let x = self.emb.forward(tokens);
        let (h, new_state) = self.decoder.forward_sequence(x, state);
        let y = self.lm_head.forward(h);
        (y, new_state)
    }
}
