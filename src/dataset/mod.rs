pub mod tinystories;
pub mod util;

use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::{
    tokenizer::{SpecialToken, Tokenizer},
    util::SeqTensor,
};

#[derive(Clone)]
pub struct SeqBatcher<T: Tokenizer> {
    pub max_length: usize,
    pub tokenizer: T,
}

impl<B: Backend, T: Tokenizer> Batcher<B, String, SeqTensor<B, 2, Int>> for SeqBatcher<T> {
    fn batch(&self, items: Vec<String>, device: &<B as Backend>::Device) -> SeqTensor<B, 2, Int> {
        let tokens_list = items
            .into_iter()
            .map(|s| {
                [self.tokenizer.special_token(SpecialToken::Begin)]
                    .into_iter()
                    .chain(self.tokenizer.encode_all(&s))
                    .chain([self.tokenizer.special_token(SpecialToken::End)])
                    .collect()
            })
            .collect();
        let pad_token = self.tokenizer.special_token(SpecialToken::Pad);
        SeqTensor::from_tokens(tokens_list, pad_token, Some(self.max_length), device)
    }
}
