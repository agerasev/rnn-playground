pub mod tinystories;
pub mod util;

use burn::{data::dataloader::batcher::Batcher, prelude::*};
use tokenizers::Tokenizer;

use crate::{
    tokenizer::{SpecialToken, encode},
    util::SeqTensor,
};

#[derive(Clone)]
pub struct SeqBatcher<'a> {
    pub max_length: usize,
    pub tokenizer: &'a Tokenizer,
}

impl<B: Backend> Batcher<B, String, SeqTensor<B, 2, Int>> for SeqBatcher<'_> {
    fn batch(&self, items: Vec<String>, device: &<B as Backend>::Device) -> SeqTensor<B, 2, Int> {
        let tokens_list = items
            .into_iter()
            .map(|s| {
                [SpecialToken::Begin.get_id(self.tokenizer)]
                    .into_iter()
                    .chain(encode(self.tokenizer, &s))
                    .chain([SpecialToken::End.get_id(self.tokenizer)])
                    .collect()
            })
            .collect();
        let pad_token = SpecialToken::Pad.get_id(self.tokenizer);
        SeqTensor::from_tokens(tokens_list, pad_token, Some(self.max_length), device)
    }
}
