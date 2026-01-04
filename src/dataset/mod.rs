use anyhow::Result;
use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::tokenizer::Tokenizer;

pub mod tinystories;

pub trait Dataset: Iterator<Item = Result<Self::Sample>> {
    type Sample;
    type Batch<B: Backend>;
    type Batcher<B: Backend, T: Tokenizer>: Batcher<B, Self::Sample, Self::Batch<B>>;
}
