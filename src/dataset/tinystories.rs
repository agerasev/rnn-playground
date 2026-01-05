use std::{
    fs::File,
    io,
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::Result;
use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::{
    dataset::Dataset,
    tokenizer::{SpecialToken, Tokenizer},
    util::SeqTensor,
};

fn read_until_slice(reader: &mut impl BufRead, delim: &[u8]) -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    'outer: loop {
        reader.read_until(delim[0], &mut buffer)?;
        for byte in &delim[1..] {
            buffer.push(0);
            let last_pos = buffer.len() - 1;
            let slot = &mut buffer[last_pos..];
            reader.read_exact(slot)?;
            if slot[0] != *byte {
                continue 'outer;
            }
        }
        break;
    }
    buffer.truncate(buffer.len() - delim.len());
    Ok(buffer)
}

pub struct TinyStories {
    file: BufReader<File>,
}

impl TinyStories {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            file: BufReader::new(File::open(path)?),
        })
    }
}

impl Iterator for TinyStories {
    type Item = Result<String>;
    fn next(&mut self) -> Option<Self::Item> {
        match read_until_slice(&mut self.file, "<|endoftext|>".as_bytes()) {
            Ok(data) => Some(String::from_utf8(data).map_err(|e| e.into())),
            Err(err) => match err.kind() {
                io::ErrorKind::UnexpectedEof => None,
                _ => Some(Err(err.into())),
            },
        }
    }
}

impl Dataset for TinyStories {
    type Sample = String;
    type Batch<B: Backend> = SeqTensor<B, 2, Int>;
    type Batcher<B: Backend, T: Tokenizer> = TinyStoriesBatcher<T>;
}

pub struct TinyStoriesBatcher<T: Tokenizer> {
    pub max_length: usize,
    pub tokenizer: T,
}

impl<B: Backend, T: Tokenizer> Batcher<B, String, SeqTensor<B, 2, Int>> for TinyStoriesBatcher<T> {
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
