use std::{fs, path::Path};

use tokenizers::{Result, TokenizerImpl};

pub use tokenizers::Tokenizer;

use crate::tokenizer::SpecialToken;

impl super::Tokenizer for Tokenizer {
    type Encoder<'a> = Encoder<'a>;
    type Decoder<'a> = Decoder<'a>;

    fn vocab_size(&self) -> usize {
        self.get_vocab_size(true)
    }

    fn encode_all(&self, text: &str) -> Vec<usize> {
        TokenizerImpl::encode(self, text, false)
            .unwrap()
            .get_ids()
            .iter()
            .map(|t| *t as usize)
            .collect()
    }

    fn decode_all<I: IntoIterator<Item = usize>>(&self, tokens: I) -> String {
        TokenizerImpl::decode(
            self,
            &tokens.into_iter().map(|t| t as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap()
    }

    fn encoder(&self) -> Self::Encoder<'_> {
        Encoder {
            tokenizer: self,
            text: String::new(),
            tokens: Vec::new(),
        }
    }

    fn decoder(&self) -> Self::Decoder<'_> {
        Decoder {
            tokenizer: self,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    fn special_token(&self, t: SpecialToken) -> usize {
        use SpecialToken::*;
        *self
            .get_vocab(true)
            .get(match t {
                Begin => "<s>",
                End => "</s>",
                Pad => "<pad>",
                Unknown => "<unk>",
            })
            .unwrap() as usize
    }
}

#[derive(Clone)]
pub struct Encoder<'a> {
    tokenizer: &'a Tokenizer,
    text: String,
    tokens: Vec<usize>,
}

impl<'a> super::Encoder for Encoder<'a> {
    fn encode_next(&mut self, text: &str) -> Vec<usize> {
        self.text += text;
        let tokens = super::Tokenizer::encode_all(self.tokenizer, &self.text);
        assert!(
            tokens.starts_with(&self.tokens),
            "Text is split in the middle of token"
        );
        let prev_len = self.tokens.len();
        self.tokens = tokens;
        self.tokens[prev_len..].to_owned()
    }
}

#[derive(Clone)]
pub struct Decoder<'a> {
    tokenizer: &'a Tokenizer,
    tokens: Vec<usize>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> super::Decoder for Decoder<'a> {
    fn decode_next<I: IntoIterator<Item = usize>>(&mut self, tokens: I) -> String {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            super::Tokenizer::decode_all(self.tokenizer, tokens.iter().copied())
        };
        self.tokens.extend(tokens);
        let text = super::Tokenizer::decode_all(
            self.tokenizer,
            self.tokens[self.prev_index..].iter().copied(),
        );
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text.len()).1;
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            text.to_string()
        } else {
            String::new()
        }
    }
}

pub fn train<S, I>(vocab_size: usize, pieces: I) -> Tokenizer
where
    S: AsRef<str> + Send,
    I: IntoIterator<Item = S, IntoIter: Send>,
{
    use tokenizers::{
        AddedToken, TokenizerBuilder,
        models::bpe::{BPE, BpeTrainerBuilder},
        normalizers::{NFC, Sequence, Strip},
        processors::byte_level::ByteLevel,
    };

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("<s>"), true),
            AddedToken::from(String::from("</s>"), true),
            AddedToken::from(String::from("<pad>"), true),
            AddedToken::from(String::from("<unk>"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .unwrap();

    tokenizer.train(&mut trainer, pieces.into_iter()).unwrap();

    tokenizer.into()
}

pub fn train_cached<P, S, I>(path: P, vocab_size: usize, pieces: I, save: bool) -> Result<Tokenizer>
where
    P: AsRef<Path>,
    S: AsRef<str> + Send,
    I: IntoIterator<Item = S, IntoIter: Send>,
{
    let path = path.as_ref();
    if path.exists() {
        Tokenizer::from_file(path)
    } else {
        let tokenizer = train(vocab_size, pieces);
        if save {
            fs::create_dir_all(path.parent().unwrap())?;
            tokenizer.save(path, true)?;
        }
        Ok(tokenizer)
    }
}
