use std::path::Path;

use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum SpecialToken {
    Begin,
    End,
    Pad,
    Unk,
}
impl SpecialToken {
    pub const ALL: [Self; 4] = [Self::Begin, Self::End, Self::Pad, Self::Unk];

    pub fn to_str(self) -> &'static str {
        match self {
            Self::Begin => "<s>",
            Self::End => "</s>",
            Self::Pad => "<pad>",
            Self::Unk => "<unk>",
        }
    }
    pub fn get_id(self, tokenizer: &Tokenizer) -> u32 {
        tokenizer.token_to_id(self.to_str()).unwrap()
    }
}

pub fn id_to_str(tokenizer: &Tokenizer, id: u32) -> String {
    tokenizer.id_to_token(id).unwrap()
}

pub fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, false).unwrap().get_ids().to_vec()
}

pub fn decode<I: IntoIterator<Item = u32>>(tokenizer: &Tokenizer, tokens: I) -> String {
    tokenizer
        .decode(&tokens.into_iter().collect::<Vec<_>>(), false)
        .unwrap()
}

#[derive(Clone)]
pub struct Encoder<'a> {
    tokenizer: &'a Tokenizer,
    text: String,
    tokens: Vec<u32>,
}

impl<'a> Encoder<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            text: String::new(),
            tokens: Vec::new(),
        }
    }
    pub fn encode_next(&mut self, text: &str) -> Vec<u32> {
        self.text += text;
        let tokens = encode(self.tokenizer, &self.text);
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
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn decode_next<I: IntoIterator<Item = u32>>(&mut self, tokens: I) -> String {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            decode(self.tokenizer, tokens.iter().copied())
        };
        self.tokens.extend(tokens);
        let text = decode(
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

pub fn train<S, I>(vocab_size: usize, samples: I) -> Tokenizer
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
        .special_tokens(
            SpecialToken::ALL
                .into_iter()
                .map(|t| AddedToken::from(t.to_str().to_string(), true))
                .collect(),
        )
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

    tokenizer.train(&mut trainer, samples.into_iter()).unwrap();

    tokenizer.into()
}

pub fn train_cached<P, S, I>(path: P, vocab_size: usize, samples: I) -> Result<Tokenizer>
where
    P: AsRef<Path>,
    S: AsRef<str> + Send,
    I: IntoIterator<Item = S, IntoIter: Send>,
{
    let path = path.as_ref();
    if path.exists() {
        Tokenizer::from_file(path)
    } else {
        let tokenizer = train(vocab_size, samples);
        tokenizer.save(path, false).map(move |_| tokenizer)
    }
    .map_err(|e| anyhow!("{e}"))
}
