use super::{Decoder, Encoder, SpecialToken, Tokenizer};
use std::collections::BTreeSet;

pub struct CharTokenizer {
    pub alphabet: Vec<char>,
    pub mask_lf: bool,
}

impl CharTokenizer {
    fn new<I: IntoIterator<Item = char>>(alphabet: I) -> Self {
        Self {
            alphabet: BTreeSet::from_iter(alphabet).into_iter().collect(),
            mask_lf: false,
        }
    }
}

impl Tokenizer for CharTokenizer {
    type Encoder<'a> = CharCoder<'a>;
    type Decoder<'a> = CharCoder<'a>;

    fn vocab_size(&self) -> usize {
        SpecialToken::N + self.alphabet.len()
    }

    fn encode_all(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| {
                if let Ok(i) = self.alphabet.binary_search(&c) {
                    SpecialToken::N + i
                } else {
                    self.special_token(SpecialToken::Unknown)
                }
            })
            .collect()
    }

    fn decode_all<I: IntoIterator<Item = usize>>(&self, tokens: I) -> String {
        let mut text = String::new();
        for t in tokens.into_iter() {
            if let Some(s) = SpecialToken::from_index(t) {
                use SpecialToken::*;
                match s {
                    Begin => text.push_str("<BOS>"),
                    End => text.push_str("<EOS>"),
                    Pad => text.push_str("<PAD>"),
                    Unknown => text.push_str("<UNK>"),
                }
            } else {
                let c = self.alphabet[t - SpecialToken::N];
                if self.mask_lf && c == '\n' {
                    text.push_str("<LF>")
                } else {
                    text.push(c)
                }
            }
        }
        text
    }

    fn encoder(&self) -> Self::Encoder<'_> {
        CharCoder { tokenizer: self }
    }
    fn decoder(&self) -> Self::Decoder<'_> {
        CharCoder { tokenizer: self }
    }

    fn special_token(&self, t: SpecialToken) -> usize {
        t as usize
    }
}

#[derive(Clone)]
pub struct CharCoder<'a> {
    tokenizer: &'a CharTokenizer,
}

impl<'a> Encoder for CharCoder<'a> {
    fn encode_next(&mut self, text: &str) -> Vec<usize> {
        self.tokenizer.encode_all(text)
    }
}

impl<'a> Decoder for CharCoder<'a> {
    fn decode_next<I: IntoIterator<Item = usize>>(&mut self, tokens: I) -> String {
        self.tokenizer.decode_all(tokens)
    }
}

const ASCII_ALPHABET: &[char] = &[
    '\n', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
    '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '{', '|', '}', '~',
];

pub fn ascii_tokenizer() -> CharTokenizer {
    CharTokenizer::new(ASCII_ALPHABET.iter().copied())
}
