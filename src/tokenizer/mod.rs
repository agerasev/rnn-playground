pub mod char_;
pub mod hf;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(usize)]
pub enum SpecialToken {
    Begin,
    End,
    Pad,
    Unknown,
}

impl SpecialToken {
    pub const INDEX_MAP: [Self; 4] = [Self::Begin, Self::End, Self::Pad, Self::Unknown];
    pub const N: usize = const { Self::INDEX_MAP.len() };

    pub fn index(self) -> usize {
        self as usize
    }
    pub fn from_index(i: usize) -> Option<Self> {
        [Self::Begin, Self::End, Self::Pad, Self::Unknown]
            .get(i)
            .copied()
    }
}

pub trait Tokenizer: Send + Sync {
    type Encoder<'a>: Encoder + 'a
    where
        Self: 'a;
    type Decoder<'a>: Decoder + 'a
    where
        Self: 'a;

    fn vocab_size(&self) -> usize;
    fn special_token(&self, t: SpecialToken) -> usize;

    fn encoder(&self) -> Self::Encoder<'_>;
    fn encode_all(&self, text: &str) -> Vec<usize> {
        self.encoder().encode_next(text)
    }

    fn decoder(&self) -> Self::Decoder<'_>;
    fn decode_all<I: IntoIterator<Item = usize>>(&self, tokens: I) -> String {
        self.decoder().decode_next(tokens)
    }
}

pub trait Encoder {
    fn encode_next(&mut self, text: &str) -> Vec<usize>;
}
pub trait Decoder {
    fn decode_next<I: IntoIterator<Item = usize>>(&mut self, tokens: I) -> String;
}
