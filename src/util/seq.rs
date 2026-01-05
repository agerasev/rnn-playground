use std::sync::{Arc, OnceLock};

use burn::{
    Tensor,
    nn::attention::SeqLengthOption,
    prelude::*,
    tensor::{BasicOps, TensorKind, ops::IntElem},
};

#[derive(Clone, Debug)]
pub struct SeqTensor<B: Backend, const D: usize, K: TensorKind<B> = Float> {
    tensor: Tensor<B, D, K>,
    seq_lengths: Arc<[usize]>,
    mask: Arc<OnceLock<Tensor<B, 2, Bool>>>,
}

impl<B: Backend, const D: usize, K: TensorKind<B> + BasicOps<B>> SeqTensor<B, D, K> {
    fn assert_contents(&self) {
        assert!(self.tensor.dims()[0] >= self.seq_lengths.len());
        if !self.seq_lengths.is_empty() {
            assert!(self.tensor.dims()[1] >= self.seq_lengths.iter().copied().max().unwrap());
        }
        if let Some(mask) = self.mask.get() {
            assert_eq!(self.tensor.device(), mask.device());
            assert_eq!(self.tensor.dims()[0..2], mask.dims());
        }
    }

    pub fn new(tensor: Tensor<B, D, K>, seq_lengths: impl Into<Vec<usize>>) -> Self {
        let seq_lengths = seq_lengths.into();
        let this = Self {
            tensor,
            seq_lengths: seq_lengths.into(),
            mask: Default::default(),
        };
        this.assert_contents();
        this
    }

    pub fn replace<const D_: usize, K_: TensorKind<B> + BasicOps<B>>(
        self,
        new_tensor: Tensor<B, D_, K_>,
    ) -> SeqTensor<B, D_, K_> {
        let this = SeqTensor {
            tensor: new_tensor,
            seq_lengths: self.seq_lengths,
            mask: self.mask,
        };
        this.assert_contents();
        this
    }
    pub fn map<
        const D_: usize,
        K_: TensorKind<B> + BasicOps<B>,
        F: FnOnce(Tensor<B, D, K>) -> Tensor<B, D_, K_>,
    >(
        self,
        f: F,
    ) -> SeqTensor<B, D_, K_> {
        let this = SeqTensor {
            tensor: f(self.tensor),
            seq_lengths: self.seq_lengths,
            mask: self.mask,
        };
        this.assert_contents();
        this
    }

    pub fn device(&self) -> B::Device {
        self.tensor.device()
    }
    pub fn batch_size(&self) -> usize {
        self.tensor.dims()[0]
    }
    pub fn max_seq_length(&self) -> usize {
        self.tensor.dims()[1]
    }

    pub fn tensor(&self) -> &Tensor<B, D, K> {
        &self.tensor
    }
    pub fn seq_lengths(&self) -> &[usize] {
        &self.seq_lengths
    }

    /// Sequence mask.
    ///
    /// Each batch item looks like that: `[1, ..., 1, 0, ..., 0]`.
    pub fn mask(&self) -> &Tensor<B, 2, Bool> {
        self.mask.get_or_init(|| {
            mask_from_seq_lengths(
                &self.seq_lengths,
                self.max_seq_length(),
                &self.tensor.device(),
            )
        })
    }
}

fn mask_from_seq_lengths<B: Backend>(
    seq_lengths: &[usize],
    max_seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let range = Tensor::<B, 1, Int>::arange(0..(max_seq_length as i64), device).unsqueeze_dim(0);
    let seq_lengths = Tensor::<B, 1, Int>::from_data(seq_lengths, device).unsqueeze_dim(1);
    range.lower(seq_lengths)
}

impl<B: Backend> SeqTensor<B, 2, Int> {
    pub fn from_tokens(
        tokens_list: Vec<Vec<usize>>,
        pad_token: usize,
        seq_length: impl Into<SeqLengthOption>,
        device: &B::Device,
    ) -> Self {
        let tokens_max = || {
            tokens_list
                .iter()
                .map(|tokens| tokens.len())
                .max()
                .unwrap_or(1)
        };

        let batch_size = tokens_list.len();
        let max_seq_length = match seq_length.into() {
            SeqLengthOption::NoMax => tokens_max(),
            SeqLengthOption::Max(max) => usize::min(tokens_max(), max),
            SeqLengthOption::Fixed(limit) => limit,
        };
        let seq_lengths = tokens_list
            .iter()
            .map(|tokens| tokens.len().min(max_seq_length))
            .collect::<Vec<_>>();

        let mut tensor = Tensor::zeros([batch_size, max_seq_length], device);
        tensor = tensor.add_scalar(pad_token as i64);

        for (index, tokens) in tokens_list.into_iter().enumerate() {
            let seq_length = tokens.len().min(max_seq_length);
            tensor = tensor.slice_assign(
                [index..index + 1, 0..seq_length],
                Tensor::from_data(
                    TensorData::new(
                        tokens
                            .into_iter()
                            .take(max_seq_length)
                            .map(|e| (e as i64).elem::<IntElem<B>>())
                            .collect(),
                        Shape::new([1, seq_length]),
                    ),
                    device,
                ),
            );
        }

        Self::new(tensor, seq_lengths)
    }
}
