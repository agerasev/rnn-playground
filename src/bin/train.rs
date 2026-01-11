use std::{env, fs, path::Path};

use anyhow::Result;
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, LearningStrategy, metric::LossMetric},
};

use rnn_exp::{
    dataset::{DatasetWrapper, SeqBatcher, tinystories::TinyStoriesConfig},
    model::{
        ModelConfig,
        lm::LmConfig,
        seq::{
            rnn::{NaiveConfig, SomeRnnConfig},
            some::SeqModelConfig,
        },
    },
    tokenizer::{self, TokenizerConfig},
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub train_dataset: TinyStoriesConfig,
    pub valid_dataset: TinyStoriesConfig,
    pub tokenizer: TokenizerConfig,
    pub model: LmConfig,
    pub optimizer: AdamConfig,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl TrainingConfig {
    pub fn load_or_default(config_path: &Path) -> Result<Self> {
        if config_path.exists() {
            Ok(TrainingConfig::load(config_path)?)
        } else {
            let home = env::home_dir().unwrap();
            let repo_dir = home.join("data/datasets/TinyStories");
            let train_dataset = TinyStoriesConfig {
                repo_dir: repo_dir.clone(),
                file_path: "TinyStoriesV2-GPT4-train.txt".into(),
            };
            let valid_dataset = TinyStoriesConfig {
                repo_dir,
                file_path: "TinyStoriesV2-GPT4-valid.txt".into(),
            };

            let vocab_size = 1024;
            let tokenizer = TokenizerConfig {
                vocab_size,
                path: train_dataset
                    .repo_dir
                    .join("tokenizers")
                    .join(format!("bpe{vocab_size}.json")),
            };

            let max_length = 256;
            let model = LmConfig {
                vocab_size,
                max_length,
                decoder: SeqModelConfig::Rnn(SomeRnnConfig::Naive(NaiveConfig { d_hidden: 256 })),
            };

            let optimizer = AdamConfig::new();

            Ok(TrainingConfig {
                train_dataset,
                valid_dataset,
                tokenizer,
                model,
                optimizer,
                batch_size: 64,
                learning_rate: 1e-4,
                seed: 0xdeadbeef,
            })
        }
    }
}

pub fn train<B: AutodiffBackend>(artifact_dir: &Path, device: B::Device) -> Result<()> {
    fs::create_dir_all(artifact_dir)?;
    let config = TrainingConfig::load_or_default(&artifact_dir.join("config.json"))?;

    B::seed(&device, config.seed);

    println!("Loading dataset ...");
    let train_dataset = config.train_dataset.init_dataset()?;
    let valid_dataset = config.valid_dataset.init_dataset()?;
    let train_samples = train_dataset.samples()?.collect::<Result<Vec<_>, _>>()?;
    let valid_samples = valid_dataset.samples()?.collect::<Result<Vec<_>, _>>()?;

    println!("Loading or training tokenizer ...");
    fs::create_dir_all(config.tokenizer.path.parent().unwrap())?;
    let tokenizer = tokenizer::train_cached(
        config.tokenizer.path,
        config.tokenizer.vocab_size,
        &train_samples,
    )?;

    let batcher = SeqBatcher {
        max_length: config.model.max_length,
        tokenizer,
    };

    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(DatasetWrapper(train_samples));

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(DatasetWrapper(valid_samples));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(1)
        .summary()
        .build(
            config.model.init_model(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    println!("Training model ...");
    let result = learner.fit(dataloader_train, dataloader_valid);

    result
        .model
        .save_file(artifact_dir.join("model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}

fn main() -> Result<()> {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    train::<MyAutodiffBackend>(Path::new("./data/00-test-rnn"), device.clone())
}
