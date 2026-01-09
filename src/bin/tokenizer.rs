use std::{env, fs};

use anyhow::Result;
use rnn_exp::{
    dataset::tinystories::TinyStoriesConfig,
    tokenizer::{self, id_to_str},
};

fn main() -> Result<()> {
    let home = env::home_dir().unwrap();
    let dataset_config = TinyStoriesConfig {
        repo_dir: home.join("data/datasets/TinyStories"),
        file_path: "TinyStoriesV2-GPT4-train.txt".into(),
    };
    let dataset = dataset_config.init_dataset()?;
    println!("Reading dataset ...");
    let samples = dataset.samples()?.collect::<Result<Vec<_>, _>>()?;

    let tokenizers_dir = dataset_config.repo_dir.join("tokenizers");
    fs::create_dir_all(&tokenizers_dir)?;

    let vocab_size = 32768;
    println!("Training tokenizer (vocab_size = {vocab_size}) ...");
    let tokenizer = tokenizer::train_cached(
        tokenizers_dir.join(format!("bpe{vocab_size}.json")),
        vocab_size,
        samples,
    )?;

    println!(
        "{:?}",
        tokenizer::encode(&tokenizer, "The quick brown fox jumps over a lazy dog")
            .into_iter()
            .map(|t| id_to_str(&tokenizer, t))
            .collect::<Vec<_>>()
    );

    Ok(())
}
