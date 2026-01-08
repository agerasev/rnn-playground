use anyhow::Result;
use rnn_exp::dataset::tinystories::TinyStoriesConfig;

fn main() -> Result<()> {
    let dataset = TinyStoriesConfig {
        repo_dir: "/home/agerasev/data/datasets/TinyStories".into(),
        file_path: "TinyStoriesV2-GPT4-train.txt".into(),
    }
    .init_dataset()?;

    let mut count = 0;
    let (mut min_len, mut max_len) = (usize::MAX, usize::MIN);
    for sample in dataset.samples()? {
        count += 1;
        let text = sample?;
        let len = text.chars().count();
        min_len = len.min(min_len);
        max_len = len.max(max_len);
    }
    dbg!((count, min_len, max_len));

    Ok(())
}
