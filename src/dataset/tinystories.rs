use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::dataset::util::git;

const GIT_REPO: &str = "git@hf.co:datasets/roneneldan/TinyStories";
pub const TRAIN_FILE: &str = "TinyStoriesV2-GPT4-train.txt";
pub const VALID_FILE: &str = "TinyStoriesV2-GPT4-valid.txt";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TinyStoriesConfig {
    pub repo_dir: PathBuf,
    pub file_path: PathBuf,
}

impl TinyStoriesConfig {
    pub fn init_dataset(&self) -> Result<TinyStories> {
        if !self.repo_dir.exists() {
            git::clone(GIT_REPO, &self.repo_dir, false);
        }
        git::lfs_pull(&self.repo_dir, [&self.file_path]);

        Ok(TinyStories {
            file_path: self.repo_dir.join(&self.file_path),
        })
    }
}

pub struct TinyStories {
    file_path: PathBuf,
}

impl TinyStories {
    pub fn samples(&self) -> Result<impl Iterator<Item = Result<String>>> {
        TinyStoriesReader::new(&self.file_path)
    }
}

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

struct TinyStoriesReader {
    file: BufReader<File>,
}

impl TinyStoriesReader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            file: BufReader::new(File::open(path)?),
        })
    }
}

impl Iterator for TinyStoriesReader {
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
