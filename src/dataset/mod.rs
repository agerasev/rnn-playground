use anyhow::Result;

pub mod tinystories;

pub trait Dataset: Iterator<Item = Result<String>> {}
