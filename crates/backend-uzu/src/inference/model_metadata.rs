use std::{fs::File, io::BufReader, path::Path};

use shoji::types::model::ModelSpecialization;

use crate::config::model::AnyModelConfig;

pub fn resolve_model_specialization(model_path: &Path) -> Option<ModelSpecialization> {
    let config_path = model_path.join("config.json");
    let file = File::open(&config_path).ok()?;
    let config: AnyModelConfig = serde_json::from_reader(BufReader::new(file)).ok()?;
    Some(match config {
        AnyModelConfig::LanguageModelConfig(_) => ModelSpecialization::Chat {},
        AnyModelConfig::ClassifierModelConfig(_) => ModelSpecialization::Classification {},
        AnyModelConfig::TTSModelConfig(_) => ModelSpecialization::TextToSpeech {},
    })
}
