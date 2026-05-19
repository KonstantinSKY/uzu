use std::{path::PathBuf, sync::Arc};

use backend_uzu::inference::resolve_model_specialization;
use shoji::types::model::Model;

pub type ModelResolver = Arc<dyn Fn(Model) -> Option<Model> + Send + Sync>;

#[derive(Clone)]
pub struct Config {
    pub identifier: String,
    pub backend_identifier: String,
    pub backend_version: String,
    pub name: String,
    pub path: String,
    pub resolver: Option<ModelResolver>,
}

impl Config {
    pub fn new(
        identifier: String,
        backend_identifier: String,
        backend_version: String,
        name: String,
        path: String,
        resolver: Option<ModelResolver>,
    ) -> Self {
        Self {
            identifier,
            backend_identifier,
            backend_version,
            name,
            path,
            resolver,
        }
    }

    pub fn lalamo(
        backend_identifier: String,
        backend_version: String,
        path: String,
    ) -> Self {
        let models_path = PathBuf::from(&path).join("models");
        let resolver: ModelResolver = Arc::new(move |mut model: Model| -> Option<Model> {
            let model_path = PathBuf::from(model.local_external_path()?);
            model.specializations = vec![resolve_model_specialization(&model_path)?];
            Some(model)
        });
        Self::new(
            "lalamo".to_string(),
            backend_identifier,
            backend_version,
            "Lalamo".to_string(),
            models_path.to_string_lossy().to_string(),
            Some(resolver),
        )
    }
}
