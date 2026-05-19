use super::{
    embedding::EmbeddingForwardPassConfig, normalization::NormalizationForwardPassConfig,
    transformer::TransformerForwardPassConfig,
};

#[derive(Debug, Clone)]
pub struct ClassifierForwardPassConfig {
    pub embedding_forward_pass_config: EmbeddingForwardPassConfig,
    pub transformer_forward_pass_config: TransformerForwardPassConfig,
    pub normalization_forward_pass_config: NormalizationForwardPassConfig,
}

impl ClassifierForwardPassConfig {
    pub fn new_for_inference() -> Self {
        Self {
            embedding_forward_pass_config: EmbeddingForwardPassConfig::new_for_inference(),
            transformer_forward_pass_config: TransformerForwardPassConfig::new_for_inference(),
            normalization_forward_pass_config: NormalizationForwardPassConfig::new_for_inference(),
        }
    }
}
