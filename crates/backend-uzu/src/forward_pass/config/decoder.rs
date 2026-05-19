use super::{embedding::EmbeddingForwardPassConfig, transformer::TransformerForwardPassConfig};

#[derive(Debug, Clone)]
pub struct DecoderForwardPassConfig {
    pub embedding_forward_pass_config: EmbeddingForwardPassConfig,
    pub transformer_forward_pass_config: TransformerForwardPassConfig,
}

impl DecoderForwardPassConfig {
    pub fn new_for_inference() -> Self {
        Self {
            embedding_forward_pass_config: EmbeddingForwardPassConfig::new_for_inference(),
            transformer_forward_pass_config: TransformerForwardPassConfig::new_for_inference(),
        }
    }
}
