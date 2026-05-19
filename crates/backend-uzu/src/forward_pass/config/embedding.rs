use crate::DataType;

#[derive(Debug, Clone)]
pub struct EmbeddingForwardPassConfig {
    pub activation_data_type: DataType,
}

impl EmbeddingForwardPassConfig {
    pub fn new_for_inference() -> Self {
        Self {
            activation_data_type: DataType::BF16,
        }
    }
}
