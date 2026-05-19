use crate::DataType;

#[derive(Debug, Clone)]
pub struct NormalizationForwardPassConfig {
    pub accumulation_data_type: DataType,
}

impl NormalizationForwardPassConfig {
    pub fn new_for_inference() -> Self {
        Self {
            accumulation_data_type: DataType::F32,
        }
    }
}
