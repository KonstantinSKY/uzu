use super::normalization::NormalizationForwardPassConfig;
use crate::DataType;

#[derive(Debug, Clone)]
pub struct MixerForwardPassConfig {
    pub rope_data_type: DataType,
    pub normalization_forward_pass_config: NormalizationForwardPassConfig,
}

impl MixerForwardPassConfig {
    pub fn new_for_inference() -> Self {
        Self {
            rope_data_type: DataType::F32,
            normalization_forward_pass_config: NormalizationForwardPassConfig::new_for_inference(),
        }
    }
}
