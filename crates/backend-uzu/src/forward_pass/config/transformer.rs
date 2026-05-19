use super::{mixer::MixerForwardPassConfig, normalization::NormalizationForwardPassConfig};

#[derive(Debug, Clone)]
pub struct TransformerForwardPassConfig {
    pub mixer_forward_pass_config: MixerForwardPassConfig,
    pub normalization_forward_pass_config: NormalizationForwardPassConfig,
}

impl TransformerForwardPassConfig {
    pub fn new_for_inference() -> Self {
        Self {
            mixer_forward_pass_config: MixerForwardPassConfig::new_for_inference(),
            normalization_forward_pass_config: NormalizationForwardPassConfig::new_for_inference(),
        }
    }
}
