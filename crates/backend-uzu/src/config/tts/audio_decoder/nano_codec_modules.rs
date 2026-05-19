use proc_macros::uzu_config;

use crate::config::tts::audio_decoder::common::{CausalConv1dConfig, CausalTransposeConv1dConfig, Snake1dConfig};

#[uzu_config]
pub struct FiniteScalarQuantizerConfig {
    pub num_levels: Box<[usize]>,
    pub eps: f32,
}

#[uzu_config]
pub struct GroupFiniteScalarQuantizerConfig {
    pub num_groups: usize,
    pub quantizer_config: FiniteScalarQuantizerConfig,
}

#[uzu_config]
pub struct HalfSnakeConfig {
    pub snake_config: Snake1dConfig,
    pub leaky_relu_negative_slope: f32,
}

#[uzu_config]
pub struct ResidualBlockConfig {
    pub activation_config: HalfSnakeConfig,
    pub conv_config: CausalConv1dConfig,
}

#[uzu_config]
pub struct HiFiGANResBlockConfig {
    pub residual_block_config: ResidualBlockConfig,
}

#[uzu_config]
pub struct HiFiGANResLayerConfig {
    pub hifigan_res_block_config: HiFiGANResBlockConfig,
}

#[uzu_config]
pub struct CausalHiFiGANDecoderConfig {
    pub activation_config: HalfSnakeConfig,
    pub pre_conv_config: CausalConv1dConfig,
    pub transpose_conv_config: CausalTransposeConv1dConfig,
    pub res_layer_config: HiFiGANResLayerConfig,
    pub post_conv_config: CausalConv1dConfig,
}
