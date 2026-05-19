use proc_macros::uzu_config;

use crate::config::tts::audio_decoder::nano_codec_modules::{
    CausalHiFiGANDecoderConfig, GroupFiniteScalarQuantizerConfig,
};

#[uzu_config(super::TTSAudioDecoderConfig)]
pub struct NanoCodecConfig {
    pub quantizer_config: GroupFiniteScalarQuantizerConfig,
    pub decoder_config: CausalHiFiGANDecoderConfig,
    pub samplerate: u32,
    pub base_channels: usize,
    pub up_sample_rates: Box<[usize]>,
    pub in_kernel_size: usize,
    pub out_kernel_size: usize,
    pub resblock_kernel_sizes: Box<[usize]>,
    pub resblock_dilations: Box<[usize]>,
}
