use proc_macros::uzu_config_abstract;

pub mod common;
pub mod descript_audio_codec;
pub mod fish_audio_modules;
pub mod nano_codec;
pub mod nano_codec_modules;

#[uzu_config_abstract(descript_audio_codec::DescriptAudioCodecConfig, nano_codec::NanoCodecConfig)]
pub struct TTSAudioDecoderConfig {}
