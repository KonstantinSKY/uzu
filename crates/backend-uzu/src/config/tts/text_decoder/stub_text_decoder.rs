use proc_macros::uzu_config;

#[uzu_config(super::TTSTextDecoderConfig)]
pub struct StubTextDecoderConfig {
    pub num_codebooks: usize,
    pub codebook_size: usize,
}
