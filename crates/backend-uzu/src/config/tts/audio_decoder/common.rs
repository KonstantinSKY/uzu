use proc_macros::uzu_config;

#[uzu_config]
pub struct CausalConv1dConfig {
    pub has_biases: bool,
}

#[uzu_config]
pub struct CausalTransposeConv1dConfig {
    pub has_biases: bool,
}

#[uzu_config]
pub struct Snake1dConfig {}
