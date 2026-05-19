mod full_precision;
mod qlora_wrapper;
mod quantized;
mod rht_wrapper;

pub use full_precision::{FullPrecisionLinear, FullPrecisionLinearError};
pub use qlora_wrapper::{QLoRALinearWrapper, QLoRALinearWrapperError};
pub use quantized::{QuantizedLinear, QuantizedLinearError};
pub use rht_wrapper::{RHTLinearWrapper, RHTLinearWrapperError};
use thiserror::Error;

use crate::{
    DataType,
    backends::common::{Allocation, Backend, Encoder, gpu_types::QuantizationMethod},
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        awq_spec::AWQSpec,
        full_precision_spec::FullPrecisionSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
        low_rank_spec::LowRankSpec,
        mlx_spec::MLXSpec,
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

pub trait Linear<B: Backend> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>;
}

#[derive(Debug, Error)]
pub enum LinearBlockError<B: Backend> {
    #[error("QuantizedLinear error: {0}")]
    QuantizedLinearError(#[from] QuantizedLinearError<B>),
    #[error("FullPrecisionLinear error: {0}")]
    FullPrecisionLinearError(#[from] FullPrecisionLinearError<B>),
    #[error("QLoRALinearWrapper error: {0}")]
    QLoRALinearWrapperError(#[from] QLoRALinearWrapperError<B>),
    #[error("RHTLinearWrapper error: {0}")]
    RHTLinearWrapperError(#[from] RHTLinearWrapperError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> dyn Linear<B> {
    pub fn new<const N: usize>(
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        data_type: DataType,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        let weights_tree = parameter_tree.subtree("weights")?;
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                layout: Layout::OutputInput,
                ..
            }) => {
                let block = FullPrecisionLinear::new(
                    context,
                    input_dimension,
                    output_dimension_sum,
                    data_type,
                    parameter_tree,
                )?;
                Ok(Box::new(block))
            },
            AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                bits,
                group_size,
                layout: Layout::OutputInput,
                ..
            }) => {
                let block = QuantizedLinear::new(
                    context,
                    bits,
                    group_size,
                    QuantizationMethod::ScaleBias,
                    input_dimension,
                    output_dimension_sum,
                    data_type,
                    &weights_tree,
                    Some(parameter_tree),
                    None,
                )?;
                Ok(Box::new(block))
            },
            AnyWeightMatrixSpec::AWQSpec(AWQSpec {
                bits,
                group_size,
                is_symmetric: false,
                layout: Layout::OutputInput,
                ..
            }) => {
                let block = QuantizedLinear::new(
                    context,
                    bits,
                    group_size,
                    QuantizationMethod::ScaleZeroPoint,
                    input_dimension,
                    output_dimension_sum,
                    data_type,
                    &weights_tree,
                    Some(parameter_tree),
                    None,
                )?;
                Ok(Box::new(block))
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                adapter_spec: None,
                incoherence_block_size: Some(32),
                incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
                ..
            }) => Ok(Box::new(RHTLinearWrapper::new(
                context,
                input_dimension,
                output_dimension_sum,
                data_type,
                parameter_tree,
            )?)),
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                adapter_spec: Some(adapter_spec),
                incoherence_block_size: None,
                ..
            }) => {
                let quantized_tree = weights_tree.subtree("quantized")?;
                let adapter_tree = weights_tree.subtree("adapter")?;
                match (*quantization_spec, *adapter_spec) {
                    (
                        AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                            bits,
                            group_size,
                            layout: Layout::OutputInput,
                            ..
                        }),
                        AnyWeightMatrixSpec::LowRankSpec(LowRankSpec {
                            rank,
                            ..
                        }),
                    ) => Ok(Box::new(QLoRALinearWrapper::new(
                        context,
                        bits,
                        group_size,
                        QuantizationMethod::ScaleBias,
                        rank,
                        input_dimension,
                        output_dimension_sum,
                        data_type,
                        &quantized_tree,
                        &adapter_tree,
                        Some(parameter_tree),
                        None,
                    )?)),
                    (
                        AnyWeightMatrixSpec::AWQSpec(AWQSpec {
                            bits,
                            group_size,
                            is_symmetric: false,
                            layout: Layout::OutputInput,
                            ..
                        }),
                        AnyWeightMatrixSpec::LowRankSpec(LowRankSpec {
                            rank,
                            ..
                        }),
                    ) => Ok(Box::new(QLoRALinearWrapper::new(
                        context,
                        bits,
                        group_size,
                        QuantizationMethod::ScaleZeroPoint,
                        rank,
                        input_dimension,
                        output_dimension_sum,
                        data_type,
                        &quantized_tree,
                        &adapter_tree,
                        Some(parameter_tree),
                        None,
                    )?)),
                    (quantization_spec, adapter_spec) => Err(LinearBlockError::UnsupportedConfiguration(format!(
                        "Hybrid quantization={quantization_spec:?}, adapter={adapter_spec:?}"
                    ))),
                }
            },
            AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                quantization_spec,
                adapter_spec: Some(adapter_spec),
                incoherence_block_size: Some(32),
                incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
                ..
            }) => {
                let quantized_tree = weights_tree.subtree("quantized")?;
                let adapter_tree = weights_tree.subtree("adapter")?;
                let incoherence_signs_tree = weights_tree.subtree("incoherence_signs")?;
                match (*quantization_spec, *adapter_spec) {
                    (
                        AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                            bits,
                            group_size,
                            layout: Layout::OutputInput,
                            ..
                        }),
                        AnyWeightMatrixSpec::LowRankSpec(LowRankSpec {
                            rank,
                            ..
                        }),
                    ) => Ok(Box::new(QLoRALinearWrapper::new(
                        context,
                        bits,
                        group_size,
                        QuantizationMethod::ScaleBias,
                        rank,
                        input_dimension,
                        output_dimension_sum,
                        data_type,
                        &quantized_tree,
                        &adapter_tree,
                        Some(parameter_tree),
                        Some(&incoherence_signs_tree),
                    )?)),
                    (
                        AnyWeightMatrixSpec::AWQSpec(AWQSpec {
                            bits,
                            group_size,
                            is_symmetric: false,
                            layout: Layout::OutputInput,
                            ..
                        }),
                        AnyWeightMatrixSpec::LowRankSpec(LowRankSpec {
                            rank,
                            ..
                        }),
                    ) => Ok(Box::new(QLoRALinearWrapper::new(
                        context,
                        bits,
                        group_size,
                        QuantizationMethod::ScaleZeroPoint,
                        rank,
                        input_dimension,
                        output_dimension_sum,
                        data_type,
                        &quantized_tree,
                        &adapter_tree,
                        Some(parameter_tree),
                        Some(&incoherence_signs_tree),
                    )?)),
                    (quantization_spec, adapter_spec) => Err(LinearBlockError::UnsupportedConfiguration(format!(
                        "Hybrid quantization={quantization_spec:?}, adapter={adapter_spec:?}"
                    ))),
                }
            },
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!("{spec:?}"))),
        }
    }

    pub fn new_with_output_hadamard(
        context: &B::Context,
        weights_tree: &ParameterTree<B::Context>,
        bias_tree: Option<&ParameterTree<B::Context>>,
        output_factors: Allocation<B>,
        input_dim: usize,
        output_dim: usize,
        data_type: DataType,
    ) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        match spec {
            AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                bits,
                group_size,
                layout: Layout::OutputInput,
                ..
            }) => Ok(Box::new(QuantizedLinear::new(
                context,
                bits,
                group_size,
                QuantizationMethod::ScaleBias,
                input_dim,
                output_dim,
                data_type,
                weights_tree,
                bias_tree,
                Some(output_factors),
            )?)),
            AnyWeightMatrixSpec::AWQSpec(AWQSpec {
                bits,
                group_size,
                is_symmetric: false,
                layout: Layout::OutputInput,
                ..
            }) => Ok(Box::new(QuantizedLinear::new(
                context,
                bits,
                group_size,
                QuantizationMethod::ScaleZeroPoint,
                input_dim,
                output_dim,
                data_type,
                weights_tree,
                bias_tree,
                Some(output_factors),
            )?)),
            spec => Err(LinearBlockError::UnsupportedConfiguration(format!(
                "{spec:?} doesn't support fused output hadamard"
            ))),
        }
    }

    pub fn new_extracting_input_hadamard<const N: usize>(
        input_dimension: usize,
        output_dimensions: [usize; N],
        context: &B::Context,
        data_type: DataType,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<(Box<dyn Linear<B>>, Option<Allocation<B>>), LinearBlockError<B>> {
        let output_dimension_sum: usize = output_dimensions.iter().sum();
        let weights_tree = parameter_tree.subtree("weights")?;
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            adapter_spec: None,
            incoherence_block_size: Some(32),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        }) = spec
        {
            let incoherence_signs_tree = weights_tree.subtree("incoherence_signs")?;
            let input_factors = incoherence_signs_tree
                .leaf("input_signs")?
                .validate(&[input_dimension], DataType::I32)?
                .read_allocation()?;
            let output_factors = incoherence_signs_tree
                .leaf("output_signs")?
                .validate(&[output_dimension_sum], DataType::I32)?
                .read_allocation()?;
            let quantized_tree = weights_tree.subtree("quantized")?;
            let inner_linear = Self::new_with_output_hadamard(
                context,
                &quantized_tree,
                Some(parameter_tree),
                output_factors,
                input_dimension,
                output_dimension_sum,
                data_type,
            )?;
            Ok((inner_linear, Some(input_factors)))
        } else {
            let linear = Self::new(input_dimension, output_dimensions, context, data_type, parameter_tree)?;
            Ok((linear, None))
        }
    }
}
