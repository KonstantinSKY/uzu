use thiserror::Error;

use super::{Linear, LinearBlockError};
use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::HadamardTransformOrder,
        kernel::{HadamardTransformKernel, Kernels},
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec,
        hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum RHTLinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] Box<LinearBlockError<B>>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported RHT linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct RHTLinearWrapper<B: Backend> {
    input_hadamard_kernel: <B::Kernels as Kernels>::HadamardTransformKernel,
    input_factors: Allocation<B>,
    inner_linear: Box<dyn Linear<B>>,
    input_dimension: usize,
}

impl<B: Backend> RHTLinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        input_dimension: usize,
        output_dimension: usize,
        data_type: DataType,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, RHTLinearWrapperError<B>> {
        let weights_tree = parameter_tree.subtree("weights")?;
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        if let AnyWeightMatrixSpec::HybridSpec(HybridSpec {
            adapter_spec: None,
            incoherence_block_size: Some(32),
            incoherence_processing_mode: IncoherenceProcessingMode::InputOutput,
            ..
        }) = &spec
        {
        } else {
            return Err(RHTLinearWrapperError::UnsupportedConfiguration(format!("{spec:?}")));
        }

        let incoherence_signs_tree = weights_tree.subtree("incoherence_signs")?;
        let input_factors = incoherence_signs_tree
            .leaf("input_signs")?
            .validate(&[input_dimension], DataType::I32)?
            .read_allocation()?;
        let output_factors = incoherence_signs_tree
            .leaf("output_signs")?
            .validate(&[output_dimension], DataType::I32)?
            .read_allocation()?;
        let quantized_tree = weights_tree.subtree("quantized")?;
        let input_hadamard_kernel =
            <B::Kernels as Kernels>::HadamardTransformKernel::new(
                context,
                data_type,
                HadamardTransformOrder::Input,
            )
            .map_err(RHTLinearWrapperError::BackendError)?;
        let inner_linear = <dyn Linear<B>>::new_with_output_hadamard(
            context,
            &quantized_tree,
            Some(parameter_tree),
            output_factors,
            input_dimension,
            output_dimension,
            data_type,
        )
        .map_err(|error| RHTLinearWrapperError::InnerLinearError(Box::new(error)))?;

        Ok(Self {
            input_hadamard_kernel,
            input_factors,
            inner_linear,
            input_dimension,
        })
    }
}

impl<B: Backend> Linear<B> for RHTLinearWrapper<B> {
    fn encode(
        &self,
        mut input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        self.input_hadamard_kernel.encode(
            &mut input,
            &self.input_factors,
            self.input_dimension as u32,
            batch_dim as u32,
            encoder,
        );
        self.inner_linear.encode(input, batch_dim, encoder)
    }
}
