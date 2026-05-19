use thiserror::Error;

use super::Linear;
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::{
            Kernels, TensorAddBiasKernel,
            quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
                QuantizedMatmulKernelEncodable,
            },
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QuantizedLinearError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("QuantizedMatmul error: {0}")]
    QuantizedMatmulError(#[from] QuantizedMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported data type for quantized kernel: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported quantized linear configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct QuantizedLinear<B: Backend> {
    kernel: QuantizedMatmulKernelEncodable<B>,
    bias_add_kernel: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
    biases: Option<Allocation<B>>,
    weights: Allocation<B>,
    scales: Allocation<B>,
    zero_points_or_biases: Allocation<B>,
    output_hadamard_factors: Option<Allocation<B>>,
    output_dim: usize,
    output_data_type: DataType,
}

impl<B: Backend> QuantizedLinear<B> {
    pub fn new(
        context: &B::Context,
        bits: u32,
        group_size: usize,
        quantization_method: QuantizationMethod,
        input_dim: usize,
        output_dim: usize,
        data_type: DataType,
        weights_tree: &ParameterTree<B::Context>,
        bias_tree: Option<&ParameterTree<B::Context>>,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, QuantizedLinearError<B>> {
        let weight_quantization_mode = match bits {
            4 => QuantizationMode::U4,
            8 => QuantizationMode::U8,
            _ => {
                return Err(QuantizedLinearError::UnsupportedConfiguration(format!(
                    "{quantization_method} bits={bits}, group_size={group_size}"
                )));
            },
        };

        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(QuantizedLinearError::UnsupportedDataType(data_type));
        }

        let packing_divisor = weight_quantization_mode.packing_divisor();
        let storage_type = weight_quantization_mode.storage_type();
        let k_g = input_dim.div_ceil(group_size);
        let weights = weights_tree
            .leaf("weights")?
            .validate(&[output_dim, input_dim / packing_divisor], storage_type)?
            .read_allocation()?;
        let scales = weights_tree.leaf("scales")?.validate(&[output_dim, k_g], data_type)?.read_allocation()?;
        let zero_points_or_biases = match quantization_method {
            QuantizationMethod::ScaleBias => weights_tree
                .leaf("biases")?
                .validate(&[output_dim, k_g], data_type)?
                .read_allocation()?,
            QuantizationMethod::ScaleZeroPoint => {
                let expected_zero_points_entries = (k_g + packing_divisor - 1) / packing_divisor;
                weights_tree
                    .leaf("zero_points")?
                    .validate(&[output_dim, expected_zero_points_entries], storage_type)?
                    .read_allocation()?
            },
        };

        let (bias_add_kernel, biases) = match bias_tree.and_then(|tree| tree.leaf("biases").ok()) {
            Some(biases_leaf) => {
                let bias_add_kernel =
                    <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, true)
                        .map_err(QuantizedLinearError::BackendError)?;
                (
                    Some(bias_add_kernel),
                    Some(biases_leaf.validate(&[output_dim], data_type)?.read_allocation()?),
                )
            },
            None => (None, None),
        };

        let kernel = QuantizedMatmulKernelEncodable::new(
            context,
            QuantizedMatmulConfiguration {
                data_type,
                group_size,
                input_dim,
                output_dim,
                mode: weight_quantization_mode,
                quantization_method,
                use_hadamard: output_hadamard_factors.is_some(),
            },
        )?;

        Ok(Self {
            kernel,
            bias_add_kernel,
            biases,
            weights,
            scales,
            zero_points_or_biases,
            output_hadamard_factors,
            output_dim,
            output_data_type: data_type,
        })
    }
}

impl<B: Backend> Linear<B> for QuantizedLinear<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.output_data_type))?;

        self.kernel.encode(
            encoder,
            QuantizedMatmulArguments {
                a: &input,
                a_offset: 0,
                b: &self.weights,
                scales: &self.scales,
                zero_points_or_biases: &self.zero_points_or_biases,
                output: &mut output,
                hadamard_factors: self.output_hadamard_factors.as_ref(),
                batch_dim,
            },
        );

        if let (Some(bias_add_kernel), Some(biases)) = (&self.bias_add_kernel, &self.biases) {
            let total_length = batch_dim * self.output_dim;
            bias_add_kernel.encode(
                None::<&Allocation<B>>,
                biases,
                &mut output,
                self.output_dim as u32,
                total_length as u32,
                encoder,
            );
        }

        Ok(output)
    }
}
