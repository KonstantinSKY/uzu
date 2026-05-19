use std::cell::RefCell;

use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{HadamardTransformOrder, QuantizationMethod},
        kernel::{
            HadamardTransformKernel, Kernels, ManualKernels, TensorAddBiasKernel,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
        },
    },
    encodable_block::{
        Linear,
        linear::{LinearBlockError, QuantizedLinear, QuantizedLinearError},
    },
    prelude::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QLoRALinearWrapperError<B: Backend> {
    #[error("Inner linear error: {0}")]
    InnerLinearError(#[from] Box<LinearBlockError<B>>),
    #[error("Quantized linear error: {0}")]
    QuantizedLinearError(#[from] QuantizedLinearError<B>),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

pub struct QLoRALinearWrapper<B: Backend> {
    base_linear: QuantizedLinear<B>,
    input_hadamard: Option<(<B::Kernels as Kernels>::HadamardTransformKernel, Allocation<B>)>,
    output_hadamard: Option<(<B::Kernels as Kernels>::HadamardTransformKernel, Allocation<B>)>,
    adapter_kernel: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    bias_add_kernel: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
    adapter_down: Allocation<B>,
    adapter_up: Allocation<B>,
    biases: Option<Allocation<B>>,
    input_dim: usize,
    output_dim: usize,
    lora_rank: usize,
    data_type: DataType,
}

impl<B: Backend> QLoRALinearWrapper<B> {
    pub fn new(
        context: &B::Context,
        bits: u32,
        group_size: usize,
        quantization_method: QuantizationMethod,
        lora_rank: usize,
        input_dim: usize,
        output_dim: usize,
        data_type: DataType,
        quantized_tree: &ParameterTree<B::Context>,
        adapter_tree: &ParameterTree<B::Context>,
        bias_tree: Option<&ParameterTree<B::Context>>,
        incoherence_signs_tree: Option<&ParameterTree<B::Context>>,
    ) -> Result<Self, QLoRALinearWrapperError<B>> {
        let base_linear = QuantizedLinear::new(
            context,
            bits,
            group_size,
            quantization_method,
            input_dim,
            output_dim,
            data_type,
            quantized_tree,
            None,
            None,
        )?;
        let (input_hadamard, output_hadamard) = if let Some(incoherence_signs_tree) = incoherence_signs_tree {
            let input_factors = incoherence_signs_tree
                .leaf("input_signs")?
                .validate(&[input_dim], DataType::I32)?
                .read_allocation()?;
            let output_factors = incoherence_signs_tree
                .leaf("output_signs")?
                .validate(&[output_dim], DataType::I32)?
                .read_allocation()?;
            (
                Some((
                    <B::Kernels as Kernels>::HadamardTransformKernel::new(
                        context,
                        data_type,
                        HadamardTransformOrder::Input,
                    )
                        .map_err(QLoRALinearWrapperError::BackendError)?,
                    input_factors,
                )),
                Some((
                    <B::Kernels as Kernels>::HadamardTransformKernel::new(
                        context,
                        data_type,
                        HadamardTransformOrder::Output,
                    )
                        .map_err(QLoRALinearWrapperError::BackendError)?,
                    output_factors,
                )),
            )
        } else {
            (None, None)
        };
        let adapter_kernel =
            RefCell::new(<<B::Kernels as ManualKernels>::MatmulKernel as MatmulKernel>::new(context, data_type)?);

        let adapter_down = adapter_tree
            .leaf("down_projection")?
            .validate(&[lora_rank, input_dim], data_type)?
            .read_allocation()?;

        let adapter_up = adapter_tree
            .leaf("up_projection")?
            .validate(&[output_dim, lora_rank], data_type)?
            .read_allocation()?;

        let (bias_add_kernel, biases) = match bias_tree.and_then(|tree| tree.leaf("biases").ok()) {
            Some(biases_leaf) => {
                let bias_add_kernel = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, true)
                    .map_err(QLoRALinearWrapperError::BackendError)?;
                (
                    Some(bias_add_kernel),
                    Some(biases_leaf.validate(&[output_dim], data_type)?.read_allocation()?),
                )
            },
            None => (None, None),
        };

        Ok(Self {
            base_linear,
            input_hadamard,
            output_hadamard,
            adapter_kernel,
            bias_add_kernel,
            adapter_down,
            adapter_up,
            biases,
            input_dim,
            output_dim,
            lora_rank,
            data_type,
        })
    }
}

impl<B: Backend> Linear<B> for QLoRALinearWrapper<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, <B as Backend>::Error> {
        let mut intermediate =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.lora_rank], self.data_type))?;
        {
            let mut adapter_kernel = self.adapter_kernel.borrow_mut();
            adapter_kernel.encode(
                MatmulArguments {
                    a: &input,
                    a_offset: 0,
                    b: &self.adapter_down,
                    b_offset: 0,
                    b_leading_dimension: None,
                    b_transpose: true,
                    ab_scale: 1.0,
                    c: MatmulArgumentC::None,
                    d: &mut intermediate,
                    batch_dim: batch_dim as u32,
                    input_dim: self.input_dim as u32,
                    output_dim: self.lora_rank as u32,
                },
                encoder,
            );
        }

        let base_input = if let Some((input_hadamard_kernel, input_factors)) = &self.input_hadamard {
            let mut base_input =
                encoder.allocate_scratch(size_for_shape(&[batch_dim, self.input_dim], self.data_type))?;
            encoder.encode_copy(&input, .., &mut base_input, ..);
            input_hadamard_kernel.encode(
                &mut base_input,
                input_factors,
                self.input_dim as u32,
                batch_dim as u32,
                encoder,
            );
            base_input
        } else {
            input
        };

        let mut output = self.base_linear.encode(base_input, batch_dim, encoder)?;

        {
            let mut adapter_kernel = self.adapter_kernel.borrow_mut();
            adapter_kernel.encode(
                MatmulArguments {
                    a: &intermediate,
                    a_offset: 0,
                    b: &self.adapter_up,
                    b_offset: 0,
                    b_leading_dimension: None,
                    b_transpose: true,
                    ab_scale: 1.0,
                    c: MatmulArgumentC::Accumulate,
                    d: &mut output,
                    batch_dim: batch_dim as u32,
                    input_dim: self.lora_rank as u32,
                    output_dim: self.output_dim as u32,
                },
                encoder,
            );
        }

        if let Some((output_hadamard_kernel, output_factors)) = &self.output_hadamard {
            output_hadamard_kernel.encode(
                &mut output,
                output_factors,
                self.output_dim as u32,
                batch_dim as u32,
                encoder,
            );
        }

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
