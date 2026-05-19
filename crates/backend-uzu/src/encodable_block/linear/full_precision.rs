use std::cell::RefCell;

use thiserror::Error;

use super::Linear;
use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            ManualKernels,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum FullPrecisionLinearError<B: Backend> {
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported data type for full precision linear kernel: {0:?}")]
    UnsupportedDataType(DataType),
}

pub struct FullPrecisionLinear<B: Backend> {
    kernel: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    bias: Option<Allocation<B>>,
    weights: Allocation<B>,
    input_dim: usize,
    output_dim: usize,
    data_type: DataType,
}

impl<B: Backend> FullPrecisionLinear<B> {
    pub fn new(
        context: &B::Context,
        input_dim: usize,
        output_dim: usize,
        data_type: DataType,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<Self, FullPrecisionLinearError<B>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(FullPrecisionLinearError::UnsupportedDataType(data_type));
        }
        let weights = parameter_tree
            .subtree("weights")?
            .leaf("weights")?
            .validate(&[output_dim, input_dim], data_type)?
            .read_allocation()?;

        let bias = match parameter_tree.leaf("biases") {
            Ok(biases_leaf) => Some(biases_leaf.validate(&[output_dim], data_type)?.read_allocation()?),
            Err(_) => None,
        };

        let kernel = <B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?;

        Ok(Self {
            kernel: RefCell::new(kernel),
            bias,
            weights,
            input_dim,
            output_dim,
            data_type,
        })
    }
}

impl<B: Backend> Linear<B> for FullPrecisionLinear<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.data_type))?;
        self.kernel.borrow_mut().encode(
            MatmulArguments {
                a: &input,
                a_offset: 0,
                b: &self.weights,
                b_offset: 0,
                b_leading_dimension: None,
                b_transpose: true,
                ab_scale: 1.0,
                c: match self.bias.as_ref() {
                    Some(bias) => MatmulArgumentC::Bias(bias),
                    None => MatmulArgumentC::None,
                },
                d: &mut output,
                batch_dim: batch_dim as u32,
                input_dim: self.input_dim as u32,
                output_dim: self.output_dim as u32,
            },
            encoder,
        );
        Ok(output)
    }
}
