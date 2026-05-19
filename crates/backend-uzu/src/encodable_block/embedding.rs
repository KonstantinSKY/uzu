use std::cell::RefCell;

use thiserror::Error;

use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::{
            FullPrecisionEmbeddingLookupKernel, ManualKernels, QuantizedEmbeddingLookupKernel,
            matmul::{MatmulArgumentC, MatmulArguments, MatmulError, MatmulKernel},
            quant_matmul::{
                QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
                QuantizedMatmulKernelEncodable,
            },
        },
    },
    config::{
        embedding::AnyEmbeddingConfig,
        weight_matrix::{
            AnyWeightMatrixSpec, Layout,
            awq_spec::AWQSpec,
            full_precision_spec::FullPrecisionSpec,
            hybrid_spec::{HybridSpec, IncoherenceProcessingMode},
            mlx_spec::MLXSpec,
        },
    },
    forward_pass::{config::embedding::EmbeddingForwardPassConfig, model_shape::ModelShape},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum EmbeddingError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Matmul error: {0}")]
    MatmulError(#[from] MatmulError<B>),
    #[error("QuantizedMatmul error: {0}")]
    QuantizedMatmulError(#[from] QuantizedMatmulError<B>),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

enum TiedEmbeddingType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Allocation<B>,
        quantization_method: QuantizationMethod,
        output_hadamard_factors: Option<Allocation<B>>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
        readout: QuantizedMatmulKernelEncodable<B>,
    },
}

enum UntiedEmbeddingLookupType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        lookup: <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Allocation<B>,
        quantization_method: QuantizationMethod,
        output_hadamard_factors: Option<Allocation<B>>,
        lookup: <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel,
    },
}

enum UntiedEmbeddingReadoutType<B: Backend> {
    FullPrecision {
        weights: Allocation<B>,
        readout: RefCell<<B::Kernels as ManualKernels>::MatmulKernel>,
    },
    Quantized {
        weights: Allocation<B>,
        scales: Allocation<B>,
        zero_points_or_biases: Allocation<B>,
        readout: QuantizedMatmulKernelEncodable<B>,
    },
}

enum EmbeddingTying<B: Backend> {
    Tied {
        ty: TiedEmbeddingType<B>,
    },
    Untied {
        input_ty: UntiedEmbeddingLookupType<B>,
        output_ty: UntiedEmbeddingReadoutType<B>,
    },
}

pub struct Embedding<B: Backend> {
    tying: EmbeddingTying<B>,
    readout_input_hadamard_factors: Option<Allocation<B>>,
    input_scale: f32,
    data_type: DataType,
    vocab_size: u32,
    model_dim: u32,
}

impl<B: Backend> Embedding<B> {
    pub fn new(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &AnyEmbeddingConfig,
        parameter_tree: &ParameterTree<B::Context>,
        model_shape: &ModelShape,
        forward_pass_config: &EmbeddingForwardPassConfig,
    ) -> Result<Self, EmbeddingError<B>> {
        Self::new_with_lookup_and_readout_trees(
            context,
            vocab_size,
            model_dim,
            config,
            parameter_tree,
            parameter_tree,
            model_shape,
            forward_pass_config,
        )
    }

    pub fn new_with_lookup_and_readout_trees(
        context: &B::Context,
        vocab_size: u32,
        model_dim: u32,
        config: &AnyEmbeddingConfig,
        lookup_tree: &ParameterTree<B::Context>,
        readout_tree: &ParameterTree<B::Context>,
        model_shape: &ModelShape,
        forward_pass_config: &EmbeddingForwardPassConfig,
    ) -> Result<Self, EmbeddingError<B>> {
        let data_type = forward_pass_config.activation_data_type;
        assert_eq!(
            data_type, model_shape.weights_data_type,
            "embedding kernels require activation dtype to match safetensors weight dtype"
        );
        let (tying, readout_input_hadamard_factors, data_type) = match config {
            AnyEmbeddingConfig::TiedEmbeddingConfig(_) => {
                let embedding_tree = lookup_tree.subtree("embedding")?;
                let embedding_spec = embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let (ty, readout_input_hadamard_factors, data_type) = match embedding_spec {
                    AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                        layout: Layout::InputOutput,
                        ..
                    }) => {
                        let weights = embedding_tree
                            .leaf("weights")?
                            .validate(&[vocab_size as usize, model_dim as usize], data_type)?
                            .read_allocation()?;

                        let lookup =
                            <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                                .map_err(EmbeddingError::BackendError)?;
                        let readout =
                            RefCell::new(<B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?);

                        (
                            TiedEmbeddingType::FullPrecision {
                                weights,
                                lookup,
                                readout,
                            },
                            None,
                            data_type,
                        )
                    },
                    AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                        bits,
                        group_size,
                        layout: Layout::InputOutput,
                        ..
                    }) => {
                        let embedding_quantization_mode = match bits {
                            4 => QuantizationMode::U4,
                            8 => QuantizationMode::U8,
                            _ => {
                                return Err(EmbeddingError::UnsupportedConfiguration(format!(
                                    "MLXSpec {{ bits: {bits}, group_size: {group_size}, layout: InputOutput }}"
                                )));
                            },
                        };
                        let packing_divisor = embedding_quantization_mode.packing_divisor();
                        let storage_data_type = embedding_quantization_mode.storage_type();
                        let num_groups = (model_dim as usize).div_ceil(group_size);

                        let weights = embedding_tree
                            .leaf("weights")?
                            .validate(
                                &[vocab_size as usize, model_dim as usize / packing_divisor],
                                storage_data_type,
                            )?
                            .read_allocation()?;
                        let scales = embedding_tree
                            .leaf("scales")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;
                        let zero_points_or_biases = embedding_tree
                            .leaf("biases")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;

                        let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                            context,
                            data_type,
                            group_size as u32,
                            embedding_quantization_mode,
                            QuantizationMethod::ScaleBias,
                            false,
                        )
                        .map_err(EmbeddingError::BackendError)?;
                        let readout = QuantizedMatmulKernelEncodable::new(
                            context,
                            QuantizedMatmulConfiguration {
                                data_type,
                                group_size,
                                input_dim: model_dim as usize,
                                output_dim: vocab_size as usize,
                                mode: embedding_quantization_mode,
                                quantization_method: QuantizationMethod::ScaleBias,
                                use_hadamard: false,
                            },
                        )?;

                        (
                            TiedEmbeddingType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method: QuantizationMethod::ScaleBias,
                                output_hadamard_factors: None,
                                lookup,
                                readout,
                            },
                            None,
                            data_type,
                        )
                    },
                    AnyWeightMatrixSpec::HybridSpec(HybridSpec {
                        quantization_spec,
                        adapter_spec: None,
                        incoherence_block_size: Some(32),
                        incoherence_processing_mode: IncoherenceProcessingMode::Output,
                        ..
                    }) => {
                        let (bits, group_size, quantization_method) = match *quantization_spec {
                            AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                                bits,
                                group_size,
                                layout: Layout::InputOutput,
                                ..
                            }) => (bits, group_size, QuantizationMethod::ScaleBias),
                            AnyWeightMatrixSpec::AWQSpec(AWQSpec {
                                bits,
                                group_size,
                                is_symmetric: false,
                                layout: Layout::InputOutput,
                                ..
                            }) => (bits, group_size, QuantizationMethod::ScaleZeroPoint),
                            spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                        };

                        let quantized_tree = embedding_tree.subtree("quantized")?;
                        let embedding_quantization_mode = match bits {
                            4 => QuantizationMode::U4,
                            8 => QuantizationMode::U8,
                            _ => {
                                return Err(EmbeddingError::UnsupportedConfiguration(format!(
                                    "{quantization_method:?} bits={bits}, group_size={group_size}"
                                )));
                            },
                        };
                        let packing_divisor = embedding_quantization_mode.packing_divisor();
                        let storage_data_type = embedding_quantization_mode.storage_type();
                        let num_groups = (model_dim as usize).div_ceil(group_size);

                        let weights = quantized_tree
                            .leaf("weights")?
                            .validate(
                                &[vocab_size as usize, model_dim as usize / packing_divisor],
                                storage_data_type,
                            )?
                            .read_allocation()?;
                        let scales = quantized_tree
                            .leaf("scales")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;

                        let zero_points_or_biases = match quantization_method {
                            QuantizationMethod::ScaleBias => {
                                quantized_tree
                                    .leaf("biases")?
                                    .validate(&[vocab_size as usize, num_groups], data_type)?
                                    .read_allocation()?
                            },
                            QuantizationMethod::ScaleZeroPoint => {
                                let expected_zero_points_entries =
                                    (num_groups + packing_divisor - 1) / packing_divisor;
                                quantized_tree
                                    .leaf("zero_points")?
                                    .validate(
                                        &[vocab_size as usize, expected_zero_points_entries],
                                        storage_data_type,
                                    )?
                                    .read_allocation()?
                            },
                        };

                        let incoherence_signs_tree = embedding_tree.subtree("incoherence_signs")?;
                        let output_hadamard_factors = Some(
                            incoherence_signs_tree
                                .leaf("output_signs")?
                                .validate(&[model_dim as usize], DataType::I32)?
                                .read_allocation()?,
                        );
                        let readout_input_hadamard_factors = Some(
                            incoherence_signs_tree
                                .leaf("output_signs")?
                                .validate(&[model_dim as usize], DataType::I32)?
                                .read_allocation()?,
                        );
                        let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                            context,
                            data_type,
                            group_size as u32,
                            embedding_quantization_mode,
                            quantization_method,
                            true,
                        )
                        .map_err(EmbeddingError::BackendError)?;
                        let readout = QuantizedMatmulKernelEncodable::new(
                            context,
                            QuantizedMatmulConfiguration {
                                data_type,
                                group_size,
                                input_dim: model_dim as usize,
                                output_dim: vocab_size as usize,
                                mode: embedding_quantization_mode,
                                quantization_method,
                                use_hadamard: false,
                            },
                        )?;

                        (
                            TiedEmbeddingType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method,
                                output_hadamard_factors,
                                lookup,
                                readout,
                            },
                            readout_input_hadamard_factors,
                            data_type,
                        )
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                (
                    EmbeddingTying::Tied {
                        ty,
                    },
                    readout_input_hadamard_factors,
                    data_type,
                )
            },
            AnyEmbeddingConfig::UntiedEmbeddingConfig(_) => {
                let input_embedding_tree = lookup_tree.subtree("input_embedding")?;
                let input_embedding_spec = input_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let (input_ty, data_type) = match input_embedding_spec {
                    AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                        layout: Layout::InputOutput,
                        ..
                    }) => {
                        let weights = input_embedding_tree
                            .leaf("weights")?
                            .validate(&[vocab_size as usize, model_dim as usize], data_type)?
                            .read_allocation()?;

                        let lookup =
                            <B::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(context, data_type)
                                .map_err(EmbeddingError::BackendError)?;

                        (
                            UntiedEmbeddingLookupType::FullPrecision {
                                weights,
                                lookup,
                            },
                            data_type,
                        )
                    },
                    AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                        bits,
                        group_size,
                        layout: Layout::InputOutput,
                        ..
                    }) => {
                        let embedding_quantization_mode = match bits {
                            4 => QuantizationMode::U4,
                            8 => QuantizationMode::U8,
                            _ => {
                                return Err(EmbeddingError::UnsupportedConfiguration(format!(
                                    "MLXSpec {{ bits: {bits}, group_size: {group_size}, layout: InputOutput }}"
                                )));
                            },
                        };
                        let packing_divisor = embedding_quantization_mode.packing_divisor();
                        let storage_data_type = embedding_quantization_mode.storage_type();
                        let num_groups = (model_dim as usize).div_ceil(group_size);

                        let weights = input_embedding_tree
                            .leaf("weights")?
                            .validate(
                                &[vocab_size as usize, model_dim as usize / packing_divisor],
                                storage_data_type,
                            )?
                            .read_allocation()?;
                        let scales = input_embedding_tree
                            .leaf("scales")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;
                        let zero_points_or_biases = input_embedding_tree
                            .leaf("biases")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;

                        let lookup = <B::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
                            context,
                            data_type,
                            group_size as u32,
                            embedding_quantization_mode,
                            QuantizationMethod::ScaleBias,
                            false,
                        )
                        .map_err(EmbeddingError::BackendError)?;

                        (
                            UntiedEmbeddingLookupType::Quantized {
                                weights,
                                scales,
                                zero_points_or_biases,
                                quantization_method: QuantizationMethod::ScaleBias,
                                output_hadamard_factors: None,
                                lookup,
                            },
                            data_type,
                        )
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                let output_embedding_tree = readout_tree.subtree("output_embedding")?;
                let output_embedding_spec = output_embedding_tree.metadata::<AnyWeightMatrixSpec>("spec")?;

                let output_ty = match output_embedding_spec {
                    AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
                        layout: Layout::OutputInput,
                        ..
                    }) => {
                        let weights = output_embedding_tree
                            .leaf("weights")?
                            .validate(&[vocab_size as usize, model_dim as usize], data_type)?
                            .read_allocation()?;
                        let readout =
                            RefCell::new(<B::Kernels as ManualKernels>::MatmulKernel::new(context, data_type)?);

                        UntiedEmbeddingReadoutType::FullPrecision {
                            weights,
                            readout,
                        }
                    },
                    AnyWeightMatrixSpec::MLXSpec(MLXSpec {
                        bits,
                        group_size,
                        layout: Layout::OutputInput,
                        ..
                    }) => {
                        let embedding_quantization_mode = match bits {
                            4 => QuantizationMode::U4,
                            8 => QuantizationMode::U8,
                            _ => {
                                return Err(EmbeddingError::UnsupportedConfiguration(format!(
                                    "MLXSpec {{ bits: {bits}, group_size: {group_size}, layout: OutputInput }}"
                                )));
                            },
                        };
                        let packing_divisor = embedding_quantization_mode.packing_divisor();
                        let storage_data_type = embedding_quantization_mode.storage_type();
                        let num_groups = (model_dim as usize).div_ceil(group_size);

                        let weights = output_embedding_tree
                            .leaf("weights")?
                            .validate(
                                &[vocab_size as usize, model_dim as usize / packing_divisor],
                                storage_data_type,
                            )?
                            .read_allocation()?;
                        let scales = output_embedding_tree
                            .leaf("scales")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;
                        let zero_points_or_biases = output_embedding_tree
                            .leaf("biases")?
                            .validate(&[vocab_size as usize, num_groups], data_type)?
                            .read_allocation()?;

                        let readout = QuantizedMatmulKernelEncodable::new(
                            context,
                            QuantizedMatmulConfiguration {
                                data_type,
                                group_size,
                                input_dim: model_dim as usize,
                                output_dim: vocab_size as usize,
                                mode: embedding_quantization_mode,
                                quantization_method: QuantizationMethod::ScaleBias,
                                use_hadamard: false,
                            },
                        )?;

                        UntiedEmbeddingReadoutType::Quantized {
                            weights,
                            scales,
                            zero_points_or_biases,
                            readout,
                        }
                    },
                    spec => return Err(EmbeddingError::UnsupportedConfiguration(format!("{spec:?}"))),
                };

                (
                    EmbeddingTying::Untied {
                        input_ty,
                        output_ty,
                    },
                    None,
                    data_type,
                )
            },
        };

        let input_scale = config.input_scale().unwrap_or(1.0);
        if let Some(logit_soft_cap) = config.logit_soft_cap() {
            return Err(EmbeddingError::UnsupportedConfiguration(format!("logit_soft_cap={logit_soft_cap:?}")));
        }

        Ok(Self {
            tying,
            readout_input_hadamard_factors,
            input_scale,
            data_type,
            vocab_size,
            model_dim,
        })
    }

    pub fn take_readout_input_hadamard_factors(&mut self) -> Option<Allocation<B>> {
        self.readout_input_hadamard_factors.take()
    }

    pub fn encode_lookup(
        &self,
        token_ids: &Allocation<B>,
        batch_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        let mut output = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.model_dim as usize], self.data_type))
            .map_err(EmbeddingError::BackendError)?;
        let batch_dim = batch_dim as u32;

        match &self.tying {
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::FullPrecision {
                        weights,
                        lookup,
                        readout: _,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty:
                    UntiedEmbeddingLookupType::FullPrecision {
                        weights,
                        lookup,
                    },
                output_ty: _,
            } => lookup.encode(
                token_ids,
                weights,
                &mut output,
                batch_dim,
                self.vocab_size,
                self.model_dim,
                self.input_scale,
                encoder,
            ),
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        quantization_method,
                        output_hadamard_factors,
                        lookup,
                        readout: _,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty:
                    UntiedEmbeddingLookupType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        quantization_method,
                        output_hadamard_factors,
                        lookup,
                    },
                output_ty: _,
            } => {
                let (zero_points, biases) = match quantization_method {
                    QuantizationMethod::ScaleZeroPoint => (Some(zero_points_or_biases), None),
                    QuantizationMethod::ScaleBias => (None, Some(zero_points_or_biases)),
                };
                lookup.encode(
                    token_ids,
                    weights,
                    scales,
                    zero_points,
                    biases,
                    &mut output,
                    output_hadamard_factors.as_ref(),
                    batch_dim,
                    self.vocab_size,
                    self.model_dim,
                    self.input_scale,
                    encoder,
                );
            },
        };

        Ok(output)
    }

    pub fn encode_readout(
        &self,
        batch_dim: usize,
        input_allocation: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, EmbeddingError<B>> {
        assert!(batch_dim > 0, "Embedding readout requires at least one row");
        let mut output_allocation = encoder
            .allocate_scratch(size_for_shape(&[batch_dim, self.vocab_size as usize], self.data_type))
            .map_err(EmbeddingError::BackendError)?;

        match &self.tying {
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::FullPrecision {
                        weights,
                        lookup: _,
                        readout,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty: _,
                output_ty:
                    UntiedEmbeddingReadoutType::FullPrecision {
                        weights,
                        readout,
                    },
            } => {
                let input_dim = self.model_dim as usize;
                let output_dim = self.vocab_size as usize;
                readout.borrow_mut().encode(
                    MatmulArguments {
                        a: input_allocation,
                        a_offset: 0,
                        b: weights,
                        b_offset: 0,
                        b_leading_dimension: None,
                        b_transpose: true,
                        ab_scale: 1.0,
                        c: MatmulArgumentC::None,
                        d: &mut output_allocation,
                        batch_dim: batch_dim as u32,
                        input_dim: input_dim as u32,
                        output_dim: output_dim as u32,
                    },
                    encoder,
                );
            },
            EmbeddingTying::Tied {
                ty:
                    TiedEmbeddingType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        quantization_method: _,
                        output_hadamard_factors: _,
                        lookup: _,
                        readout,
                    },
            }
            | EmbeddingTying::Untied {
                input_ty: _,
                output_ty:
                    UntiedEmbeddingReadoutType::Quantized {
                        weights,
                        scales,
                        zero_points_or_biases,
                        readout,
                    },
            } => {
                readout.encode(
                    encoder,
                    QuantizedMatmulArguments {
                        a: input_allocation,
                        a_offset: 0,
                        b: weights,
                        scales,
                        zero_points_or_biases,
                        output: &mut output_allocation,
                        hadamard_factors: None,
                        batch_dim,
                    },
                );
            },
        };

        Ok(output_allocation)
    }
}
