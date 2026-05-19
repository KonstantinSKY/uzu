use std::{fs::File, path::Path, rc::Rc};

use crate::{
    DataType,
    backends::common::{Backend, Context},
    classifier::ClassifierError,
    config::{decoder::DecoderConfig, model::classifier_model::ClassifierModelConfig},
    encodable_block::{
        ClassifierLayer, ClassifierPredictionHead, Embedding, Linear, Normalization, Pooling, QkUnpack, Rope,
    },
    forward_pass::{config::classifier::ClassifierForwardPassConfig, model_shape::ModelShape, state::SharedBuffers},
    parameters::ParameterLoader,
    session::types::Error,
};

pub struct ClassifierContext<B: Backend> {
    pub context: Rc<B::Context>,

    pub shared_buffers: Rc<SharedBuffers<B>>,

    pub model_config: ClassifierModelConfig,
    #[cfg(feature = "tracing")]
    pub forward_pass_config: ClassifierForwardPassConfig,
    #[cfg(feature = "tracing")]
    pub model_shape: ModelShape,

    pub embed: Embedding<B>,
    pub embedding_norm: Normalization<B>,
    pub layers: Box<[ClassifierLayer<B>]>,
    pub output_norm: Normalization<B>,

    pub pooling: Pooling<B>,
    pub prediction_head: ClassifierPredictionHead<B>,
    pub logits_data_type: DataType,
}

impl<B: Backend> ClassifierContext<B> {
    pub fn new(
        model_path: &Path,
        model_config: &ClassifierModelConfig,
    ) -> Result<Self, Error> {
        let context = B::Context::new().map_err(|e| Error::UnableToCreateContext(e.into()))?;
        let classifier_config = &model_config.classifier_config;

        let decoder_config = Rc::new(DecoderConfig {
            embedding_config: classifier_config.embedding_config.clone(),
            transformer_config: classifier_config.transformer_config.clone(),
            vocab_size: classifier_config.vocab_size,
            pard_token: None,
            ple_model_config: None,
        });

        let weights_path = model_path.join("model.safetensors");
        if !weights_path.exists() {
            return Err(Error::UnableToLoadWeights);
        }
        let weights_file = File::open(&weights_path).map_err(|_| Error::UnableToLoadWeights)?;
        let loader = ParameterLoader::new(&weights_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let root_loader_view = loader
            .tree()
            .subtree("classifier")
            .map_err(|_| Error::Classifier(ClassifierError::WeightSubtreeNotFound("classifier".to_string())))?;

        let model_shape = ModelShape::from_decoder_config(&decoder_config);
        let forward_pass_config = ClassifierForwardPassConfig::new_for_inference();
        let activation_data_type = forward_pass_config.embedding_forward_pass_config.activation_data_type;

        let mut shared_buffers = SharedBuffers::new(
            context.as_ref(),
            &decoder_config,
            &forward_pass_config.transformer_forward_pass_config,
        );
        shared_buffers.update_data(&root_loader_view)?;
        let shared_buffers = Rc::new(shared_buffers);

        let transformer_tree = root_loader_view
            .subtree("transformer")
            .map_err(|_| Error::Classifier(ClassifierError::WeightSubtreeNotFound("transformer".to_string())))?;

        let output_norm_tree = transformer_tree
            .subtree("output_norm")
            .map_err(|_| Error::Classifier(ClassifierError::WeightSubtreeNotFound("output_norm".to_string())))?;

        let embed = Embedding::new(
            context.as_ref(),
            decoder_config.vocab_size as u32,
            decoder_config.transformer_config.model_dim as u32,
            &decoder_config.embedding_config,
            &root_loader_view.subtree("embedding").expect("Failed to get embedding subtree"),
            &model_shape,
            &forward_pass_config.embedding_forward_pass_config,
        )
        .expect("Failed to create embedding");

        let rope = Rc::new(
            Rope::<B>::new(
                context.as_ref(),
                activation_data_type,
                forward_pass_config.transformer_forward_pass_config.mixer_forward_pass_config.rope_data_type,
            )
                .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("RoPE: {:?}", e))))?,
        );
        let qk_unpack = Rc::new(
            QkUnpack::<B>::new(context.as_ref(), activation_data_type)
                .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("QkUnpack: {:?}", e))))?,
        );

        let layers = classifier_config
            .transformer_config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| {
                layer_config.mixer_config.as_attention().ok_or(ClassifierError::NonAttentionMixer)?;

                let layer_tree = transformer_tree
                    .subtree(&format!("layers.{}", layer_index))
                    .map_err(|_| ClassifierError::WeightSubtreeNotFound(format!("layers.{}", layer_index)))?;

                Ok(ClassifierLayer::new(
                    context.as_ref(),
                    &classifier_config.transformer_config,
                    layer_config,
                    layer_index,
                    &layer_tree,
                    rope.clone(),
                    qk_unpack.clone(),
                    &forward_pass_config.transformer_forward_pass_config,
                    activation_data_type,
                    model_shape.weights_data_type,
                ))
            })
            .collect::<Result<Vec<_>, ClassifierError>>()
            .map_err(Error::Classifier)?
            .into_boxed_slice();

        let output_norm = Normalization::new(
            context.as_ref(),
            activation_data_type,
            classifier_config.transformer_config.model_dim,
            &forward_pass_config.normalization_forward_pass_config,
            classifier_config.transformer_config.output_norm_config.clone(),
            &output_norm_tree,
        )
        .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("output norm: {:?}", e))))?;

        let embedding_norm_tree = root_loader_view
            .subtree("embedding_norm")
            .map_err(|_| Error::Classifier(ClassifierError::WeightSubtreeNotFound("embedding_norm".to_string())))?;
        let embedding_norm = Normalization::new(
            context.as_ref(),
            activation_data_type,
            classifier_config.transformer_config.model_dim,
            &forward_pass_config.normalization_forward_pass_config,
            classifier_config.embedding_norm_config.clone(),
            &embedding_norm_tree,
        )
        .map_err(|e| Error::Classifier(ClassifierError::KernelCreationFailed(format!("embedding norm: {:?}", e))))?;

        let model_dim = classifier_config.model_dim;
        let num_labels = classifier_config.num_labels;
        let prediction_head_config = &classifier_config.prediction_head_config;
        let prediction_head_tree = root_loader_view
            .subtree("prediction_head")
            .map_err(|_| Error::Classifier(ClassifierError::WeightSubtreeNotFound("prediction_head".to_string())))?;

        let prediction_head_dense_tree = prediction_head_tree.subtree("dense").map_err(|_| {
            Error::Classifier(ClassifierError::WeightSubtreeNotFound("prediction_head.dense".to_string()))
        })?;
        let prediction_head_data_type = model_shape.weights_data_type;
        let prediction_head_dense = <dyn Linear<B>>::new::<1>(
            model_dim,
            [model_dim],
            context.as_ref(),
            prediction_head_data_type,
            &prediction_head_dense_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head dense: {:?}", e)))
        })?;

        let prediction_head_norm_tree = prediction_head_tree.subtree("norm").map_err(|_| {
            Error::Classifier(ClassifierError::WeightSubtreeNotFound("prediction_head.norm".to_string()))
        })?;
        let prediction_head_norm = Normalization::new(
            context.as_ref(),
            prediction_head_data_type,
            model_dim,
            &forward_pass_config.normalization_forward_pass_config,
            prediction_head_config.normalization_config.clone(),
            &prediction_head_norm_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head norm: {:?}", e)))
        })?;

        let prediction_head_readout_tree = prediction_head_tree.subtree("readout").map_err(|_| {
            Error::Classifier(ClassifierError::WeightSubtreeNotFound("prediction_head.readout".to_string()))
        })?;
        let logits_data_type = model_shape.weights_data_type;
        let prediction_head_final_linear = <dyn Linear<B>>::new::<1>(
            model_dim,
            [num_labels],
            context.as_ref(),
            logits_data_type,
            &prediction_head_readout_tree,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!(
                "prediction head readout: {:?}",
                e
            )))
        })?;

        let pooling = Pooling::<B>::new(
            context.as_ref(),
            activation_data_type,
            classifier_config.classifier_pooling.clone(),
            model_dim,
        )
        .map_err(|e| {
            eprintln!("Failed to create pooling: {:?}", e);
            Error::UnableToCreateContext(e.into())
        })?;

        let prediction_head = ClassifierPredictionHead::new(
            context.as_ref(),
            prediction_head_dense,
            prediction_head_config.activation.clone().into(),
            prediction_head_data_type,
            prediction_head_norm,
            prediction_head_final_linear,
            model_dim,
        )
        .map_err(|e| {
            Error::Classifier(ClassifierError::KernelCreationFailed(format!("prediction head activation: {:?}", e)))
        })?;

        Ok(Self {
            context,
            shared_buffers,
            model_config: model_config.clone(),
            #[cfg(feature = "tracing")]
            forward_pass_config,
            #[cfg(feature = "tracing")]
            model_shape,
            embed,
            embedding_norm,
            layers,
            output_norm,
            pooling,
            prediction_head,
            logits_data_type,
        })
    }
}
