use super::RopeBuffers;
use crate::{
    DataType,
    array::ArrayContextExt,
    backends::common::{Allocation, AsBufferRangeMut, Backend, DenseBuffer},
    config::{decoder::DecoderConfig, rope::AnyRoPEConfig},
    forward_pass::config::transformer::TransformerForwardPassConfig,
    parameters::ParameterTree,
    session::types::Error,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerRopeKind {
    NoKernel,
    Indexed(usize),
}

pub struct SharedBuffers<B: Backend> {
    pub rope_buffers: Box<[RopeBuffers<B>]>,
    layer_rope_kinds: Box<[LayerRopeKind]>,
    pub attention_sinks: Box<[Option<Allocation<B>>]>,
}

impl<B: Backend> SharedBuffers<B> {
    pub fn new(
        context: &B::Context,
        decoder_config: &DecoderConfig,
        forward_pass_config: &TransformerForwardPassConfig,
    ) -> Self {
        let tf = &decoder_config.transformer_config;

        let mut configs = Vec::<(AnyRoPEConfig, usize)>::new();
        let layer_rope_kinds: Box<[LayerRopeKind]> = tf
            .layer_configs
            .iter()
            .map(|layer_config| {
                let Some(attention_config) = layer_config.mixer_config.as_attention() else {
                    return LayerRopeKind::NoKernel;
                };
                let Some(rope_config) = &layer_config.rope_config else {
                    return LayerRopeKind::NoKernel;
                };
                let head_dim = rope_config.head_dim().unwrap_or(attention_config.head_dim);
                let index = configs
                    .iter()
                    .position(|(existing_config, existing_head_dim)| {
                        existing_config == rope_config && *existing_head_dim == head_dim
                    })
                    .unwrap_or_else(|| {
                        configs.push((rope_config.clone(), head_dim));
                        configs.len() - 1
                    });
                LayerRopeKind::Indexed(index)
            })
            .collect();

        let rope_buffers: Box<[RopeBuffers<B>]> = configs
            .iter()
            .map(|(config, head_dim)| {
                RopeBuffers::new(
                    context,
                    *config.max_sequence_length(),
                    *head_dim,
                    forward_pass_config.mixer_forward_pass_config.rope_data_type,
                )
            })
            .collect();

        let attention_sinks = tf
            .layer_configs
            .iter()
            .map(|layer| {
                let attn = layer.mixer_config.as_attention()?;
                attn.has_sinks
                    .then(|| context.create_array_uninitialized(&[attn.num_heads], DataType::F32).into_allocation())
            })
            .collect();

        Self {
            rope_buffers,
            layer_rope_kinds,
            attention_sinks,
        }
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
    ) -> Result<(), Error> {
        let transformer_tree = parameter_tree.subtree("transformer").map_err(|_| Error::UnableToLoadWeights)?;
        self.update_data_from_transformer_tree(&transformer_tree)?;
        Ok(())
    }

    pub fn update_data_from_transformer_tree(
        &mut self,
        transformer_tree: &ParameterTree<B::Context>,
    ) -> Result<(), Error> {
        for (rope_index, rope_buffers) in self.rope_buffers.iter_mut().enumerate() {
            rope_buffers.update_data(transformer_tree, rope_index)?;
        }
        for (layer_idx, sink_cell) in self.attention_sinks.iter_mut().enumerate() {
            let Some(sink_cell) = sink_cell.as_mut() else {
                continue;
            };
            let layer_tree = transformer_tree.subtree(&format!("layers.{}", layer_idx)).unwrap();
            let attn_tree = layer_tree.subtree("mixer").unwrap();
            let dst_slice = unsafe {
                let buffer_range = sink_cell.as_buffer_range_mut();
                let range = buffer_range.range();
                std::slice::from_raw_parts_mut(
                    (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start) as *mut f32,
                    range.len() / std::mem::size_of::<f32>(),
                )
            };

            let sinks_leaf = attn_tree.leaf("sinks").unwrap();
            let sinks_leaf = sinks_leaf.validate(&[dst_slice.len()], DataType::F32).unwrap();
            let src = sinks_leaf.read_slice::<f32>().unwrap();
            dst_slice.copy_from_slice(&src);
        }
        Ok(())
    }

    pub fn rope_buffers_for_layer(
        &self,
        layer_index: usize,
    ) -> Option<&RopeBuffers<B>> {
        match self.layer_rope_kinds[layer_index] {
            LayerRopeKind::NoKernel => None,
            LayerRopeKind::Indexed(index) => Some(&self.rope_buffers[index]),
        }
    }

    pub fn attention_sinks(
        &self,
        layer_index: usize,
    ) -> Option<&Allocation<B>> {
        self.attention_sinks.get(layer_index)?.as_ref()
    }
}
