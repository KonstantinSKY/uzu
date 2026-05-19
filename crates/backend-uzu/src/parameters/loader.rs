use std::{
    collections::{HashMap, hash_map::Keys},
    fs::File,
};

use thiserror::Error;

use super::safetensors_metadata::{HashMetadata as STMetadata, HeaderLoadingError, read_metadata as read_st_metadata};
use crate::{
    ArrayElement, DataType,
    backends::common::{Allocation, AllocationType, AsBufferRangeRef, Backend, Context, DenseBuffer},
    utils::{fs::file_read_exact_at, strict_serde::DeserializeStrictOwned},
};

// TODO: This entire file (and siblings) are utter garbage, rewrite

pub struct ParameterMetadata {
    shape: Box<[usize]>,
    data_type: DataType,
    offset: usize,
    size: usize,
}

fn st_metadata_into_index_metadata(
    global_offset: usize,
    st_metadata: STMetadata,
) -> (HashMap<String, ParameterMetadata>, HashMap<String, String>) {
    (
        st_metadata
            .tensors
            .into_iter()
            .map(|(key, value)| {
                let (local_begin, local_end) = value.data_offsets;
                let actual_local_offset = local_begin;
                let actual_size = local_end - local_begin;
                let weight_metadata = ParameterMetadata {
                    shape: value.shape.into(),
                    data_type: value.dtype.into(),
                    offset: global_offset + actual_local_offset,
                    size: actual_size,
                };
                (key, weight_metadata)
            })
            .collect(),
        st_metadata.metadata.unwrap_or_default(),
    )
}

#[derive(Debug, Error)]
pub enum ParameterLoaderError<B: Backend> {
    #[error("Array with key \"{0}\" not found.")]
    KeyNotFound(String),
    #[error("Couldn't find any arrays with prefix \"{0}\".")]
    SubtreeNotFound(String),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Failed to read data")]
    ArrayLoadingError(#[from] std::io::Error),
    #[error("Failed to deserialize metadata")]
    MetadataDeserializationError(#[from] serde_json::Error),
    #[error("Invalid tensor: got {shape:?} @ {data_type:?}, expected {expected_shape:?} @ {expected_data_type:?}")]
    InvalidTensor {
        shape: Box<[usize]>,
        data_type: DataType,
        expected_shape: Box<[usize]>,
        expected_data_type: DataType,
    },
}

pub struct ParameterLoader<'context, 'file, C: Context>
where
    'file: 'context,
{
    context: &'context C,
    index: HashMap<String, ParameterMetadata>,
    metadata: HashMap<String, String>,
    file: &'file File,
}

impl<'file, 'context, C: Context> ParameterLoader<'file, 'context, C>
where
    'file: 'context,
{
    pub fn new(
        file: &'file File,
        context: &'context C,
    ) -> Result<Self, HeaderLoadingError> {
        let (global_offset, st_metadata) = read_st_metadata(file)?;
        let (index, metadata) = st_metadata_into_index_metadata(global_offset, st_metadata);
        Ok(ParameterLoader {
            context,
            index,
            metadata,
            file,
        })
    }

    pub fn keys(&self) -> Keys<'_, String, ParameterMetadata> {
        self.index.keys()
    }

    fn get_leaf<'leaf>(
        &'leaf self,
        key: &str,
    ) -> Result<ParameterLeaf<'file, 'context, 'leaf, C, false>, ParameterLoaderError<C::Backend>> {
        Ok(ParameterLeaf {
            metadata: self.index.get(key).ok_or_else(|| ParameterLoaderError::KeyNotFound(key.to_string()))?,
            loader: self,
        })
    }

    pub fn tree<'loader>(&'loader self) -> ParameterTree<'loader, C> {
        ParameterTree {
            loader: self,
            prefix: None,
        }
    }
}

pub struct ParameterLeaf<'file, 'context, 'leaf, C: Context, const VALIDATED: bool> {
    metadata: &'leaf ParameterMetadata,
    loader: &'leaf ParameterLoader<'context, 'file, C>,
}

impl<'file, 'context, 'leaf, C: Context> ParameterLeaf<'file, 'context, 'leaf, C, false> {
    pub fn validate(
        self,
        expected_shape: &[usize],
        expected_data_type: DataType,
    ) -> Result<ParameterLeaf<'file, 'context, 'leaf, C, true>, ParameterLoaderError<C::Backend>> {
        let shape = self.metadata.shape.as_ref();
        let data_type = self.metadata.data_type;
        if (shape, data_type) != (expected_shape, expected_data_type) {
            return Err(ParameterLoaderError::InvalidTensor {
                shape: shape.into(),
                data_type,
                expected_shape: expected_shape.into(),
                expected_data_type,
            });
        }
        Ok(ParameterLeaf {
            metadata: self.metadata,
            loader: self.loader,
        })
    }

    #[cfg(test)]
    pub fn unvalidated(self) -> ParameterLeaf<'file, 'context, 'leaf, C, true> {
        ParameterLeaf {
            metadata: self.metadata,
            loader: self.loader,
        }
    }
}

impl<'file, 'context, 'leaf, C: Context> ParameterLeaf<'file, 'context, 'leaf, C, true> {
    pub fn read_slice<T: ArrayElement>(&self) -> Result<Box<[T]>, ParameterLoaderError<C::Backend>> {
        let element_count = self.metadata.size / std::mem::size_of::<T>();
        let mut data = vec![T::zeroed(); element_count];
        file_read_exact_at(self.loader.file, bytemuck::cast_slice_mut(&mut data), self.metadata.offset as u64)?;
        Ok(data.into_boxed_slice())
    }

    pub fn read_allocation(&self) -> Result<Allocation<C::Backend>, ParameterLoaderError<C::Backend>> {
        let allocation = self
            .loader
            .context
            .create_allocation(self.metadata.size, AllocationType::Global)
            .map_err(ParameterLoaderError::BackendError)?;
        let buffer_range = allocation.as_buffer_range_ref();
        let range = buffer_range.range();
        file_read_exact_at(
            self.loader.file,
            unsafe {
                std::slice::from_raw_parts_mut(
                    (buffer_range.buffer().cpu_ptr().as_ptr() as *mut u8).add(range.start),
                    range.len(),
                )
            },
            self.metadata.offset as u64,
        )?;
        Ok(allocation)
    }
}

pub struct ParameterTree<'loader, C: Context> {
    loader: &'loader ParameterLoader<'loader, 'loader, C>,
    prefix: Option<String>,
}

impl<'loader, C: Context> ParameterTree<'loader, C> {
    pub fn path_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    fn join_prefix(
        &self,
        name: &str,
    ) -> String {
        self.prefix.as_ref().map_or_else(|| name.to_string(), |p| format!("{p}.{name}"))
    }

    pub fn subtree(
        &self,
        name: &str,
    ) -> Result<Self, ParameterLoaderError<C::Backend>> {
        let new_prefix = self.join_prefix(name);
        let num_suffixes = self.loader.keys().filter_map(|suffix| suffix.strip_prefix(&new_prefix)).count();
        if num_suffixes > 0 {
            Ok(Self {
                loader: self.loader,
                prefix: Some(new_prefix),
            })
        } else {
            Err(ParameterLoaderError::SubtreeNotFound(name.to_string()))
        }
    }

    pub fn leaf<'leaf>(
        &'leaf self,
        name: &str,
    ) -> Result<ParameterLeaf<'loader, 'loader, 'leaf, C, false>, ParameterLoaderError<C::Backend>> {
        self.loader.get_leaf(&self.join_prefix(name))
    }

    pub fn metadata<T: DeserializeStrictOwned>(
        &self,
        name: &str,
    ) -> Result<T, ParameterLoaderError<C::Backend>> {
        let new_prefix = self.join_prefix(name);

        Ok(serde_json::from_str(
            self.loader.metadata.get(&new_prefix).ok_or(ParameterLoaderError::KeyNotFound(new_prefix))?,
        )?)
    }
}
