//! any-gpu — Tensor engine for every GPU. AMD, NVIDIA, Intel, Apple.
//! One codebase, zero vendor lock-in. wgpu under the hood.
//! Token-Optimized Code Representation per `docs/compression_map.md`.
//! t500=GpuDevice, t501=GpuBuffer, t502=Tensor.
// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7

#![allow(non_camel_case_types, non_snake_case, dead_code, unused_imports)]

mod device;
mod ops;
mod tensor;

// Training-side modules pull in std::fs and CPU autograd machinery that don't
// belong in a browser bundle. Forward-only inference is all wasm32 needs.
#[cfg(not(target_arch = "wasm32"))]
pub mod autograd;
#[cfg(not(target_arch = "wasm32"))]
pub mod nanosign;
#[cfg(not(target_arch = "wasm32"))]
pub mod safetensors;
#[cfg(not(target_arch = "wasm32"))]
pub mod pager;
#[cfg(not(target_arch = "wasm32"))]
pub mod optim;
#[cfg(not(target_arch = "wasm32"))]
pub mod train;

pub use device::{t500, t501};
pub use ops::transformer::t534;
pub use pager::{t539, DEFAULT_STAGE_BYTES};
pub use safetensors::t538;
pub use tensor::t502;
