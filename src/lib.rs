//! any-gpu — Tensor engine for every GPU. AMD, NVIDIA, Intel, Apple.
//! One codebase, zero vendor lock-in. wgpu under the hood.
// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

mod device;
mod ops;

pub use device::{GpuBuffer, GpuDevice};
