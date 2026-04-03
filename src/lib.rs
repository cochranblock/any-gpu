//! any-gpu — Bare metal tensor engine. Any GPU, zero vendor lock-in.
//! AMD, NVIDIA, Intel, Apple. wgpu under the hood.
// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

mod device;
mod ops;

pub use device::{GpuBuffer, GpuDevice};
