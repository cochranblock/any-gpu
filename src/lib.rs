//! candle-vulkan — Vulkan compute backend for candle tensor ops via wgpu.
//! GPU-agnostic ML training. AMD, NVIDIA, Intel. No CUDA lock-in.
// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6

mod device;
mod ops;

pub use device::{GpuBuffer, VulkanDevice};
