// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6
//
// Nearest-neighbor upsampling for UNet decoder path.

use crate::device::{GpuBuffer, GpuDevice};
use anyhow::{ensure, Result};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpsampleParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    _pad: [u32; 2],
}

const SHADER_UPSAMPLE_NEAREST: &str = "
struct P { batch: u32, channels: u32, in_h: u32, in_w: u32, out_h: u32, out_w: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.channels * p.out_h * p.out_w;
    if idx >= total { return; }

    let ow = idx % p.out_w;
    let oh = (idx / p.out_w) % p.out_h;
    let c  = (idx / (p.out_w * p.out_h)) % p.channels;
    let n  = idx / (p.out_w * p.out_h * p.channels);

    let ih = oh * p.in_h / p.out_h;
    let iw = ow * p.in_w / p.out_w;

    let in_idx = n * (p.channels * p.in_h * p.in_w)
               + c * (p.in_h * p.in_w)
               + ih * p.in_w + iw;
    out[idx] = input[in_idx];
}
";

impl GpuDevice {
    /// Nearest-neighbor 2D upsample. Input: [N,C,H,W], output: [N,C,H*scale_h,W*scale_w].
    pub fn upsample_nearest2d(
        &self,
        input: &GpuBuffer,
        batch: u32, channels: u32, in_h: u32, in_w: u32,
        scale_h: u32, scale_w: u32,
    ) -> Result<GpuBuffer> {
        ensure!(input.len == (batch * channels * in_h * in_w) as usize);
        let out_h = in_h * scale_h;
        let out_w = in_w * scale_w;
        let total = batch * channels * out_h * out_w;
        let out = self.alloc(total as usize);
        let params = UpsampleParams { batch, channels, in_h, in_w, out_h, out_w, _pad: [0; 2] };
        self.dispatch_shader(
            SHADER_UPSAMPLE_NEAREST, Some("upsample"),
            &params, &[input], &out,
            super::dispatch_1d(total),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    static DEV: LazyLock<GpuDevice> = LazyLock::new(|| GpuDevice::gpu().expect("need a GPU"));

    #[test]
    fn test_upsample_2x() {
        // 1x1x2x2 -> 1x1x4x4
        let input = DEV.upload(&[1.0, 2.0, 3.0, 4.0]);
        let out = DEV.upsample_nearest2d(&input, 1, 1, 2, 2, 2, 2).unwrap();
        let result = DEV.read(&out).unwrap();
        assert_eq!(result, vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ]);
    }

    #[test]
    fn test_upsample_multichannel() {
        // 1x2x1x1 (2 channels, 1x1 spatial) -> 1x2x2x2
        let input = DEV.upload(&[5.0, 10.0]);
        let out = DEV.upsample_nearest2d(&input, 1, 2, 1, 1, 2, 2).unwrap();
        let result = DEV.read(&out).unwrap();
        assert_eq!(result, vec![5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0]);
    }
}
