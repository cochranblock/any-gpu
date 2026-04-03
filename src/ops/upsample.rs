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
    fn dev() -> &'static GpuDevice { &crate::ops::TEST_DEV }

    // CPU reference upsample
    fn cpu_upsample(input: &[f32], batch: usize, ch: usize, h: usize, w: usize, sh: usize, sw: usize) -> Vec<f32> {
        let oh = h * sh; let ow = w * sw;
        let mut out = vec![0.0f32; batch * ch * oh * ow];
        for n in 0..batch {
            for c in 0..ch {
                for y in 0..oh {
                    for x in 0..ow {
                        let iy = y * h / oh; let ix = x * w / ow;
                        out[n*ch*oh*ow + c*oh*ow + y*ow + x] = input[n*ch*h*w + c*h*w + iy*w + ix];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_upsample_2x() {
        let input = dev().upload(&[1.0, 2.0, 3.0, 4.0]);
        let result = dev().read(&dev().upsample_nearest2d(&input, 1, 1, 2, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(result, vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ]);
    }

    #[test]
    fn test_upsample_3x_vs_cpu() {
        // Non-power-of-2 scale
        let data: Vec<f32> = (1..=6).map(|x| x as f32).collect(); // 1x1x2x3
        let expected = cpu_upsample(&data, 1, 1, 2, 3, 3, 3);
        let result = dev().read(&dev().upsample_nearest2d(&dev().upload(&data), 1, 1, 2, 3, 3, 3).unwrap()).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_upsample_batched_multichannel_vs_cpu() {
        // batch=2, channels=3, 2x2 spatial, scale 2x
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let expected = cpu_upsample(&data, 2, 3, 2, 2, 2, 2);
        let result = dev().read(&dev().upsample_nearest2d(&dev().upload(&data), 2, 3, 2, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_upsample_1x1() {
        // 1x1 spatial -> 3x3 spatial
        let result = dev().read(&dev().upsample_nearest2d(&dev().upload(&[7.0]), 1, 1, 1, 1, 3, 3).unwrap()).unwrap();
        assert_eq!(result, vec![7.0; 9]);
    }
}
