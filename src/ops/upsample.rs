// Unlicense — cochranblock.org
// Contributors: GotEmCoach, KOVA, Claude Opus 4.6, Claude Opus 4.7
//
// Nearest-neighbor upsampling for UNet decoder path.
// f660=upsample_nearest2d, f661=upsample_nearest2d_backward.

use crate::device::{t500, t501};
use anyhow::{ensure, Result};

/// t526 = UpsampleParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t526 {
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

// Backward: each input pixel accumulates gradients from all output pixels that map to it.
const SHADER_UPSAMPLE_NEAREST_BACKWARD: &str = "
struct P { batch: u32, channels: u32, in_h: u32, in_w: u32, out_h: u32, out_w: u32, scale_h: u32, scale_w: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_in: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = p.batch * p.channels * p.in_h * p.in_w;
    if idx >= total { return; }

    let iw = idx % p.in_w;
    let ih = (idx / p.in_w) % p.in_h;
    let c  = (idx / (p.in_w * p.in_h)) % p.channels;
    let n  = idx / (p.in_w * p.in_h * p.channels);

    var sum: f32 = 0.0;
    let base = n * (p.channels * p.out_h * p.out_w) + c * (p.out_h * p.out_w);
    for (var dy: u32 = 0u; dy < p.scale_h; dy++) {
        for (var dx: u32 = 0u; dx < p.scale_w; dx++) {
            let oh = ih * p.scale_h + dy;
            let ow = iw * p.scale_w + dx;
            sum += grad_out[base + oh * p.out_w + ow];
        }
    }
    grad_in[idx] = sum;
}
";

/// t527 = UpsampleBackwardParams.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct t527 {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    scale_h: u32,
    scale_w: u32,
}

impl t500 {
    /// f660 = upsample_nearest2d. Input: [N,C,H,W], output: [N,C,H*scale_h,W*scale_w].
    pub fn f660(
        &self,
        p0: &t501,
        p1: u32, p2: u32, p3: u32, p4: u32,
        p5: u32, p6: u32,
    ) -> Result<t501> {
        ensure!(p0.s507 == (p1 * p2 * p3 * p4) as usize);
        let v0 = p3 * p5;
        let v1 = p4 * p6;
        let v2 = p1 * p2 * v0 * v1;
        let v3 = self.f503(v2 as usize);
        let v4 = t526 { batch: p1, channels: p2, in_h: p3, in_w: p4, out_h: v0, out_w: v1, _pad: [0; 2] };
        self.f543(
            SHADER_UPSAMPLE_NEAREST, Some("upsample"),
            &v4, &[p0], &v3,
            super::f540(v2),
        );
        Ok(v3)
    }

    /// f661 = upsample_nearest2d_backward. Each input pixel accumulates gradients
    /// from the scale_h*scale_w output block that mapped to it.
    pub fn f661(
        &self,
        p0: &t501,
        p1: u32, p2: u32, p3: u32, p4: u32,
        p5: u32, p6: u32,
    ) -> Result<t501> {
        let v0 = p3 * p5;
        let v1 = p4 * p6;
        ensure!(p0.s507 == (p1 * p2 * v0 * v1) as usize);
        let v2 = p1 * p2 * p3 * p4;
        let v3 = self.f503(v2 as usize);
        let v4 = t527 {
            batch: p1, channels: p2, in_h: p3, in_w: p4, out_h: v0, out_w: v1, scale_h: p5, scale_w: p6,
        };
        self.f543(
            SHADER_UPSAMPLE_NEAREST_BACKWARD, Some("upsample_back"),
            &v4, &[p0], &v3,
            super::f540(v2),
        );
        Ok(v3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> &'static t500 { &crate::ops::TEST_DEV }

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
    fn f660_2x() {
        let v0 = dev().f502(&[1.0, 2.0, 3.0, 4.0]);
        let v1 = dev().f504(&dev().f660(&v0, 1, 1, 2, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(v1, vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ]);
    }

    #[test]
    fn f660_3x_vs_cpu() {
        let v0: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let v1 = cpu_upsample(&v0, 1, 1, 2, 3, 3, 3);
        let v2 = dev().f504(&dev().f660(&dev().f502(&v0), 1, 1, 2, 3, 3, 3).unwrap()).unwrap();
        assert_eq!(v2, v1);
    }

    #[test]
    fn f660_batched_multichannel_vs_cpu() {
        let v0: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let v1 = cpu_upsample(&v0, 2, 3, 2, 2, 2, 2);
        let v2 = dev().f504(&dev().f660(&dev().f502(&v0), 2, 3, 2, 2, 2, 2).unwrap()).unwrap();
        assert_eq!(v2, v1);
    }

    #[test]
    fn f660_1x1() {
        let v0 = dev().f504(&dev().f660(&dev().f502(&[7.0]), 1, 1, 1, 1, 3, 3).unwrap()).unwrap();
        assert_eq!(v0, vec![7.0; 9]);
    }
}
