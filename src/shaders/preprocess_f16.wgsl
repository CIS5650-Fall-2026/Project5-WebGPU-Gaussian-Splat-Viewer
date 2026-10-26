// f16 math backend for the preprocess step (extra credit), selected on the host when the device supports
// `shader-f16`. The module is prefixed with `enable f16;`.
//
// What runs in f16:
//   - decoding of the packed gaussian / SH data (bitcast to vec2<f16>, no unpack to f32)
//   - spherical harmonics evaluation (48 coefficients per gaussian, the bulk of the ALU work)
//   - quaternion -> rotation matrix and the 3D covariance, computed with scales normalized by the largest
//     scale so every entry is in [0, 1] (exp(scale)^2 of small gaussians underflows f16 otherwise)
//   - opacity sigmoid
// What stays in f32:
//   - view / projection transforms (world positions and depths need the mantissa)
//   - the EWA projection to pixel space: covariance entries reach focal^2 * sigma^2 / z^2 ~ 1e6, far above
//     the f16 maximum of 65504, and the conic of large splats would be an f16 denormal

const SH_C0: f16 = 0.28209479177387814h;
const SH_C1: f16 = 0.4886025119029199h;
const SH_C2 = array<f16,5>(
    1.0925484305920792h,
    -1.0925484305920792h,
    0.31539156525252005h,
    -1.0925484305920792h,
    0.5462742152960396h
);
const SH_C3 = array<f16,7>(
    -0.5900435899266435h,
    2.890611442640554h,
    -0.4570457994644658h,
    0.3731763325901154h,
    -0.4570457994644658h,
    1.445305721320277h,
    -0.5900435899266435h
);

fn half2(packed: u32) -> vec2<f16> {
    return bitcast<vec2<f16>>(packed);
}

/// reads the ith sh coef from the storage buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f16> {
    // each gaussian stores 16 coefs x RGB = 48 f16 = 24 u32, coefficients are interleaved RGB
    let base = splat_idx * 24u + (c_idx * 3u) / 2u;
    let a = half2(sh_buffer[base]);
    let b = half2(sh_buffer[base + 1u]);
    if (c_idx & 1u) == 0u {
        return vec3<f16>(a.x, a.y, b.x);
    }
    return vec3<f16>(a.y, b.x, b.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f16>, v_idx: u32, sh_deg: u32) -> vec3<f16> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0h * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0h * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0h * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0h * zz - 3.0h * xx - 3.0h * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0h * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0h * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5h;

    return  max(vec3<f16>(0.0h), result);
}

// 3D covariance divided by the largest squared scale: R (S / s_max)^2 R^T, all entries in [0, 1]
fn compute_cov3d_normalized(rot: vec4<f16>, scale_rel: vec3<f16>) -> mat3x3<f16> {
    let q = normalize(rot);
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    // same construction as the reference CUDA rasterizer (column major, so this is R^T)
    let R = mat3x3<f16>(
        1.0h - 2.0h * (y * y + z * z), 2.0h * (x * y - r * z), 2.0h * (x * z + r * y),
        2.0h * (x * y + r * z), 1.0h - 2.0h * (x * x + z * z), 2.0h * (y * z - r * x),
        2.0h * (x * z - r * y), 2.0h * (y * z + r * x), 1.0h - 2.0h * (x * x + y * y)
    );
    let S = mat3x3<f16>(
        scale_rel.x, 0.0h, 0.0h,
        0.0h, scale_rel.y, 0.0h,
        0.0h, 0.0h, scale_rel.z
    );
    let M = S * R;
    return transpose(M) * M;
}

// EWA splatting in f32 (see header), returns a, b, c of the 2x2 pixel space covariance
fn compute_cov2d(view_pos: vec3<f32>, cov3d: mat3x3<f32>) -> vec3<f32> {
    let tan_fovx = camera.viewport.x / (2.0 * camera.focal.x);
    let tan_fovy = camera.viewport.y / (2.0 * camera.focal.y);
    let limx = 1.3 * tan_fovx;
    let limy = 1.3 * tan_fovy;

    var t = view_pos;
    t.x = clamp(t.x / t.z, -limx, limx) * t.z;
    t.y = clamp(t.y / t.z, -limy, limy) * t.z;

    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;

    var cov = transpose(T) * transpose(cov3d) * T;
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn project_gaussian(idx: u32) -> Projected {
    var p: Projected;
    p.visible = false;

    let g = gaussians[idx];
    let xy = half2(g.pos_opacity[0]);
    let z_op = half2(g.pos_opacity[1]);
    let pos = vec3<f32>(vec2<f32>(xy), f32(z_op.x));

    let view_pos = (camera.view * vec4<f32>(pos, 1.0)).xyz;
    if view_pos.z < 0.2 {
        return p;
    }
    let clip = camera.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xy / clip.w;
    if render_settings.frustum_culling > 0.5 && (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        return p;
    }

    let rot = vec4<f16>(half2(g.rot[0]), half2(g.rot[1]));
    let s01 = half2(g.scale[0]);
    let s23 = half2(g.scale[1]);
    // log space scales: dividing by the largest scale is a subtraction before the exp
    let log_scale = vec3<f16>(s01, s23.x);
    let log_max = max(max(log_scale.x, log_scale.y), log_scale.z);
    let scale_rel = exp(log_scale - log_max);
    let s_max = exp(f32(log_max)) * render_settings.gaussian_scaling;

    let cov3d = mat3x3<f32>(compute_cov3d_normalized(rot, scale_rel)) * (s_max * s_max);
    let cov2d = compute_cov2d(view_pos, cov3d);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if det <= 0.0 {
        return p;
    }
    let det_inv = 1.0 / det;
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let opacity = f32(1.0h / (1.0h + exp(-z_op.y)));
    let extent = splat_extent(cov2d, lambda1, opacity);
    if extent.x <= 0.0 {
        return p;
    }

    let cam_pos = camera.view_inv[3].xyz;
    let dir = vec3<f16>(normalize(pos - cam_pos));

    p.visible = true;
    p.ndc = ndc;
    p.view_z = view_pos.z;
    p.extent = extent;
    p.conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    p.color = vec3<f32>(computeColorFromSH(dir, idx, u32(render_settings.sh_deg)));
    p.opacity = opacity;
    return p;
}
