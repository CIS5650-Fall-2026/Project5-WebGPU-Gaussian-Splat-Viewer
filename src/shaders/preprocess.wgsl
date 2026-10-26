// f32 math backend for the preprocess step (see preprocess_common.wgsl for the declarations).

const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

/// reads the ith sh coef from the storage buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    // each gaussian stores 16 coefs x RGB = 48 f16 = 24 u32, coefficients are interleaved RGB
    let base = splat_idx * 24u + (c_idx * 3u) / 2u;
    let a = unpack2x16float(sh_buffer[base]);
    let b = unpack2x16float(sh_buffer[base + 1u]);
    // an odd coefficient starts in the upper half of a u32
    if (c_idx & 1u) == 0u {
        return vec3<f32>(a.x, a.y, b.x);
    }
    return vec3<f32>(a.y, b.x, b.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
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

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

// 3D covariance from rotation (quaternion, w first) and scale, Sigma = R S S^T R^T
fn compute_cov3d(rot: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
    let q = normalize(rot);
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    // same construction as the reference CUDA rasterizer (column major, so this is R^T)
    let R = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    );
    let S = mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    let M = S * R;
    return transpose(M) * M;
}

// EWA splatting: project the 3D covariance into pixel space (returns a, b, c of the 2x2 matrix)
fn compute_cov2d(view_pos: vec3<f32>, cov3d: mat3x3<f32>) -> vec3<f32> {
    let tan_fovx = camera.viewport.x / (2.0 * camera.focal.x);
    let tan_fovy = camera.viewport.y / (2.0 * camera.focal.y);
    let limx = 1.3 * tan_fovx;
    let limy = 1.3 * tan_fovy;

    var t = view_pos;
    t.x = clamp(t.x / t.z, -limx, limx) * t.z;
    t.y = clamp(t.y / t.z, -limy, limy) * t.z;

    // Jacobian of the perspective projection (column major, i.e. J^T)
    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );
    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;

    var cov = transpose(T) * transpose(cov3d) * T;
    // low pass filter: every gaussian covers at least ~1 pixel
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}

fn project_gaussian(idx: u32) -> Projected {
    var p: Projected;
    p.visible = false;

    let g = gaussians[idx];
    let xy = unpack2x16float(g.pos_opacity[0]);
    let z_op = unpack2x16float(g.pos_opacity[1]);
    let pos = vec3<f32>(xy, z_op.x);

    let view_pos = (camera.view * vec4<f32>(pos, 1.0)).xyz;
    // near plane cull (also needed without frustum culling to avoid dividing by ~0)
    if view_pos.z < 0.2 {
        return p;
    }
    let clip = camera.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xy / clip.w;
    if render_settings.frustum_culling > 0.5 && (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) {
        return p;
    }

    let rot01 = unpack2x16float(g.rot[0]);
    let rot23 = unpack2x16float(g.rot[1]);
    let s01 = unpack2x16float(g.scale[0]);
    let s23 = unpack2x16float(g.scale[1]);
    let scale = exp(vec3<f32>(s01, s23.x)) * render_settings.gaussian_scaling;

    let cov3d = compute_cov3d(vec4<f32>(rot01, rot23), scale);
    let cov2d = compute_cov2d(view_pos, cov3d);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if det <= 0.0 {
        return p;
    }
    let det_inv = 1.0 / det;
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(lambda1));

    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos - cam_pos);

    p.visible = true;
    p.ndc = ndc;
    p.view_z = view_pos.z;
    p.radius = radius;
    p.conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    p.color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));
    p.opacity = sigmoid(z_op.y);
    return p;
}
