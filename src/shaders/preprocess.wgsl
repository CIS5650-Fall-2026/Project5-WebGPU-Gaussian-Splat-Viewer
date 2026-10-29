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

override workgroupSize: u32;
override sortKeyPerThread: u32;
override sortWorkgroupSize: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    center: vec2<f32>,
    half_size: vec2<f32>,
    conic: vec4<f32>,
    color: vec4<f32>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_coeffs: array<u32>;
@group(1) @binding(2)
var<storage, read_write> splats: array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let offset = splat_idx * 24u + c_idx * 3u / 2u;
    let a = unpack2x16float(sh_coeffs[offset]);
    let b = unpack2x16float(sh_coeffs[offset + 1u]);
    if c_idx % 2u == 0u {
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

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) || settings.gaussian_scaling <= 0.0 {
        return;
    }

    let g = gaussians[idx];
    let a = unpack2x16float(g.pos_opacity[0]);
    let b = unpack2x16float(g.pos_opacity[1]);
    let pos = vec3<f32>(a.x, a.y, b.x);

    let view_pos = camera.view * vec4<f32>(pos, 1.0);
    let clip = camera.proj * view_pos;
    if clip.w <= 0.0 {
        return;
    }
    let ndc = clip.xyz / clip.w;

    // Slightly wider than the screen, as suggested by the recitation.
    if abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z < 0.0 || ndc.z > 1.0 {
        return;
    }

    // Recover the rotation quaternion and scale from the PLY data.
    let q0 = unpack2x16float(g.rot[0]);
    let q1 = unpack2x16float(g.rot[1]);
    let q = normalize(vec4<f32>(q0.x, q0.y, q1.x, q1.y));
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let s0 = unpack2x16float(g.scale[0]);
    let s1 = unpack2x16float(g.scale[1]);
    let scale = exp(vec3<f32>(s0.x, s0.y, s1.x)) * settings.gaussian_scaling;

    let r0 = vec3<f32>(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z), 2.0*(x*z - w*y));
    let r1 = vec3<f32>(2.0*(x*y - w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x));
    let r2 = vec3<f32>(2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y));
    let rs = mat3x3<f32>(r0 * scale.x, r1 * scale.y, r2 * scale.z);
    let cov3 = rs * transpose(rs);

    // Projection Jacobian: converts camera-space displacement to screen pixels.
    let proj_x = vec3<f32>(camera.proj[0][0], camera.proj[1][0], camera.proj[2][0]);
    let proj_y = vec3<f32>(camera.proj[0][1], camera.proj[1][1], camera.proj[2][1]);
    let proj_w = vec3<f32>(camera.proj[0][3], camera.proj[1][3], camera.proj[2][3]);
    let jx = (proj_x - ndc.x * proj_w) * (camera.viewport.x * 0.5 / clip.w);
    let jy = (proj_y - ndc.y * proj_w) * (camera.viewport.y * 0.5 / clip.w);

    let view3 = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let dx = transpose(view3) * jx;
    let dy = transpose(view3) * jy;

    // Sigma_2d = J * W * Sigma_3d * transpose(J * W).
    let cov_x = dot(dx, cov3 * dx) + 0.3;
    let cov_y = dot(dy, cov3 * dy) + 0.3;
    let cov_xy = dot(dx, cov3 * dy);
    let det = cov_x * cov_y - cov_xy * cov_xy;
    if det <= 0.0 {
        return;
    }

    let mid = (cov_x + cov_y) * 0.5;
    let eig = mid + sqrt(max(0.0, mid*mid - det));
    let radius = ceil(3.0 * sqrt(max(eig, 0.0)));

    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos - cam_pos);
    let color = computeColorFromSH(dir, idx, u32(settings.sh_deg));
    let opacity = 1.0 / (1.0 + exp(-b.y));

    splats[idx].center = ndc.xy;
    splats[idx].half_size = 2.0 * vec2<f32>(radius) / camera.viewport;
    splats[idx].conic = vec4<f32>(cov_y / det, -cov_xy / det, cov_x / det, opacity);
    splats[idx].color = vec4<f32>(color, 1.0);

    // The sorter orders unsigned keys ascending; invert positive camera depth
    // so the farthest splats are drawn first (back to front).
    let slot = atomicAdd(&sort_infos.keys_size, 1u);
    sort_depths[slot] = ~bitcast<u32>(clip.w);
    sort_indices[slot] = idx;

    let keys_per_dispatch = sortWorkgroupSize * sortKeyPerThread;
    if slot % keys_per_dispatch == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}