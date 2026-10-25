alias SplatFloat = f32;
// Rotation and SH arithmetic can use f16; projection keeps its f32 range.
alias ShapeFloat = f32;

const SH_C0: ShapeFloat = 0.28209479177387814;
const SH_C1: ShapeFloat = 0.4886025119029199;
const SH_C2 = array<ShapeFloat,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<ShapeFloat,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortBlockSize: u32;

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
    culling: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    center: vec4<f32>,
    half_size: vec2<f32>,
    radius: f32,
    conic: vec4<SplatFloat>,
    color: vec4<SplatFloat>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coeffs: array<u32>;
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<ShapeFloat> {
    // Each point has 16 RGB coefficients, packed as consecutive half floats.
    let offset = splat_idx * 48u + c_idx * 3u;
    let r = unpack2x16float(sh_coeffs[offset / 2u]);
    let g = unpack2x16float(sh_coeffs[(offset + 1u) / 2u]);
    let b = unpack2x16float(sh_coeffs[(offset + 2u) / 2u]);
    return vec3<ShapeFloat>(vec3<f32>(r[offset % 2u], g[(offset + 1u) % 2u], b[(offset + 2u) % 2u]));
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<ShapeFloat>, v_idx: u32, sh_deg: u32) -> vec3<ShapeFloat> {
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

    return  max(vec3<ShapeFloat>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    let view_pos = camera.view * pos;
    let clip = camera.proj * view_pos;
    // Reject points behind the camera before dividing by w.
    if (clip.w <= 0.) {
        return;
    }
    let ndc = clip.xyz / clip.w;
    // Keep a margin around the screen so quads near the edges survive.
    if (settings.culling > 0. && (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z < 0. || ndc.z > 1.)) {
        return;
    }

    if (settings.gaussian_scaling <= 0.) {
        return;
    }
    let rot_a = unpack2x16float(gaussian.rot[0]);
    let rot_b = unpack2x16float(gaussian.rot[1]);
    let q = vec4<f32>(rot_a, rot_b);
    let rotation = vec4<ShapeFloat>(q * inverseSqrt(max(dot(q, q), 1e-8)));
    let w = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    let r = mat3x3<ShapeFloat>(
        vec3<ShapeFloat>(1. - 2. * (y*y + z*z), 2. * (x*y + w*z), 2. * (x*z - w*y)),
        vec3<ShapeFloat>(2. * (x*y - w*z), 1. - 2. * (x*x + z*z), 2. * (y*z + w*x)),
        vec3<ShapeFloat>(2. * (x*z + w*y), 2. * (y*z - w*x), 1. - 2. * (x*x + y*y))
    );
    let scale_a = unpack2x16float(gaussian.scale[0]);
    let scale_b = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(scale_a, scale_b.x)) * settings.gaussian_scaling;
    let rs = mat3x3<f32>(vec3<f32>(r[0]) * scale.x, vec3<f32>(r[1]) * scale.y, vec3<f32>(r[2]) * scale.z);
    let covariance = rs * transpose(rs);
    let view_rotation = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let view_covariance = view_rotation * covariance * transpose(view_rotation);

    // Project the covariance with the perspective Jacobian, in pixels.
    let focal = vec2<f32>(camera.proj[0][0], camera.proj[1][1]) * camera.viewport * 0.5;
    let inv_z = 1. / view_pos.z;
    let jx = vec3<f32>(focal.x * inv_z, 0., -focal.x * view_pos.x * inv_z * inv_z);
    let jy = vec3<f32>(0., focal.y * inv_z, -focal.y * view_pos.y * inv_z * inv_z);
    // A small pixel footprint keeps distant splats from disappearing.
    let cov_xx = dot(jx, view_covariance * jx) + 0.3;
    let cov_xy = dot(jx, view_covariance * jy);
    let cov_yy = dot(jy, view_covariance * jy) + 0.3;
    let determinant = cov_xx * cov_yy - cov_xy * cov_xy;
    if (determinant <= 0.) {
        return;
    }
    let mid = 0.5 * (cov_xx + cov_yy);
    let largest = mid + sqrt(max(0., 0.25 * (cov_xx - cov_yy) * (cov_xx - cov_yy) + cov_xy * cov_xy));
    let radius = ceil(3. * sqrt(largest));
    let direction = normalize(pos.xyz - camera.view_inv[3].xyz);
    let color = vec3<f32>(computeColorFromSH(vec3<ShapeFloat>(direction), idx, u32(settings.sh_deg)));
    let opacity = 1. / (1. + exp(-b.y));

    splats[idx].center = vec4<f32>(ndc, 1.);
    splats[idx].half_size = 2. * vec2<f32>(radius) / camera.viewport;
    splats[idx].radius = radius;
    splats[idx].conic = vec4<SplatFloat>(vec4<f32>(cov_yy, -cov_xy, cov_xx, 0.) / determinant);
    splats[idx].color = vec4<SplatFloat>(vec4<f32>(color, opacity));
    let visible_idx = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[visible_idx] = idx;
    // The radix sorter is ascending. Inverting positive depth bits puts far points first.
    sort_depths[visible_idx] = ~bitcast<u32>(view_pos.z);
    if (visible_idx % sortBlockSize == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
