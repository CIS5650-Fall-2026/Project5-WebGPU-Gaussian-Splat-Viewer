// The renderer prepends `alias Scalar = f16;` (with `enable f16;`) or
// `alias Scalar = f32;` to compile both precision modes from this shader.
const SH_C0: Scalar = 0.28209479177387814;
const SH_C1: Scalar = 0.4886025119029199;
const SH_C2 = array<Scalar,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<Scalar,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortWorkgroupSize: u32;
override sortKeyPerThread: u32;
override frustumCulling: bool = true;

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
    // Shared by both precision modes; must match gaussian.wgsl (32 bytes).
    // Keep centers and conics in f32; sizes, RGB and opacity are packed f16.
    center: vec2<f32>,
    size: u32,
    color_rg: u32,
    color_b_opacity: u32,
    conic: array<f32,3>,
};

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
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<Scalar> {
    // load.ts stores 16 RGB coefficients per Gaussian, packed two f16s per u32.
    // RGB triples alternate between even and odd half-word offsets.
    let offset = splat_idx * 48u + c_idx * 3u;
    let a = vec2<Scalar>(unpack2x16float(sh_coeffs[offset / 2u]));
    let b = vec2<Scalar>(unpack2x16float(sh_coeffs[offset / 2u + 1u]));
    if (offset % 2u == 0u) {
        return vec3<Scalar>(a.x, a.y, b.x);
    }
    return vec3<Scalar>(a.y, b.x, b.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<Scalar>, v_idx: u32, sh_deg: u32) -> vec3<Scalar> {
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

    return max(vec3<Scalar>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians) || settings.gaussian_scaling <= 0.0) {
        return;
    }

    let gaussian = gaussians[idx];
    let xy = unpack2x16float(gaussian.pos_opacity[0]);
    let zo = unpack2x16float(gaussian.pos_opacity[1]);
    let position = vec3<f32>(xy, zo.x);
    let view_position = camera.view * vec4<f32>(position, 1.0);
    let clip = camera.proj * view_position;

    // Keep a 20% guard band for splats whose centers are just off screen.
    // This camera looks along +Z; WebGPU's clip depth is [0, w].
    if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w) {
        return;
    }
    if (frustumCulling && any(abs(clip.xy) > vec2<f32>(1.2 * clip.w))) {
        return;
    }

    let opacity = 1.0 / (1.0 + exp(-zo.y));
    if (opacity < 1.0 / 255.0) {
        return;
    }

    // PLY rotations are quaternions in (w, x, y, z) order.
    let raw_rotation = vec4<f32>(unpack2x16float(gaussian.rot[0]),
                                unpack2x16float(gaussian.rot[1]));
    // Normalize in f32 before converting: a squared small quaternion can
    // underflow in f16 even when the input components are representable.
    let q = vec4<Scalar>(raw_rotation * inverseSqrt(max(dot(raw_rotation, raw_rotation), 1e-8)));
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let rotation = mat3x3<Scalar>(
        vec3<Scalar>(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y + w*z), 2.0 * (x*z - w*y)),
        vec3<Scalar>(2.0 * (x*y - w*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z + w*x)),
        vec3<Scalar>(2.0 * (x*z + w*y), 2.0 * (y*z - w*x), 1.0 - 2.0 * (x*x + y*y))
    );
    let scale_xy = unpack2x16float(gaussian.scale[0]);
    let scale_z = unpack2x16float(gaussian.scale[1]).x;
    let log_scale = vec3<f32>(scale_xy, scale_z);
    let max_log_scale = max(log_scale.x, max(log_scale.y, log_scale.z));
    // Relative scales stay in [0, 1], avoiding f16 overflow or loss of all
    // three axes when a Gaussian is very large or very small.
    let relative_log_scale = log_scale - vec3<f32>(max_log_scale);
    let scale = exp(vec3<Scalar>(relative_log_scale));
    let scaled_rotation = mat3x3<Scalar>(rotation[0] * scale.x,
                                        rotation[1] * scale.y,
                                        rotation[2] * scale.z);
    // Form covariance in f32 from its half-precision factor. Squaring tiny
    // scales in f16 underflows; rounded covariance entries can also lose
    // positive definiteness for elongated Gaussians.
    var covariance_factor = mat3x3<f32>(scaled_rotation) * (exp(max_log_scale) * settings.gaussian_scaling);
    // f16 subnormals may be flushed to zero on some GPUs. Recover very thin
    // axes in f32 when their relative scale is below the f16 normal range.
    for (var axis = 0u; axis < 3u; axis++) {
        if (relative_log_scale[axis] < -9.70406053) {
            covariance_factor[axis] = vec3<f32>(rotation[axis]) * (exp(log_scale[axis]) * settings.gaussian_scaling);
        }
    }
    let covariance_3d = covariance_factor * transpose(covariance_factor);

    let view_rotation = mat3x3<f32>(camera.view[0].xyz,
                                   camera.view[1].xyz,
                                   camera.view[2].xyz);
    let covariance_view = view_rotation * covariance_3d * transpose(view_rotation);

    // Project covariance with the perspective Jacobian, in framebuffer pixels
    // (X right, Y down). Clamp the Jacobian near the edges of the field of view.
    let inv_z = 1.0 / view_position.z;
    let limit = 1.3 * camera.viewport / (2.0 * camera.focal);
    let slope = clamp(view_position.xy * inv_z, -limit, limit);
    let jx = vec3<f32>(camera.focal.x * inv_z, 0.0, -camera.focal.x * slope.x * inv_z);
    let jy = vec3<f32>(0.0, camera.focal.y * inv_z, -camera.focal.y * slope.y * inv_z);
    // A small low-pass filter keeps subpixel splats visible and invertible.
    let a = dot(jx, covariance_view * jx) + 0.3;
    let b = dot(jx, covariance_view * jy);
    let c = dot(jy, covariance_view * jy) + 0.3;
    let determinant = a * c - b * b;
    if (!(determinant > 0.0)) {
        return;
    }
    let conic = vec3<f32>(c, -b, a) / determinant;
    let midpoint = 0.5 * (a + c);
    let max_eigenvalue = midpoint + sqrt(max(0.1, midpoint * midpoint - determinant));
    let radius = ceil(3.0 * sqrt(max_eigenvalue));
    // radius is a pixel half-size; multiply by two to convert it to NDC.
    let half_size_ndc = 2.0 * radius / camera.viewport;

    let direction = normalize(position - camera.view_inv[3].xyz);
    let color = vec3<f32>(computeColorFromSH(vec3<Scalar>(direction), idx, u32(clamp(settings.sh_deg, 0.0, 3.0))));
    splats[idx].center = clip.xy / clip.w;
    // Clamp only values being compressed, so their f16 representation stays finite.
    splats[idx].size = pack2x16float(min(half_size_ndc, vec2<f32>(65504.0)));
    splats[idx].color_rg = pack2x16float(min(color.rg, vec2<f32>(65504.0)));
    splats[idx].color_b_opacity = pack2x16float(vec2<f32>(min(color.b, 65504.0), opacity));
    splats[idx].conic = array<f32,3>(conic.x, conic.y, conic.z);

    let visible_idx = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[visible_idx] = idx;
    // The radix sorter orders unsigned keys ascending. Complementing positive
    // camera-space depth bits produces the back-to-front order alpha blending needs.
    sort_depths[visible_idx] = ~bitcast<u32>(view_position.z);

    // Sorting uses its own fixed workgroup size, independently of preprocessing.
    let keys_per_dispatch = sortWorkgroupSize * sortKeyPerThread;
    if (visible_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
