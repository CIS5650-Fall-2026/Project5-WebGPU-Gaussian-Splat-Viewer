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
    pos_size: array<u32,2>,
    color: array<u32,2>,
    conic_opacity: array<u32,2>,
};

//TODO: bind your data here
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;
@group(1) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coefficients: array<u32>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3f {
    // Each Gaussian stores 16 RGB coefficients as packed f16 values.
    let base = splat_idx * 48u + c_idx * 3u;
    var rgb: vec3f;

    for (var channel = 0u; channel < 3u; channel++) {
        let offset = base + channel;
        let pair = unpack2x16float(sh_coefficients[offset / 2u]);
        rgb[channel] = pair[offset % 2u];
    }

    return rgb;
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
    //TODO: set up pipeline as described in instruction

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    if (idx >= arrayLength(&gaussians)) { return; }
let g = gaussians[idx];

let scaleXY = unpack2x16float(g.scale[0]);
let scaleZ = unpack2x16float(g.scale[1]).x;

let scale3D = exp(vec3f(scaleXY, scaleZ))
    * render_settings.gaussian_scaling;

let rotationWX = unpack2x16float(g.rot[0]);
let rotationYZ = unpack2x16float(g.rot[1]);

let quaternion = normalize(vec4f(rotationWX, rotationYZ));

// Quaternion components in (w, x, y, z) order
let w = quaternion.x;
let x = quaternion.y;
let y = quaternion.z;
let z = quaternion.w;

// Convert the quaternion to a rotation matrix.
// WGSL matrix constructors take column vectors.
let R = mat3x3f(
    vec3f(
        1.0 - 2.0 * (y*y + z*z),
        2.0 * (x*y + w*z),
        2.0 * (x*z - w*y)
    ),
    vec3f(
        2.0 * (x*y - w*z),
        1.0 - 2.0 * (x*x + z*z),
        2.0 * (y*z + w*x)
    ),
    vec3f(
        2.0 * (x*z + w*y),
        2.0 * (y*z - w*x),
        1.0 - 2.0 * (x*x + y*y)
    )
);

// Scale each column along its corresponding local axis.
let RS = mat3x3f(
    R[0] * scale3D.x,
    R[1] * scale3D.y,
    R[2] * scale3D.z
);

// Compute the 3D covariance: Sigma = (R * S) * transpose(R * S).
let cov3D = RS * transpose(RS);

let a = unpack2x16float(g.pos_opacity[0]);
let b = unpack2x16float(g.pos_opacity[1]);
let view_pos = camera.view * vec4f(a.x, a.y, b.x, 1.0);
let clip = camera.proj * view_pos;
let ndc = clip.xyz / clip.w;
// Reject near-camera points and off-screen centers, with a margin for edge splats.
if (view_pos.z < 0.2 || abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) { return; }

// Transform the covariance from world space to camera space.
let viewRotation = mat3x3f(
    camera.view[0].xyz,
    camera.view[1].xyz,
    camera.view[2].xyz
);

let viewCov = viewRotation * cov3D * transpose(viewRotation);

// Compute the perspective projection Jacobian in pixel space.
let invZ = 1.0 / view_pos.z;

let Jx = vec3f(
    camera.focal.x * invZ,
    0.0,
    -camera.focal.x * view_pos.x * invZ * invZ
);

let Jy = vec3f(
    0.0,
    camera.focal.y * invZ,
    -camera.focal.y * view_pos.y * invZ * invZ
);

// Project to 2D and add a subpixel low-pass filter.
let covXX = dot(Jx, viewCov * Jx) + 0.3;
let covXY = dot(Jx, viewCov * Jy);
let covYY = dot(Jy, viewCov * Jy) + 0.3;

// Find the largest eigenvalue of the 2D covariance.
let midpoint = 0.5 * (covXX + covYY);
let discriminant = max(
    0.0,
    0.25 * (covXX - covYY) * (covXX - covYY)
        + covXY * covXY
);
let lambdaMax = midpoint + sqrt(discriminant);

// Convert the three-sigma radius from pixels to NDC half-extents.
let radius = ceil(3.0 * sqrt(lambdaMax));
let size = vec2f(2.0 * radius) / camera.viewport;


splats[idx].pos_size[0] = pack2x16float(ndc.xy);
splats[idx].pos_size[1] = pack2x16float(size);

// Evaluate view-dependent color using spherical harmonics.
let worldPosition = vec3f(a.x, a.y, b.x);
let direction = normalize(worldPosition - camera.view_inv[3].xyz);

let color = computeColorFromSH(
    direction,
    idx,
    u32(render_settings.sh_deg)
);

// Store RGB as packed f16 values.
splats[idx].color[0] = pack2x16float(color.rg);
splats[idx].color[1] = pack2x16float(vec2f(color.b, radius));

let det = covXX * covYY - covXY * covXY;
if (!(det > 0.0)) {
    return;
}

let conic = vec3f(covYY, -covXY, covXX) / det;

let opacity = 1.0 / (1.0 + exp(-b.y));

splats[idx].conic_opacity[0] = pack2x16float(conic.xy);
splats[idx].conic_opacity[1] = pack2x16float(
    vec2f(conic.z, opacity)
);

let slot = atomicAdd(&sort_infos.keys_size, 1u);
sort_depths[slot] = bitcast<u32>(100.0 - view_pos.z);
sort_indices[slot] = idx;
if (slot % keys_per_dispatch == 0u) {
    atomicAdd(&sort_dispatch.dispatch_x, 1u);
}
}
