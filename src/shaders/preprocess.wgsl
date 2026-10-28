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
override enableFrustumCulling: u32;

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
    pos: vec3<f32>,
    radius: f32,
    conic: mat2x2<f32>,
    color: vec4<f32>
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<uniform> settings: RenderSettings;
@group(1) @binding(3)
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
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let offset = splat_idx * 48u + c_idx * 3u;
    let a = unpack2x16float(sh_coeffs[offset / 2]);
    let b = unpack2x16float(sh_coeffs[offset / 2 + 1]);

    if (offset % 2 == 0) {
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

fn rotationMatrixHelper(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    return mat3x3<f32>(
        vec3f(
            1.0 - 2.0 * (y*y + z*z),
            2.0 * (x*y + z*w),
            2.0 * (x*z - y*w)
        ),
        vec3f(
            2.0 * (x*y - z*w),
            1.0 - 2.0 * (x*x + z*z),
            2.0 * (y*z + x*w)
        ),
        vec3f(
            2.0 * (x*z + y*w),
            2.0 * (y*z - x*w),
            1.0 - 2.0 * (x*x + y*y)
        )
    );
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // view frustum culling to remove non-visible Gaussians
    let p0 = unpack2x16float(gaussians[idx].pos_opacity[0]);
    let p1 = unpack2x16float(gaussians[idx].pos_opacity[1]);
    let pos = vec3f(p0.x, p0.y, p1.x);
    let view_pos = camera.view * vec4f(pos, 1.0);
    let clip = camera.proj * view_pos;
    if (clip.w <= 0) {
        return;
    }

    let ndc = clip.xyz / clip.w;

    if (enableFrustumCulling != 0 && (ndc.z < 0 || ndc.z > 1)) {
        return;
    }

    //Compute 3D covariance based on rotation and scale, also user inputted gaussian multipler
    let q0 = unpack2x16float(gaussians[idx].rot[0]);
    let q1 = unpack2x16float(gaussians[idx].rot[1]);
    let q_raw = vec4f(q0.x, q0.y, q1.x, q1.y);
    let q = q_raw * inverseSqrt(max(dot(q_raw, q_raw), 1e-12));

    let s0 = unpack2x16float(gaussians[idx].scale[0]);
    let s1 = unpack2x16float(gaussians[idx].scale[1]);
    let s = exp(vec3f(s0.x, s0.y, s1.x)) * settings.gaussian_scaling;

    let r = rotationMatrixHelper(q);
    let t = mat3x3f(
        r[0] * s.x,
        r[1] * s.y,
        r[2] * s.z
    );
    let cov_3d = t * transpose(t);

    // Compute 2D conic, maximum radius, and maximum quad size in NDC
    let view_r = mat3x3f(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );
    let cov_view = view_r * cov_3d * transpose(view_r);

    var j: mat3x2f;
    for (var k = 0u; k < 3u; k++) {
        let col = camera.proj[k];
            j[k] = vec2f(0.5 * camera.viewport.x * (col.x * clip.w - clip.x * col.w) / (clip.w * clip.w),
            -0.5 * camera.viewport.y * (col.y * clip.w - clip.y * col.w) / (clip.w * clip.w));
    }

    var cov_2d = j * cov_view * transpose(j);
    cov_2d[0][0] += 0.3;
    cov_2d[1][1] += 0.3;

    let a = cov_2d[0][0];
    let b = 0.5 * (cov_2d[0][1] + cov_2d[1][0]);
    let c = cov_2d[1][1];
    let det = a * c - b * b;
    if (det <= 1e-8) {
        return;
    }

    let conic = mat2x2f(vec2f(c, -b) / det, vec2f(-b, a) / det);
    let lambda = 0.5 * (a + c) + sqrt(max(0, 0.25 * (a - c) * (a - c) + b * b));
    let radius_px = ceil(3 * sqrt(max(lambda, 0)));
    let radius = max(2 * radius_px / camera.viewport.x, 2 * radius_px / camera.viewport.y);

    if (enableFrustumCulling != 0 && (abs(ndc.x) > 1.2 + radius || abs(ndc.y) > 1.2 + radius)) {
        return;
    }
    // Using spherical harmonics coeffiecients to evaluate the color of the gaussian from particular view direction
    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos - camera_pos);
    let color = computeColorFromSH(dir, idx, u32(settings.sh_deg));
    let opacity = 1 / (1 + exp(-p1.y));

    // Store essential 2D gaussian data for later rasteriation pipeline
    let output_idx = atomicAdd(&sort_infos.keys_size, 1u);
    splats[output_idx].pos = ndc;
    splats[output_idx].radius = radius;
    splats[output_idx].conic = conic;
    splats[output_idx].color = vec4f(color, opacity);

    // Add key_size, indices, and depth to sorter
    sort_indices[output_idx] = output_idx;
    sort_depths[output_idx] = bitcast<u32>(1 - ndc.z);

    let keys_per_dispatch = 256 * sortKeyPerThread; // radix sort set 256 fixed histogram workgroup size
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (output_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}