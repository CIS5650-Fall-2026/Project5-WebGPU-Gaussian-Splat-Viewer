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
override sortKeysPerDispatch: u32;
override shDegree: u32;

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
    frustum_culling: u32,
    point_count: u32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    center: vec4<f32>,
    color: vec4<f32>,
    half_size: vec2<f32>,
    radius: f32,
    conic_opacity: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(0) @binding(1)
var<storage, read> gaussians: array<Gaussian>;

@group(0) @binding(2)
var<storage, read_write> splats: array<Splat>;

@group(0) @binding(3)
var<storage, read> sh_coeffs: array<u32>;

@group(0) @binding(4)
var<uniform> settings: RenderSettings;
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
    let offset = splat_idx * 48u + c_idx * 3u;

    let r = unpack2x16float(sh_coeffs[offset / 2u]);
    let g = unpack2x16float(sh_coeffs[(offset + 1u) / 2u]);
    let b = unpack2x16float(sh_coeffs[(offset + 2u) / 2u]);

    return vec3<f32>(
        r[offset % 2u],
        g[(offset + 1u) % 2u],
        b[(offset + 2u) % 2u]
    );
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

fn computeCov3D(gaussian: Gaussian) -> mat3x3<f32> {
    let s0 = unpack2x16float(gaussian.scale[0]);
    let s1 = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(s0, s1.x)) * settings.gaussian_scaling;

    let r0 = unpack2x16float(gaussian.rot[0]);
    let r1 = unpack2x16float(gaussian.rot[1]);
    let rotation = vec4<f32>(r0, r1);
    let q = rotation / max(length(rotation), 0.00000001);
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z),
                  2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3<f32>(2.0 * (x * y - w * z),
                  1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3<f32>(2.0 * (x * z + w * y), 2.0 * (y * z - w * x),
                  1.0 - 2.0 * (x * x + y * y))
    );
    let M = mat3x3<f32>(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    return M * transpose(M);
}

fn computeCov2D(view_pos: vec3<f32>, cov3d: mat3x3<f32>) -> vec3<f32> {
    let V = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let cov = V * cov3d * transpose(V);
    let limit = 1.3 * camera.viewport / (2.0 * camera.focal);
    let xy = clamp(view_pos.xy / view_pos.z, -limit, limit) * view_pos.z;
    let inv_z = 1.0 / view_pos.z;
    let jx = vec3<f32>(camera.focal.x * inv_z, 0.0,
                      -camera.focal.x * xy.x * inv_z * inv_z);
    let jy = vec3<f32>(0.0, camera.focal.y * inv_z,
                      -camera.focal.y * xy.y * inv_z * inv_z);

    return vec3<f32>(
        dot(jx, cov * jx) + 0.3,
        dot(jx, cov * jy),
        dot(jy, cov * jy) + 0.3
    );
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn preprocess(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) wgs: vec3<u32>
) {
    let count = min(settings.point_count, arrayLength(&gaussians));
    if (gid.x >= count || settings.gaussian_scaling <= 0.0) {
        return;
    }
    let idx = min(u32(floor(f32(gid.x) * (f32(arrayLength(&gaussians)) / f32(count)))),
                  arrayLength(&gaussians) - 1u);

    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.0);

    let view_pos = camera.view * pos;
    let clip = camera.proj * view_pos;

    if (clip.w <= 0.0) {
        return;
    }

    let ndc = clip.xyz / clip.w;

    if (settings.frustum_culling != 0u &&
        (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 ||
         ndc.z < 0.0 || ndc.z > 1.0)) {
        return;
    }

    let cov3d = computeCov3D(gaussian);
    let cov2d = computeCov2D(view_pos.xyz, cov3d);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 0.0) {
        return;
    }
    let conic = vec3<f32>(cov2d.z, -cov2d.y, cov2d.x) / det;
    let opacity = 1.0 / (1.0 + exp(-b.y));
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let diff = 0.5 * (cov2d.x - cov2d.z);
    let lambda_max = mid + sqrt(diff * diff + cov2d.y * cov2d.y);
    let radius = ceil(3.0 * sqrt(max(lambda_max, 0.0)));
    let half_size = vec2<f32>(2.0 * radius) / camera.viewport;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - camera_pos);
    let color = computeColorFromSH(dir, idx, shDegree);

    let visible_idx = atomicAdd(&sort_infos.keys_size, 1u);
    splats[visible_idx].center = vec4<f32>(ndc, 1.0);
    splats[visible_idx].color = vec4<f32>(color, 1.0);
    splats[visible_idx].half_size = half_size;
    splats[visible_idx].radius = radius;
    splats[visible_idx].conic_opacity = vec4<f32>(conic, opacity);

    sort_depths[visible_idx] = ~bitcast<u32>(clip.w);
    sort_indices[visible_idx] = visible_idx;

    if (visible_idx % sortKeysPerDispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}