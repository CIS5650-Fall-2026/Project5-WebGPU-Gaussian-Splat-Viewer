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

override preprocessWorkgroupSize: u32;
override sortKeysPerWorkgroup: u32;

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
    frustum_culling: f32,
    gaussian_stride: f32
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    clip_position: vec4<f32>,
    cov_radius: vec4<f32>,
    conic_opacity: vec4<f32>,
    color: vec4<f32>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats : array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_data: array<u32>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

fn read_sh_value(half_index: u32) -> f32 {
    let word_index = half_index / 2u;
    let component = half_index % 2u;

    let pair = unpack2x16float(sh_data[word_index]);

    return pair[component];
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base = splat_idx * 48u + c_idx * 3u;

    return vec3<f32>(
        read_sh_value(base),
        read_sh_value(base + 1u),
        read_sh_value(base + 2u),
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

fn compute_cov3d(gaussian: Gaussian, scaling: f32) -> mat3x3<f32> {
    let scale01 = unpack2x16float(gaussian.scale[0]);
    let scale23 = unpack2x16float(gaussian.scale[1]);

    let s = exp(vec3<f32>(
        scale01.x,
        scale01.y,
        scale23.x,
    )) * scaling;

    let q01 = unpack2x16float(gaussian.rot[0]);
    let q23 = unpack2x16float(gaussian.rot[1]);
    let q = normalize(vec4<f32>(q01, q23));

    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let R = mat3x3<f32>(
        vec3<f32>(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        vec3<f32>(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        vec3<f32>(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    );

    let M = mat3x3<f32>(
        R[0] * s.x,
        R[1] * s.y,
        R[2] * s.z,
    );

    return M * transpose(M);
}

fn compute_cov2d(cov_world: mat3x3<f32>, center_view: vec3<f32>) -> vec3<f32> {
    let V = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz,
    );

    let cov_view = V * cov_world * transpose(V);

    let X = center_view.x;
    let Y = center_view.y;
    let Z = center_view.z;

    let fx = camera.focal.x;
    let fy = camera.focal.y;

    let J = mat3x2<f32>(
        vec2<f32>(fx / Z, 0.0),
        vec2<f32>(0.0, fy / Z),
        vec2<f32>(
            -fx * X / (Z * Z),
            -fy * Y / (Z * Z),
        ),
    );

    let cov = J * cov_view * transpose(J);

    return vec3<f32>(
        cov[0][0] + 0.3,
        cov[1][0],
        cov[1][1] + 0.3,
    );
}

@compute @workgroup_size(preprocessWorkgroupSize, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let stride = max(1u, u32(render_settings.gaussian_stride));
    let idx = gid.x * stride;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let world_position = vec4<f32>(a.x, a.y, b.x, 1.0);
    
    let view_position = camera.view * world_position;
    let clip_position = camera.proj * view_position;
    if (clip_position.w <= 0.0) {
        return;
    }

    let ndc = clip_position.xyz / clip_position.w;

    if (render_settings.frustum_culling > 0.5) {
        let cull_limit = 1.2;

        if (abs(ndc.x) > cull_limit ||
            abs(ndc.y) > cull_limit ||
            ndc.z < 0.0 ||
            ndc.z > 1.0)
        {
            return;
        }
    }

    let scaling = render_settings.gaussian_scaling;
    if (scaling <= 0.0) {
        return;
    }

    let cov_world = compute_cov3d(gaussian, scaling);
    let cov2d = compute_cov2d(cov_world, view_position.xyz);

    let cov_a = cov2d.x;
    let cov_b = cov2d.y;
    let cov_c = cov2d.z;

    let det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 0.0) {
        return;
    }

    let conic = vec3<f32>(cov_c, -cov_b, cov_a) / det;

    let raw_opacity = b.y;
    let opacity = 1.0 / (1.0 + exp(-raw_opacity));

    let mid = 0.5 * (cov_a + cov_c);
    let half_diff = 0.5 * (cov_a - cov_c);

    let lambda_max = mid + sqrt(half_diff * half_diff + cov_b * cov_b);

    let radius_pixels = ceil(3.0 * sqrt(max(lambda_max, 0.0)));

    let camera_position = camera.view_inv[3].xyz;

    let view_dir = normalize(world_position.xyz - camera_position);

    let sh_degree = u32(clamp(render_settings.sh_deg, 0.0, 3.0));

    let color = computeColorFromSH(view_dir, idx, sh_degree);

    let visible_idx = atomicAdd(&sort_infos.keys_size, 1u);

    splats[visible_idx].clip_position = clip_position;
    splats[visible_idx].cov_radius = vec4<f32>(cov2d, radius_pixels);
    splats[visible_idx].conic_opacity = vec4<f32>(conic, opacity);
    splats[visible_idx].color = vec4<f32>(color, 1.0);

    let depth_key = ~bitcast<u32>(view_position.z);

    sort_depths[visible_idx] = depth_key;
    sort_indices[visible_idx] = visible_idx;

    if (visible_idx % sortKeysPerWorkgroup == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}