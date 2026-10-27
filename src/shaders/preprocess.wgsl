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
    //TODO: store information for 2D splat rendering
    center: u32,
    size: u32,
    conic_xy: u32,
    conic_z_opacity: u32,
    color_rg: u32,
    color_b: u32,
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
var<storage, read> sh_buffer : array<u32>;

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
    let first = splat_idx * 48u + c_idx * 3u;
    return vec3<f32>(sh_half(first), sh_half(first + 1u), sh_half(first + 2u));
}

fn sh_half(i: u32) -> f32 {
    let pair = unpack2x16float(sh_buffer[i / 2u]);
    if (i % 2u == 0u) {
        return pair.x;
    }
    return pair.y;
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
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);

    let view_pos = camera.view * pos;
    let clip = camera.proj * view_pos;
    let ndc = clip.xyz / clip.w;

    if (clip.w <= 0 || abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z < 0 || ndc.z > 1) {
        return;
    }

    // 3D covariance
    let rot_wx = unpack2x16float(vertex.rot[0]);
    let rot_yz = unpack2x16float(vertex.rot[1]);
    let q = normalize(vec4<f32>(rot_wx, rot_yz));
    let qw = q.x;
    let qx = q.y;
    let qy = q.z;
    let qz = q.w;

    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)),
        vec3<f32>(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx)),
        vec3<f32>(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)),
    );
    let scale_xy = unpack2x16float(vertex.scale[0]);
    let scale_z = unpack2x16float(vertex.scale[1]);
    let s = exp(vec3<f32>(scale_xy, scale_z.x)) * render_settings.gaussian_scaling;
    let S = mat3x3<f32>(
        vec3<f32>(s.x, 0.0, 0.0),
        vec3<f32>(0.0, s.y, 0.0),
        vec3<f32>(0.0, 0.0, s.z),
    );
    let M = R * S;
    let sigma = M * transpose(M);

    // 2D covariance
    let t = view_pos.xyz;
    let lim = 1.3 * 0.5 * camera.viewport / camera.focal;
    let tx = clamp(t.x / t.z, -lim.x, lim.x) * t.z;
    let ty = clamp(t.y / t.z, -lim.y, lim.y) * t.z;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let J = mat3x3<f32>(
        vec3<f32>(fx / t.z, 0.0, 0.0),
        vec3<f32>(0.0, fy / t.z, 0.0),
        vec3<f32>(-fx * tx / (t.z * t.z), -fy * ty / (t.z * t.z), 0.0),
    );
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    let cov = T * sigma * transpose(T);

    let cov_xx = cov[0][0] + 0.3;
    let cov_xy = cov[0][1];
    let cov_yy = cov[1][1] + 0.3;

    // conic
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0) {
        return;
    }
    let conic = vec3<f32>(cov_yy, -cov_xy, cov_xx) / det;

    // radius
    let mid = 0.5 * (cov_xx + cov_yy);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(lambda1));
    let size = vec2<f32>(radius) * 2.0 / camera.viewport;

    // opacity
    let opacity = 1.0 / (1.0 + exp(-b.y));
    let cam_pos = camera.view_inv[3].xyz;
    let color = computeColorFromSH(normalize(pos.xyz - cam_pos), idx, u32(render_settings.sh_deg));

    splats[idx].center = pack2x16float(ndc.xy);
    splats[idx].size = pack2x16float(size);
    splats[idx].conic_xy = pack2x16float(conic.xy);
    splats[idx].conic_z_opacity = pack2x16float(vec2<f32>(conic.z, opacity));
    splats[idx].color_rg = pack2x16float(color.rg);
    splats[idx].color_b = pack2x16float(vec2<f32>(color.b, 0.0));

    let slot = atomicAdd(&sort_infos.keys_size, 1u);
    sort_depths[slot] = ~bitcast<u32>(view_pos.z);
    sort_indices[slot] = idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (slot % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

}