// f16 in the half precision version of this shader
alias real = f32;

const SH_C0: real = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<real,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<real,7>(
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
    culling: f32,       // 1 = frustum culling on
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos: vec2<f32>,             // ndc center
    size: u32,                  // quad half size in ndc, 2 x f16
    opacity: f32,
    conic: array<u32,2>,        // 2d conic (a, b, c), f16
    color: array<u32,2>,        // rgb, f16
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coefs: array<u32>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// two f16 values from one u32
fn unpack_half(w: u32) -> vec2<real> {
    return unpack2x16float(w);
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<real> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    // each gaussian has 16 x rgb f16 values = 24 u32
    let i = splat_idx * 24u + c_idx * 3u / 2u;
    let a = unpack_half(sh_coefs[i]);
    let b = unpack_half(sh_coefs[i + 1u]);
    // odd coefs start in the high half
    if (c_idx % 2u == 0u) {
        return vec3<real>(a.x, a.y, b.x);
    }
    return vec3<real>(a.y, b.x, b.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<real>, v_idx: u32, sh_deg: u32) -> vec3<real> {
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

    return  max(vec3<real>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos = vec3<f32>(a.x, a.y, b.x);
    let view_pos = camera.view * vec4<f32>(pos, 1.0);
    let clip_pos = camera.proj * view_pos;
    let ndc = clip_pos.xyz / clip_pos.w;

    // 1.2x bigger box so splats at the edge are kept
    let outside = abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z > 1.0;
    // splats behind the near plane can't be projected, so they are always skipped
    if (clip_pos.w <= 0.0 || ndc.z < 0.0 || (render_settings.culling > 0.5 && outside)) {
        return;
    }

    // rotation matrix from the quaternion (w, x, y, z), given column by column
    let rot_a = unpack_half(gaussian.rot[0]);
    let rot_b = unpack_half(gaussian.rot[1]);
    let q = normalize(vec4<real>(rot_a, rot_b));
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let R = mat3x3<real>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)
    );
    // scale is stored in log space
    let scale_a = unpack_half(gaussian.scale[0]);
    let scale_b = unpack_half(gaussian.scale[1]);
    let s = exp(vec3<real>(scale_a, scale_b.x)) * real(render_settings.gaussian_scaling);
    // M = R * S, cov3d = M * M^T
    let M = mat3x3<real>(R[0] * s.x, R[1] * s.y, R[2] * s.z);
    // the projection needs f32, the 2d covariance can be thousands of pixels^2
    let cov3d = mat3x3<f32>(M * transpose(M));

    // jacobian of the projection, clamped a bit outside the screen like the paper
    let t = view_pos.xyz;
    let lim = 1.3 * 0.5 * camera.viewport / camera.focal;
    let tx = clamp(t.x / t.z, -lim.x, lim.x) * t.z;
    let ty = clamp(t.y / t.z, -lim.y, lim.y) * t.z;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let J = mat3x3<f32>(
        fx / t.z, 0.0, 0.0,
        0.0, fy / t.z, 0.0,
        -fx * tx / (t.z * t.z), -fy * ty / (t.z * t.z), 0.0
    );
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    let cov2d = T * cov3d * transpose(T);

    // + 0.3 so every splat is at least about one pixel wide
    let cov = vec3<f32>(cov2d[0][0] + 0.3, cov2d[0][1], cov2d[1][1] + 0.3);
    let det = cov.x * cov.z - cov.y * cov.y;
    if (det <= 0.0) {
        return;
    }
    let conic = vec3<f32>(cov.z, -cov.y, cov.x) / det;

    // radius in pixels = 3 sigma along the longer axis
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    let cam_pos = camera.view_inv[3].xyz;
    let color = vec3<f32>(computeColorFromSH(vec3<real>(normalize(pos - cam_pos)), idx, u32(render_settings.sh_deg)));

    splats[idx].pos = ndc.xy;
    splats[idx].size = pack2x16float(radius * 2.0 / camera.viewport);
    splats[idx].opacity = 1.0 / (1.0 + exp(-b.y));
    splats[idx].conic = array<u32,2>(pack2x16float(conic.xy), pack2x16float(vec2<f32>(conic.z, 0.0)));
    splats[idx].color = array<u32,2>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, 0.0)));

    let key = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[key] = idx;
    // the sort is ascending, so far splats get small keys
    sort_depths[key] = 0xffffffffu - bitcast<u32>(view_pos.z);

    let keys_per_dispatch = sortWorkgroupSize * sortKeyPerThread;
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (key % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}