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
// workgroup size of the radix sort; one sort dispatch handles sortWorkgroupSize * sortKeyPerThread keys
override sortWorkgroupSize: u32 = 256u;
// screen space culling toggle (only turned off for the performance analysis)
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

// 2D splat, 32 bytes. Center and conic stay f32 for sub-pixel accuracy, the rest is packed as f16.
struct Splat {
    center: vec2<f32>,      // NDC
    size: u32,              // f16x2: quad half extent in NDC
    conic: array<f32, 3>,   // inverse 2D covariance (xx, xy, yy) in pixels
    color: array<u32, 2>,   // f16x4: rgb, opacity
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_buffer: array<u32>;
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
    // per gaussian: 16 coefs x rgb as f16 = 24 u32, coef c starts at half index 3c
    let base = splat_idx * 24u + (c_idx / 2u) * 3u;
    if (c_idx % 2u) == 0u {
        let rg = unpack2x16float(sh_buffer[base]);
        let b = unpack2x16float(sh_buffer[base + 1u]).x;
        return vec3<f32>(rg, b);
    }
    let r = unpack2x16float(sh_buffer[base + 1u]).y;
    let gb = unpack2x16float(sh_buffer[base + 2u]);
    return vec3<f32>(r, gb);
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

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// world space covariance: Sigma = R S S^T R^T
fn compute_cov3d(rot: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
    let q = normalize(rot);  // (w, x, y, z)
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let R = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z), 2.0 * (x * z - r * y),
        2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + r * x),
        2.0 * (x * z + r * y), 2.0 * (y * z - r * x), 1.0 - 2.0 * (x * x + y * y)
    );
    let M = R * mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    return M * transpose(M);
}

// screen space covariance (pixels): Sigma' = J W Sigma W^T J^T
fn compute_cov2d(t_in: vec3<f32>, cov3d: mat3x3<f32>) -> vec3<f32> {
    // clamp to a slightly larger fov so the Jacobian stays stable at the screen border
    let tan_fov = 0.5 * camera.viewport / camera.focal;
    let lim = 1.3 * tan_fov;
    let tz = t_in.z;
    let txy = clamp(t_in.xy / tz, -lim, lim) * tz;

    let fx = camera.focal.x;
    let fy = camera.focal.y;
    // columns of the projection Jacobian (third row is 0)
    let J = mat3x3<f32>(
        fx / tz, 0.0, 0.0,
        0.0, fy / tz, 0.0,
        -fx * txy.x / (tz * tz), -fy * txy.y / (tz * tz), 0.0
    );
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    let cov = T * cov3d * transpose(T);

    // low-pass filter: every splat covers at least ~1 pixel
    return vec3<f32>(cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x + gid.y * wgs.x * workgroupSize;
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let g = gaussians[idx];
    let xy = unpack2x16float(g.pos_opacity[0]);
    let z_opacity = unpack2x16float(g.pos_opacity[1]);
    let pos = vec4<f32>(xy, z_opacity.x, 1.0);

    // view frustum culling: near plane + 1.2x NDC box to keep splats on the border.
    // The near test is always on, the projection is invalid behind the camera.
    let pos_view = camera.view * pos;
    if pos_view.z <= 0.2 {
        return;
    }
    let pos_clip = camera.proj * pos_view;
    let ndc = pos_clip.xyz / pos_clip.w;
    if frustumCulling && any(abs(ndc.xy) > vec2<f32>(1.2)) {
        return;
    }

    let rot = vec4<f32>(unpack2x16float(g.rot[0]), unpack2x16float(g.rot[1]));
    let log_scale = vec3<f32>(unpack2x16float(g.scale[0]), unpack2x16float(g.scale[1]).x);
    let scale = exp(log_scale) * settings.gaussian_scaling;

    let cov3d = compute_cov3d(rot, scale);
    let cov2d = compute_cov2d(pos_view.xyz, cov3d);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if det <= 0.0 {
        return;
    }
    let conic = vec3<f32>(cov2d.z, -cov2d.y, cov2d.x) / det;

    // 3 sigma of the larger eigenvalue
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    let size_ndc = 2.0 * radius / camera.viewport;

    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - cam_pos);
    let color = computeColorFromSH(dir, idx, u32(settings.sh_deg));
    let opacity = sigmoid(z_opacity.y);

    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    var splat: Splat;
    splat.center = ndc.xy;
    splat.size = pack2x16float(size_ndc);
    splat.conic = array<f32, 3>(conic.x, conic.y, conic.z);
    splat.color = array<u32, 2>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, opacity)));
    splats[store_idx] = splat;

    // ascending radix sort on ~depth = back to front (depth > 0, so float bits are monotonic)
    sort_depths[store_idx] = ~bitcast<u32>(pos_view.z);
    sort_indices[store_idx] = store_idx;

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    let keys_per_dispatch = sortWorkgroupSize * sortKeyPerThread;
    if (store_idx % keys_per_dispatch) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
