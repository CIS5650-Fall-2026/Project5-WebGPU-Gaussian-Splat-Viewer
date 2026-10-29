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
    center: vec2<f32>,
    quad: vec2<f32>,
    color: vec4<f32>,
    conic: vec3<f32>,
    opacity: f32
};

@group(0) @binding(0) 
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) 
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2) 
var<storage, read> spherical_harmonics: array<u32>; 
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(3) @binding(0) 
var<uniform> renderSettings: RenderSettings;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let sh_idx = splat_idx * 16u * 3u + c_idx * 3u; // 16 coefs, 3 channels
    let first = unpack2x16float(spherical_harmonics[sh_idx / 2u]); //32u -> 16u
    let second = unpack2x16float(spherical_harmonics[sh_idx / 2u + 1u]);

    var sh: vec3<f32>;
    if (c_idx % 2u == 0u) {
        sh = vec3<f32>(first.x, first.y, second.x);
    } else {
        sh = vec3<f32>(first.y, second.x, second.y);      
    }
    return sh;
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

// stretch and rotate into correct ellipsoid shape 
// returns covariance matrix of the gaussian
fn compute_covariance(g: Gaussian) -> mat3x3<f32> {
    // rot in quaternion
    let q_wx = unpack2x16float(g.rot[0]);
    let q_yz = unpack2x16float(g.rot[1]);
    let q = normalize(vec4<f32>(q_wx.x, q_wx.y, q_yz.x, q_yz.y)); // must be unit length to be a rotation
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    // scale 
    let scale_xy = unpack2x16float(g.scale[0]);
    let scale_z = unpack2x16float(g.scale[1]).x;
    let scale = exp(vec3<f32>(scale_xy.x, scale_xy.y, scale_z)) * renderSettings.gaussian_scaling;

    let R = mat3x3<f32>(
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z),      2. * (x * z + r * y),
        2. * (x * y + r * z),      1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y),      2. * (y * z + r * x),      1. - 2. * (x * x + y * y)
    );


    let S = mat3x3<f32>(
        scale.x, 0., 0.,
        0., scale.y, 0.,
        0., 0., scale.z
    );

    let M = S * R;
    return transpose(M) * M; // Rt * St * S * R
}

fn check_frustum_culling(clip_pos: vec4<f32>) -> bool {
    // behind the camera
    if (clip_pos.w <= 0.) { 
        return false; 
    }

    let ndc = clip_pos.xy / clip_pos.w;
    let edgeBounds = 1.2;
    if (abs(ndc.x) > edgeBounds || abs(ndc.y) > edgeBounds) { return false; }

    return true;
}

// t: center of gaussian
fn project_to_2d(t: vec3<f32>) -> mat3x3<f32> {
    // view matrix 
    // dont include translation 
    let W = transpose(mat3x3<f32>(
    camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));

    // projection (jacobian) matrix 
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let J = mat3x3<f32>(
    fx / t.z, 0.0,      -(fx * t.x) / (t.z * t.z),
    0.0,      fy / t.z, -(fy * t.y) / (t.z * t.z),
    0.0,      0.0,       0.0
    );

    return W * J;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let g = gaussians[idx];
    let xy = unpack2x16float(g.pos_opacity[0]);
    let za = unpack2x16float(g.pos_opacity[1]);
    let pos = vec4<f32>(xy.x, xy.y, za.x, 1.0);

    // world -> view -> clip
    let view_pos = camera.view * pos;
    let clip_pos = camera.proj * view_pos;

    // 0. check gaussian is within the view frustum
    if (!check_frustum_culling(clip_pos)) {
        return;
    }
    let ndc = clip_pos.xy / clip_pos.w;
    let ndc_z = clip_pos.z / clip_pos.w;
    if (ndc_z < 0.0 || ndc_z > 1.0) { return; } // near far plane 

    // 1. compute covariance (shape)
    let sigma = compute_covariance(gaussians[idx]);
    let WJ = project_to_2d(view_pos.xyz);

    let cov2d = transpose(WJ) * sigma * WJ;

    // a b 
    // b c 
    let a = cov2d[0][0] + 0.3;
    let b = cov2d[0][1];
    let c = cov2d[1][1] + 0.3;
    let det = (a * c - b * b);
    if (det <= 0.0) {
        return;
    }

    // conic (inverse of cov)
    // falloff 
    let conic = vec3<f32>(c, -b, a) / det;

    // radius (eigenvalues)
    // quad size 
    let mid = 0.5 * (a + c);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    let quad = vec2<f32>(radius) / camera.viewport * 2.0;

    let opacity = 1.0 / (1.0 + exp(-za.y));

    // sphereical harmonics
    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - cam_pos);
    let color = computeColorFromSH(dir, idx, u32(renderSettings.sh_deg));
    
    let key_idx = atomicAdd(&sort_infos.keys_size, 1u);
    // add to sorting buffer
    sort_indices[key_idx] = key_idx;    
    // add to splats buffer
    splats[key_idx] = Splat(ndc.xy, quad, vec4<f32>(color, 1.0), conic, opacity);
    // add to depth buffer
    sort_depths[key_idx] = bitcast<u32>(view_pos.z);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (key_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}