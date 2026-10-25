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
    focal: vec2<f32>,
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
    ndc: vec2<f32>,
    size: vec2<f32>,
    conic: vec3<f32>,
    op: f32,
    color: vec3<f32>,
};

// read only
@group(0) @binding(0) 
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;

// gaussian input
@group(1) @binding(0) 
var<storage, read> gaussians: array<Gaussian>;
// splats output
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coefs: array<u32>;

// read_write
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// My own fcns for making the math a little easier
fn diag(v: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(v.x, 0.0, 0.0),
        vec3<f32>(0.0, v.y, 0.0),
        vec3<f32>(0.0, 0.0, v.z),
    );
}

// assumes normalized
fn q_to_mat(q: vec4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y + q.w * q.z), 2 * (q.x * q.z - q.w * q.y)),
        vec3<f32>(2 * (q.x * q.y - q.w * q.z), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z + q.w * q.x)),
        vec3<f32>(2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x), 1 - 2 * (q.x * q.x + q.y * q.y)),
    );
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let sh_idx = (splat_idx * 48 + 3 * c_idx) / 2u;

    let a = unpack2x16float(sh_coefs[sh_idx]);
    let b = unpack2x16float(sh_coefs[sh_idx + 1u]);

    if (c_idx % 2u == 1u) { return vec3f(a.y, b.x, b.y); }
    return vec3f(a.x, a.y, b.x);
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
    if (idx >= arrayLength(&gaussians)) { return; }

    // unpacking data
    let g = gaussians[idx];
    let xy = unpack2x16float(g.pos_opacity[0]);
    let zo = unpack2x16float(g.pos_opacity[1]);
    let pos = vec3f(xy.x, xy.y, zo.x);
    let rot_wx = unpack2x16float(g.rot[0]);
    let rot_yz = unpack2x16float(g.rot[1]);
    let scale_xy = unpack2x16float(g.scale[0]);
    let scale_zw = unpack2x16float(g.scale[1]);
    let raw_opacity = zo.y;

    // making data usable
    // splat opacity
    let opacity = 1.0 / (1.0 + exp(-raw_opacity));

    let rot_q = vec4<f32>(rot_wx.y, rot_yz.x, rot_yz.y, rot_wx.x);
    let scale = exp(vec3<f32>(scale_xy.x, scale_xy.y, scale_zw.x)) * settings.gaussian_scaling;

    let rMat = q_to_mat(normalize(rot_q));
    let sMat = diag(scale);
    let mMat = rMat * sMat;

    // camera and pos info
    let t = (camera.view * vec4f(pos, 1.0)).xyz;
    let W = mat3x3f(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let fx = camera.focal.x;
    let fy = camera.focal.y;

    let clip = camera.proj * vec4f(t, 1.0);
    // splat ndc
    let ndc = clip.xyz / clip.w;

    if (t.z <= 0.0 || ndc.z < 0 || ndc.z > 1 || abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2) { return; }

    // finding gaussian and splat info
    let cov = mMat * transpose(mMat);
    
    let jacobian = mat3x3f(
        vec3f(fx / t.z, 0.0, 0.0),
        vec3f(0.0, fy / t.z, 0.0),
        vec3f(-fx * t.x / (t.z * t.z), -fy * t.y / (t.z * t.z), 0.0),
    );
    let jw = jacobian * W;
    let cov2d = jw * cov * transpose(jw);

    let a = cov2d[0][0] + 0.3;
    let b = cov2d[0][1];
    let c = cov2d[1][1] + 0.3;

    // determinant
    let det = a * c - b * b;
    if (det == 0.0) { return; }

    // index for accessing splats
    let i = atomicAdd(&sort_infos.keys_size, 1u);

    let m = (a + c) / 2.0;
    // larger eigenvalue
    let lambMAX = m + sqrt(max(0.1, m * m - det));

    // radius (largest eigenvalue converted to pixels)
    let r = ceil(3 * sqrt(lambMAX));

    // splat size
    let ndcRadius = vec2f(
        (r * 2.0) / camera.viewport.x,
        (r * 2.0) / camera.viewport.y
    );

    // splat conic
    let conic = vec3f(c, -b, a) / det;

    let cam_pos = camera.view_inv[3].xyz;
    let viewDir = normalize(pos - cam_pos);
    // we use same index as the one we use to access gaussian data
    let color = computeColorFromSH(viewDir, idx, u32(settings.sh_deg));

    splats[i].ndc = ndc.xy;
    splats[i].size = ndcRadius;
    splats[i].op = opacity;
    splats[i].conic = conic;
    splats[i].color = color;

    sort_depths[i] = bitcast<u32>(100.0 - t.z);
    sort_indices[i] = i;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    if (i % keys_per_dispatch == 0u) { atomicAdd(&sort_dispatch.dispatch_x, 1u); }
}