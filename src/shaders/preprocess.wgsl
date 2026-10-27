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
    NDC: vec2<f32>,
    size: vec2<f32>
};



//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage,read_write> splat: array<Splat>;

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
    return vec3<f32>(0.0);
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
    if (idx >= arrayLength(&gaussians)){
        return;
    }


    let instance = gaussians[idx];
    let a = unpack2x16float(instance.pos_opacity[0]);
    let b = unpack2x16float(instance.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    let temp = camera.proj * camera.view * pos;

    //culling
    if (temp.w <= 0.0) {
        return;
    }
    let ndc = temp.xyz / temp.w;
    if (abs(ndc.x)>1.2 || abs(ndc.y)>1.2) {
        return;
    }

    let rot01 = unpack2x16float(instance.rot[0]);
    let rot23 = unpack2x16float(instance.rot[1]);
    var w = rot01.x;
    var x = rot01.y;
    var y = rot23.x;
    var z = rot23.y;
    let quat = normalize(vec4<f32>(x, y, z, w));
    w = quat.w;
    x = quat.x;
    y = quat.y;
    z = quat.z; 

    let scale01 = unpack2x16float(instance.scale[0]);
    let scale23 = unpack2x16float(instance.scale[1]);
    let scale = exp(vec3<f32>(scale01.x, scale01.y, scale23.x)) * settings.gaussian_scaling;

    let R = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)
    );

    let S = mat3x3<f32>(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );

    let M = R * S;
    let sigma = M * transpose(M);
    var t = (camera.view*pos).xyz;
    let tan_fov = 0.5 * camera.viewport / camera.focal;
    let lim = 1.3 * tan_fov;
    //clamp
    t.x = clamp(t.x/t.z, -lim.x, lim.x) * t.z;
    t.y = clamp(t.y/t.z, -lim.y, lim.y) * t.z;

    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let J = mat3x3<f32>(
        fx/t.z, 0.0, 0.0,
        0, fy/t.z, 0.0,
        -fx*t.x/(t.z*t.z), -fy*t.y/(t.z*t.z), 0.0
    );
    let view = camera.view;
    let W = mat3x3<f32>(
        view[0].xyz, view[1].xyz, view[2].xyz
    );
    let T = J * W;
    let cov = T * sigma * transpose(T);
    let cova = cov[0][0] + 0.3;
    let covb = cov[0][1];
    let covc = cov[1][1] + 0.3;
    let det = cova * covc - covb * covb;
    if (det <= 0.0) {
        return;
    }

    let conic = vec3<f32>(cova, -covb, covc) / det;
    let mid = 0.5 * (cova + covc);
    let lambda1 = mid + sqrt(max(0.1, mid*mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid*mid - det));
    let radius = ceil(sqrt(max(lambda1, lambda2)) * 3.0);
    let size = 2 * radius / camera.viewport;

    splat[idx].NDC = ndc.xy;
    splat[idx].size = size;

    let slot = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[slot] = idx;
    sort_depths[slot] = u32((ndc.z + 1.0) * 0.5 * 4294967295.0);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (slot % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}