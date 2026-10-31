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
    pos: vec2<f32>, // NDC center
    size: vec2<f32>, // NDC half width
    color: vec4<f32>,
    conic_opacity: vec4<f32>
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
var<storage, read> sh: array<u32>;
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
    let h = splat_idx * 48 + c_idx * 3; // half index
    let w = h / 2;
    
    let a = unpack2x16float(sh[w]);
    let b = unpack2x16float(sh[w+1]);

    if (h%2 == 0) {
        return vec3<f32>(a.x, a.y, b.x);
    } else {
        return vec3<f32>(a.y, b.x, b.y);
    }
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

fn diag(s: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        s.x, 0.f, 0.f,
        0.f, s.y, 0.f,
        0.f, 0.f, s.z,
    );
}

// Quaternion -> rotation matrix
fn quatToRot(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    return transpose(mat3x3<f32>(
        1.f - 2.f * (y*y + z*z),       2.f * (x*y - w*z),        2.f * (x*z + w*y),
              2.f * (x*y + w*z), 1.f - 2.f * (x*x + z*z),        2.f * (y*z - w*x),
              2.f * (x*z - w*y),       2.f * (y*z + w*x),  1.f - 2.f * (x*x + y*y),
    ));
}

// Get world space 3D covariance
fn computeCov3D(scale: vec3<f32>, R: mat3x3<f32>) -> mat3x3<f32> {
    let S = diag(render_settings.gaussian_scaling * scale);
    let M = R * S;
    return M * transpose(M);
}

// Get jacobia of perspective proj of point t
fn computeJacobian(t: vec3<f32>) -> mat3x3<f32> {
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    return transpose(mat3x3<f32>(
        fx/t.z, 0,      -fx*t.x / (t.z*t.z),
        0,      fy/t.z, -fy*t.y / (t.z*t.z),
        0,      0,       0,
    ));
}

// Get screenspace 2D covariance
fn computeCov2D(q: vec4<f32>, scale: vec3<f32>, t: vec3<f32>) -> mat3x3<f32> {
    // Quaternion -> rotatoin matrix
    let R = quatToRot(q);

    // Calculate covariance 3d
    let cov3d = computeCov3D(scale, R);

    // Calculate jacobian
    let J = computeJacobian(t);

    // Calculate covariance 2d
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    var cov2d = T * cov3d * transpose(T);

    // Low pass filter- keep det away from 0
    cov2d[0][0] += 0.3f;
    cov2d[1][1] += 0.3f;
    return cov2d;
}

// Get determinant of screen space covariance
fn computeDet(cov2d: mat3x3<f32>) -> f32 {
    let a = cov2d[0][0];
    let b = cov2d[0][1];
    let c = cov2d[1][1];
    return a*c - b*b;
}

// Get screen radius in pixels from cov2d eigne values
fn computeRadius(cov2d: mat3x3<f32>, det: f32) -> f32 {
    let a = cov2d[0][0];
    let c = cov2d[1][1];

    let mid = (a+c) / 2;
    let d1 = mid + sqrt(max(0.1, mid*mid - det));
    let d2 = mid - sqrt(max(0.1, mid*mid - det));

    return ceil(3 * sqrt(max(d1, d2)));
}

fn sigmoid(x: f32) -> f32 {
    return 1.f / (1.f + exp(-x));
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 

    // Unpack position data
    let vertex = gaussians[idx];
    let pos_xy = unpack2x16float(vertex.pos_opacity[0]);
    let pos_z_opacity = unpack2x16float(vertex.pos_opacity[1]);
    let worldPos = vec4<f32>(pos_xy.x, pos_xy.y, pos_z_opacity.x, 1.); // World space pos

    // ndc <- clip <- view <- world space 
    let viewPos: vec4<f32> = camera.view * worldPos;
    let clipPos: vec4<f32> =  camera.proj * viewPos;
    if (clipPos.w <= 0.f) {
        return;
    }
    let ndcPos: vec4<f32> = clipPos / clipPos.w;

    // Frustum culling
    if (abs(ndcPos.x) > 1.2f || abs(ndcPos.y) > 1.2f || 
        ndcPos.z > 1.f || ndcPos.z < 0.f) {
        return;
    }

    // Unwrap rotation/ scale
    let rot01 = unpack2x16float(vertex.rot[0]);
    let rot23 = unpack2x16float(vertex.rot[1]);
    let scale01 = unpack2x16float(vertex.scale[0]);
    let scale2N = unpack2x16float(vertex.scale[1]);

    let q = normalize(vec4<f32>(rot01.y, rot23.xy, rot01.x)); // x y z w
    let scale = exp(vec3<f32>(scale01.xy, scale2N.x));

    // Project to screen space covariance
    let cov2d = computeCov2D(q, scale, viewPos.xyz);

    // Cull degenerate ellipse
    let det = computeDet(cov2d);
    if (det == 0.f) {
        return;
    }

    let radius = computeRadius(cov2d, det); // in pixels
    let size = radius * 2 / camera.viewport;

    // Save depth/ sort index
    let slot = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[slot] = idx;
    sort_depths[slot] = ~bitcast<u32>(viewPos.z); // float->int, order preserving, then reverse order

    // Save 2D splats data
    splats[idx].pos = ndcPos.xy;
    splats[idx].size = size;

    let camPos = camera.view_inv[3].xyz;
    let dir = normalize(worldPos.xyz - camPos);
    let alpha = 1.f;
    splats[idx].color = vec4<f32>(computeColorFromSH(dir, idx, u32(render_settings.sh_deg)), alpha);

    let a = cov2d[0][0];
    let b = cov2d[0][1];
    let c = cov2d[1][1];
    let conic = vec3<f32>(c/det, -b/det, a/det);
    let opacity = sigmoid(pos_z_opacity.y);
    splats[idx].conic_opacity = vec4<f32>(conic.xyz, opacity);

    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (slot % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

}