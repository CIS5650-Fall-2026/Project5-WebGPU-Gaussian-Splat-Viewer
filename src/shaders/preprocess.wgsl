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
    position: vec4<f32>,
    conic_opacity: vec4<f32>,
    quad_size: vec4<f32>,
    color: vec4<f32>,
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
var<storage, read> sh_coeffs: array<u32>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

fn read_sh_f16(index: u32) -> f32 {
    let packed = unpack2x16float(sh_coeffs[index / 2u]);

    if (index % 2u == 0u) {
        return packed.x;
    }

    return packed.y;
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base = splat_idx * 48u + c_idx * 3u;

    return vec3<f32>(
        read_sh_f16(base),
        read_sh_f16(base + 1u),
        read_sh_f16(base + 2u)
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

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn rotationMatrixFromQuaternion(q_raw: vec4<f32>) -> mat3x3<f32> {
    let q = normalize(q_raw);

    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    return mat3x3<f32>(
        vec3<f32>(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y)
        ),
        vec3<f32>(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x)
        ),
        vec3<f32>(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y)
        )
    );
}

fn computeCov3D(scale: vec3<f32>, rotation: vec4<f32>) -> mat3x3<f32> {
    let R = rotationMatrixFromQuaternion(rotation);

    let S = mat3x3<f32>(
        vec3<f32>(scale.x, 0.0, 0.0),
        vec3<f32>(0.0, scale.y, 0.0),
        vec3<f32>(0.0, 0.0, scale.z)
    );

    let M = R * S;
    return M * transpose(M);
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
    let pos = vec4<f32>(a.x, a.y, b.x, 1.0);

    let clip_pos = camera.proj * camera.view * pos;

    // Simple view-frustum culling.
    // Use a slightly larger screen bound to keep Gaussians near the edges.
    if (clip_pos.w <= 0.0) {
        return;
    }

    let ndc = clip_pos.xyz / clip_pos.w;
    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || ndc.z < 0.0 || ndc.z > 1.0) {
        return;
    }

    // Unpack opacity.
    let opacity = sigmoid(b.y);

    // Unpack quaternion rotation.
    let rot0 = unpack2x16float(gaussian.rot[0]);
    let rot1 = unpack2x16float(gaussian.rot[1]);
    let rotation = vec4<f32>(rot0.x, rot0.y, rot1.x, rot1.y);

    // Scale is stored in log space.
    let scale0 = unpack2x16float(gaussian.scale[0]);
    let scale1 = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(scale0.x, scale0.y, scale1.x))
        * render_settings.gaussian_scaling;

    // Compute the 3D covariance in world space.
    let cov3D = computeCov3D(scale, rotation);

    // Gaussian center in view space.
    let view_pos4 = camera.view * pos;
    let view_pos = view_pos4.xyz;

    if (view_pos.z <= 0.0001) {
        return;
    }

    // Rotate the 3D covariance from world space into view space.
    let view_rotation = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );

    let cov_view = view_rotation * cov3D * transpose(view_rotation);

    // Jacobian rows for perspective projection.
    let inv_z = 1.0 / view_pos.z;
    let inv_z2 = inv_z * inv_z;

    let j0 = vec3<f32>(
        camera.focal.x * inv_z,
        0.0,
        -camera.focal.x * view_pos.x * inv_z2
    );

    let j1 = vec3<f32>(
        0.0,
        camera.focal.y * inv_z,
        -camera.focal.y * view_pos.y * inv_z2
    );

    // Project the 3D covariance into 2D pixel space.
    var cov_xx = dot(j0, cov_view * j0);
    let cov_xy = dot(j0, cov_view * j1);
    var cov_yy = dot(j1, cov_view * j1);

    cov_xx += 0.3;
    cov_yy += 0.3;

    // Invert the 2D covariance to get the conic matrix.
    let det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det <= 0.0) {
        return;
    }

    let inv_det = 1.0 / det;
    let conic = vec3<f32>(
        cov_yy * inv_det,
        -cov_xy * inv_det,
        cov_xx * inv_det
    );

    // Largest eigenvalue gives maximum screen-space extent.
    let mid = 0.5 * (cov_xx + cov_yy);
    let discriminant = max(0.1, mid * mid - det);
    let lambda1 = mid + sqrt(discriminant);
    let lambda2 = mid - sqrt(discriminant);
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Pixel radius -> NDC half-size.
    let quad_size = vec2<f32>(
        2.0 * radius / camera.viewport.x,
        2.0 * radius / camera.viewport.y
    );

    // Evaluate view-dependent color from spherical harmonics.
    let camera_pos = camera.view_inv[3].xyz;
    let view_dir = normalize(pos.xyz - camera_pos);
    let color = computeColorFromSH(view_dir, idx, u32(render_settings.sh_deg));

    // Compact only after every validity check has passed.
    let compact_idx = atomicAdd(&sort_infos.keys_size, 1u);

    splats[compact_idx].position = clip_pos;
    splats[compact_idx].conic_opacity = vec4<f32>(conic, opacity);
    splats[compact_idx].quad_size = vec4<f32>(quad_size, 0.0, 0.0);
    splats[compact_idx].color = vec4<f32>(color, 1.0);

    // Sort payload: index into the compacted splat buffer.
    sort_indices[compact_idx] = compact_idx;

    // Radix sort is ascending. NDC depth grows from near (0) to far (1),
    // so reverse it to obtain back-to-front order.
    let sort_depth = 1.0 - ndc.z;
    sort_depths[compact_idx] = bitcast<u32>(sort_depth);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (compact_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}