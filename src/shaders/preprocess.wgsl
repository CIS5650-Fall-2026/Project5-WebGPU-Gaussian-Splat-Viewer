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
    frustum_culling: f32, // culling toggle: 1.0 is on
    _pad: f32
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    // store information for 2D splat rendering
    pos: u32, // ndc
    size: u32, // quad
    conic_opacity: array<u32, 2>, // conic xyz and opacity
    color: array<u32, 2> // rgb
};

// group 0 and 1 bindings
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;
@group(1) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2) var<storage, read> sh_coefs: array<u32>;

// bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// helper to read f16 in buffer
fn read_hth(h: u32) -> f32 {
    let pair = unpack2x16float(sh_coefs[h / 2u]);
    return select(pair.x, pair.y, (h & 1u) == 1u);
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    // Access binded sh_coeff, see load.ts for how it is stored
    // each gaussian is stored in 16 coeffs x 3 channels.  this is 96 bytes (48 halves)
    let b = splat_idx * 48u + c_idx * 3u;
    return vec3<f32>(read_hth(b), read_hth(b + 1u), read_hth(b + 2u));
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
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // set up pipeline as described in instruction...
    let g = gaussians[idx];
    let a = unpack2x16float(g.pos_opacity[0]);
    let b = unpack2x16float(g.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);
    let pos_view = camera.view * pos;
    let pos_clip = camera.proj * pos_view;

    // culling business (using suggested 1.2x frustum)
    if (pos_clip.w <= 0.0 || pos_view.z < 0.2) {
        return; // behind camera
    }
    
    let ndc = pos_clip.xyz / pos_clip.w;

    // frustum testing
    if (render_settings.frustum_culling > 0.5) {
        if (any(abs(ndc.xy) > vec2(1.2)) || ndc.z > 1.0) {
            return;
        }
    }

    // Covariance computation before writing the splats***
    // unpack, rotate, scale (multiply by render_settings.gaussian_scaling)
    let opacity = 1.0 / (1.0 + exp(-b.y));

    // in rotation step, requires normalization
    let r0 = unpack2x16float(g.rot[0]);
    let r1 = unpack2x16float(g.rot[1]);
    // quaternion is stored as (w, x, y, z)
    let q = normalize(vec4<f32>(r0.x, r0.y, r1.x, r1.y));
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let s0 = unpack2x16float(g.scale[0]);
    let s1 = unpack2x16float(g.scale[1]);
    let scale = exp(vec3<f32>(s0.x, s0.y, s1.x)) * render_settings.gaussian_scaling;

    // 3D covariance is RSS^TR^T
    let R = mat3x3<f32>(
        vec3(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y)
        ),
        vec3(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x)
        ),
        vec3(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y)
        )
    );

    let S = mat3x3<f32>(
        vec3(scale.x, 0.0, 0.0),
        vec3(0.0, scale.y, 0.0),
        vec3(0.0, 0.0, scale.z)
    );

    let M = R * S;
    let cov_3d = M * transpose(M);

    // 2D covariance is JWsumW^TJ^T
    // the Jacobian is the projection's derivative
    let t = pos_view.xyz;
    let lim_x = 1.3 * 0.5 * camera.viewport.x / camera.focal.x;
    let lim_y = 1.3 * 0.5 * camera.viewport.y / camera.focal.y;
    let tx = clamp(t.x / t.z, -lim_x, lim_x) * t.z;
    let ty = clamp(t.y / t.z, -lim_y, lim_y) * t.z;

    let J = mat3x3<f32>(
        vec3(
            camera.focal.x / t.z, 0.0, 0.0
        ), 
        vec3(
            0.0, camera.focal.y / t.z, 0.0
        ),
        vec3(
            -camera.focal.x * tx / (t.z * t.z),
            -camera.focal.y * ty / (t.z * t.z),
            0.0
        )
    );

    // upper left 3x3 mat
    let W = mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );

    let cov_2d = J * W * cov_3d * transpose(W) * transpose(J);

    // adds 0.3 square pixels of variance in each screen direction
    // ensures every splat gets min size ~pixel
    let cov_a = cov_2d[0][0] + 0.3;
    let cov_b = cov_2d[0][1];
    let cov_c = cov_2d[1][1] + 0.3;

    // do conic inverse stuff (return if determinant is 0)
    let det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 0.0) {
        return;
    }

    let conic = vec3<f32>(cov_c, -cov_b, cov_a) / det;

    // find radius from largest eigenvalue (convert to ndc half size)
    let mid = 0.5 * (cov_a + cov_c);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // now pack opacity into splat
    splats[idx].conic_opacity[0] = pack2x16float(conic.xy);
    splats[idx].conic_opacity[1] = pack2x16float(vec2(conic.z, opacity));
    splats[idx].pos = pack2x16float(ndc.xy);
    splats[idx].size = pack2x16float(vec2(radius) * 2.0 / camera.viewport);

    // true color gaussians
    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos.xyz - cam_pos);
    let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));
    splats[idx].color[0] = pack2x16float(color.rg);
    splats[idx].color[1] = pack2x16float(vec2(color.b, 0.0));

    let key = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[key] = idx;
    // note: as z increases, this means getting further away
    // so radix sort (ascending) will put the back splats first (since we need back to front)
    sort_depths[key] = bitcast<u32>(100.0 - pos_view.z);

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (key % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
    // after this, gaussian will read the splats using the sorted indices
}