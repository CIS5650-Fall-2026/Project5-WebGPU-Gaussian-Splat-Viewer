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
    pos_size: array<u32,2>,
    conic_opacity: array<u32,2>,
    color: array<u32,2>,
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
var<storage, read> sh_coefs : array<u32>;

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
    // 16 coefs * 3 channels * 2 bytes = 96 bytes = 24 u32s per gaussian
    // because its packed as u32, dividing by 2 gets the word index
    let word = splat_idx * 24u + (c_idx * 3u) / 2u;
    let first = unpack2x16float(sh_coefs[word]);
    let second = unpack2x16float(sh_coefs[word + 1u]);

    // even = 1 | 2 3, odd = 1 2 | 3
    if (c_idx % 2u == 0u) {
        return vec3<f32>(first.x, first.y, second.x);
    }
    return vec3<f32>(first.y, second.x, second.y);
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
    // each thread processes one gaussian
    let idx = gid.x;
    // cull threads outside of the view frustum
    if (idx < arrayLength(&gaussians)) {
        let vertex = gaussians[idx];
        let a = unpack2x16float(vertex.pos_opacity[0]);  // unpack from the buffers
        let b = unpack2x16float(vertex.pos_opacity[1]);
        let pos_world = vec4<f32>(a.x, a.y, b.x, 1.);

        let pos_view = camera.view * pos_world;
        let pos_clip = camera.proj * pos_view;
        let pos_ndc = pos_clip.xyz / pos_clip.w;

        // inside the view frustum, perform all of the preprocessing and add to the sort buffer
        if (abs(pos_ndc.x) <= 1.2 && abs(pos_ndc.y) <= 1.2 &&
            pos_ndc.z >= 0. && pos_ndc.z <= 1.) {
            // 3d covariance from rotation and scaling
            let rot_a = unpack2x16float(vertex.rot[0]);
            let rot_b = unpack2x16float(vertex.rot[1]);
            let q = normalize(vec4<f32>(rot_a.x, rot_a.y, rot_b.x, rot_b.y));
            let r = q.x;
            let x = q.y;
            let y = q.z;
            let z = q.w;

            let R = mat3x3<f32>(
                1. - 2. * (y * y + z * z), 2. * (x * y + r * z), 2. * (x * z - r * y),
                2. * (x * y - r * z), 1. - 2. * (x * x + z * z), 2. * (y * z + r * x),
                2. * (x * z + r * y), 2. * (y * z - r * x), 1. - 2. * (x * x + y * y)
            );

            let scale_a = unpack2x16float(vertex.scale[0]);
            let scale_b = unpack2x16float(vertex.scale[1]);
            let s = render_settings.gaussian_scaling * exp(vec3<f32>(scale_a.x, scale_a.y, scale_b.x));
            let S = mat3x3<f32>(
                s.x, 0., 0.,
                0., s.y, 0.,
                0., 0., s.z
            );

            let cov3d = R * S * transpose(S) * transpose(R);

            // from this, calculate the 2d covariance
            let tan_fov = 0.5 * camera.viewport / camera.focal;
            let lim = 1.3 * tan_fov;
            let t = vec3<f32>(
                clamp(pos_view.x / pos_view.z, -lim.x, lim.x) * pos_view.z,
                clamp(pos_view.y / pos_view.z, -lim.y, lim.y) * pos_view.z,
                pos_view.z
            );

            // jacobian of perspective projection, to keep the transformation linear
            let J = mat3x3<f32>(
                camera.focal.x / t.z, 0., 0.,
                0., camera.focal.y / t.z, 0.,
                -camera.focal.x * t.x / (t.z * t.z), -camera.focal.y * t.y / (t.z * t.z), 0.
            );

            // view matrix without translation
            let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
            let cov2d = J * W * cov3d * transpose(W) * transpose(J);

            let cov_a = cov2d[0][0] + 0.3;  // numerical stability
            let cov_b = cov2d[0][1];
            let cov_c = cov2d[1][1] + 0.3;

            // conic and radius
            let det = cov_a * cov_c - cov_b * cov_b;
            if (det != 0.) {
                let conic_inv = vec3<f32>(cov_c, -cov_b, cov_a) / det;

                let mid = 0.5 * (cov_a + cov_c);
                let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
                let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
                let radius = ceil(3. * sqrt(max(lambda1, lambda2)));  // larger eigenvalue/side

                let size_ndc = 2. * radius / camera.viewport;  // quad size
                let opacity = 1. / (1. + exp(-b.y));

                // we have all of the required 2d attributes, now we can calculate the color using sh values
                let cam_pos = camera.view_inv[3].xyz;
                let dir = normalize(pos_world.xyz - cam_pos);
                let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));

                // add to the sort buffer and splats buffer
                let sort_idx = atomicAdd(&sort_infos.keys_size, 1u);
                splats[sort_idx].pos_size[0] = pack2x16float(pos_ndc.xy);
                splats[sort_idx].pos_size[1] = pack2x16float(size_ndc);
                splats[sort_idx].conic_opacity[0] = pack2x16float(conic_inv.xy);
                splats[sort_idx].conic_opacity[1] = pack2x16float(vec2<f32>(conic_inv.z, opacity));
                splats[sort_idx].color[0] = pack2x16float(color.rg);
                splats[sort_idx].color[1] = pack2x16float(vec2<f32>(color.b, 1.));

                // for eventual sorting by depth
                sort_depths[sort_idx] = bitcast<u32>(pos_view.z);
                sort_indices[sort_idx] = sort_idx;

                let keys_per_dispatch = workgroupSize * sortKeyPerThread;
                // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
                if (sort_idx % keys_per_dispatch == 0u) {
                    atomicAdd(&sort_dispatch.dispatch_x, 1u);
                }
            }
        }
    }
}