enable f16;

const SH_C0: f16 = 0.28209479177387814;
const SH_C1: f16 = 0.4886025119029199;
const SH_C2 = array<f16,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f16,7>(
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
    pos_opacity: vec4<f16>,
    rot: vec4<f16>,
    scale: vec4<f16>
};

struct Splat {
    pos: vec2<f16>,
    radius: vec2<f16>,
    conic_opacity: vec4<f16>,
    color: vec4<f16>,
};

struct Projected {
    visible: bool,
    ndc: vec2<f32>,
    depth: f32,
    radius: f32,
    conic: vec3<f32>,
    opacity: f32,
    color: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh: array<f16>;
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
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f16> {
    let i = splat_idx * 48u + c_idx * 3u;
    return vec3<f16>(sh[i], sh[i + 1u], sh[i + 2u]);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f16>, v_idx: u32, sh_deg: u32) -> vec3<f16> {
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

    return  max(vec3<f16>(0.), result);
}

fn project(idx: u32) -> Projected {
    var p: Projected;
    let gaussian = gaussians[idx];
    let pos = vec4<f32>(vec3<f32>(gaussian.pos_opacity.xyz), 1.);

    let view_pos = camera.view * pos;
    let clip_pos = camera.proj * view_pos;
    let ndc = clip_pos.xyz / clip_pos.w;
    p.visible = ndc.z >= 0. && ndc.z <= 1. && all(abs(ndc.xy) <= vec2<f32>(1.2));
    if !p.visible {
        return p;
    }

    let q = normalize(gaussian.rot);
    let r = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let R = mat3x3<f16>(
        1. - 2. * (y * y + z * z), 2. * (x * y + r * z), 2. * (x * z - r * y),
        2. * (x * y - r * z), 1. - 2. * (x * x + z * z), 2. * (y * z + r * x),
        2. * (x * z + r * y), 2. * (y * z - r * x), 1. - 2. * (x * x + y * y)
    );
    let s = exp(gaussian.scale.xyz) * f16(settings.gaussian_scaling);
    let S = mat3x3<f16>(s.x, 0., 0., 0., s.y, 0., 0., 0., s.z);
    let M = mat3x3<f32>(R * S);
    let cov3d = M * transpose(M);

    let t = view_pos.xyz;
    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0., 0.,
        0., camera.focal.y / t.z, 0.,
        -camera.focal.x * t.x / (t.z * t.z), -camera.focal.y * t.y / (t.z * t.z), 0.
    );
    let W = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * W;
    let cov = T * cov3d * transpose(T);
    let cov2d = vec3<f32>(cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda = mid + sqrt(max(0.1, mid * mid - det));

    p.ndc = ndc.xy;
    p.depth = view_pos.z;
    p.radius = ceil(3. * sqrt(lambda));
    p.conic = vec3<f32>(cov2d.z, -cov2d.y, cov2d.x) / det;
    p.opacity = f32(1. / (1. + exp(-gaussian.pos_opacity.w)));
    p.color = vec3<f32>(computeColorFromSH(vec3<f16>(normalize(pos.xyz - camera.view_inv[3].xyz)), idx, u32(settings.sh_deg)));
    return p;
}

fn pack_splat(p: Projected) -> Splat {
    return Splat(
        vec2<f16>(p.ndc),
        vec2<f16>(vec2<f32>(p.radius) * 2. / camera.viewport),
        vec4<f16>(vec4<f32>(p.conic, p.opacity)),
        vec4<f16>(vec4<f32>(p.color, 0.))
    );
}

fn tile_bits(tiles: vec2<u32>) -> u32 {
    return 32u - countLeadingZeros(max(tiles.x * tiles.y - 1u, 1u));
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }
    let p = project(idx);
    if !p.visible {
        return;
    }
    splats[idx] = pack_splat(p);

    let key_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    if key_idx % keys_per_dispatch == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
    sort_depths[key_idx] = ~bitcast<u32>(p.depth);
    sort_indices[key_idx] = idx;
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess_tile(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }
    let p = project(idx);
    if !p.visible {
        return;
    }
    splats[idx] = pack_splat(p);

    let tiles = vec2<u32>(ceil(camera.viewport / 16.));
    let center = (p.ndc * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport;
    let rect_min = vec2<u32>(clamp(floor((center - p.radius) / 16.), vec2<f32>(0.), vec2<f32>(tiles)));
    let rect_max = vec2<u32>(clamp(ceil((center + p.radius) / 16.), vec2<f32>(0.), vec2<f32>(tiles)));
    let count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    if count == 0u {
        return;
    }

    let capacity = arrayLength(&sort_indices);
    let base = min(atomicAdd(&sort_infos.keys_size, count), capacity);
    let end = min(base + count, capacity);
    let keys_per_dispatch = workgroupSize * sortKeyPerThread;
    atomicAdd(&sort_dispatch.dispatch_x, (end + keys_per_dispatch - 1u) / keys_per_dispatch - (base + keys_per_dispatch - 1u) / keys_per_dispatch);

    let depth_bits = 32u - tile_bits(tiles);
    let depth_key = bitcast<u32>(p.depth) >> tile_bits(tiles);
    var i = base;
    for (var y = rect_min.y; y < rect_max.y; y++) {
        for (var x = rect_min.x; x < rect_max.x; x++) {
            if i < end {
                sort_depths[i] = ((y * tiles.x + x) << depth_bits) | depth_key;
                sort_indices[i] = idx;
            }
            i++;
        }
    }
}
