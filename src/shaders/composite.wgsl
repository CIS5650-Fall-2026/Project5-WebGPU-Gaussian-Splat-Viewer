struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct Splat {
    pos_radius: array<u32,2>,
    conic_opacity: array<u32,2>,
    color: array<u32,2>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<storage, read> sort_infos: SortInfos;
@group(0) @binding(3)
var<storage, read> sort_depths: array<u32>;
@group(0) @binding(4)
var<storage, read> sort_indices: array<u32>;
@group(0) @binding(5)
var out_image: texture_storage_2d<rgba8unorm, write>;

var<workgroup> range: vec2<u32>;
var<workgroup> batch_center: array<vec2<f32>, 256>;
var<workgroup> batch_conic_opacity: array<vec4<f32>, 256>;
var<workgroup> batch_color: array<vec3<f32>, 256>;

fn tile_bits(tiles: vec2<u32>) -> u32 {
    return 32u - countLeadingZeros(max(tiles.x * tiles.y - 1u, 1u));
}

fn first_key_of_tile(tile: u32, depth_bits: u32, n: u32) -> u32 {
    var lo = 0u;
    var hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        if (sort_depths[mid] >> depth_bits) < tile {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(16, 16, 1)
fn composite(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(global_invocation_id) gid: vec3<u32>) {
    let tiles = vec2<u32>(ceil(camera.viewport / 16.));
    let depth_bits = 32u - tile_bits(tiles);
    if lid == 0u {
        let n = min(sort_infos.keys_size, arrayLength(&sort_indices));
        let tile = wid.y * tiles.x + wid.x;
        range = vec2<u32>(first_key_of_tile(tile, depth_bits, n), first_key_of_tile(tile + 1u, depth_bits, n));
    }
    let r = workgroupUniformLoad(&range);

    let pixel = vec2<f32>(gid.xy) + 0.5;
    var color = vec3<f32>(0.);
    var transmittance = 1.;
    for (var start = r.x; start < r.y; start += 256u) {
        let i = start + lid;
        if i < r.y {
            let splat = splats[sort_indices[i]];
            let pos = unpack2x16float(splat.pos_radius[0]);
            let conic_xy = unpack2x16float(splat.conic_opacity[0]);
            let conic_z_opacity = unpack2x16float(splat.conic_opacity[1]);
            let color_rg = unpack2x16float(splat.color[0]);
            let color_b = unpack2x16float(splat.color[1]);
            batch_center[lid] = (pos * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport;
            batch_conic_opacity[lid] = vec4<f32>(conic_xy, conic_z_opacity);
            batch_color[lid] = vec3<f32>(color_rg, color_b.x);
        }
        workgroupBarrier();
        for (var j = 0u; j < min(256u, r.y - start) && transmittance >= 1e-4; j++) {
            let d = pixel - batch_center[j];
            let co = batch_conic_opacity[j];
            let power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            let alpha = min(0.99, co.w * exp(power));
            if power > 0. || alpha < 1. / 255. {
                continue;
            }
            color += batch_color[j] * alpha * transmittance;
            transmittance *= 1. - alpha;
        }
        workgroupBarrier();
    }
    if all(gid.xy < vec2<u32>(camera.viewport)) {
        textureStore(out_image, gid.xy, vec4<f32>(color, 1.));
    }
}
