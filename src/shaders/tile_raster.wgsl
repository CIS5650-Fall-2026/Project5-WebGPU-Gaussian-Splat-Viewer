// Tile based rasterizer (extra credit), following the reference CUDA implementation:
//   1. identify_ranges: the sorted keys are [tile id | depth], find the [start, end) range of every tile
//   2. rasterize: one 16x16 workgroup per tile, one thread per pixel. Splats of the tile are loaded in
//      batches into workgroup memory and alpha composited front to back until the pixel saturates.

const TILE_SIZE: u32 = 16u;
const BATCH_SIZE: u32 = TILE_SIZE * TILE_SIZE;

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
    pos_ndc: vec2<f32>,
    conic_xy: vec2<f32>,
    conic_z: f32,
    size: u32,
    color_rg: u32,
    color_b_opacity: u32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> sort_infos: SortInfos;
@group(0) @binding(2)
var<storage, read> sorted_keys: array<u32>;
@group(0) @binding(3)
var<storage, read_write> tile_ranges: array<vec2<u32>>;
@group(0) @binding(4)
var<storage, read> splats: array<Splat>;
@group(0) @binding(5)
var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(6)
var output: texture_storage_2d<rgba8unorm, write>;

fn tile_grid() -> vec2<u32> {
    return (vec2<u32>(camera.viewport) + TILE_SIZE - 1u) / TILE_SIZE;
}

// must match tile_key_bits in preprocess_entries.wgsl
fn tile_key_bits(tiles: vec2<u32>) -> u32 {
    let num_tiles = tiles.x * tiles.y;
    if num_tiles <= 1u {
        return 1u;
    }
    return firstLeadingBit(num_tiles - 1u) + 1u;
}

@compute @workgroup_size(256)
fn identify_ranges(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = sort_infos.keys_size;
    let shift = 32u - tile_key_bits(tile_grid());
    let stride = nwg.x * 256u;
    for (var i = gid.x; i < n; i += stride) {
        let tile = sorted_keys[i] >> shift;
        if i == 0u || (sorted_keys[i - 1u] >> shift) != tile {
            tile_ranges[tile].x = i;
        }
        if i == n - 1u || (sorted_keys[i + 1u] >> shift) != tile {
            tile_ranges[tile].y = i + 1u;
        }
    }
}

var<workgroup> batch_center: array<vec2<f32>, BATCH_SIZE>;
var<workgroup> batch_conic_opacity: array<vec4<f32>, BATCH_SIZE>;
var<workgroup> batch_color: array<vec4<f32>, BATCH_SIZE>;
var<workgroup> tile_range: vec2<u32>;
var<workgroup> done_count: atomic<u32>;
var<workgroup> done_snapshot: u32;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let tiles = tile_grid();
    let pix = wid.xy * TILE_SIZE + lid.xy;
    let inside = all(pix < vec2<u32>(camera.viewport));
    let pixf = vec2<f32>(pix) + 0.5;   // pixel center, same convention as @builtin(position)

    if li == 0u {
        tile_range = tile_ranges[wid.y * tiles.x + wid.x];
        atomicStore(&done_count, 0u);
    }
    // uniform load: the loop below contains barriers, so its trip count must be workgroup uniform
    let range = workgroupUniformLoad(&tile_range);
    let total = range.y - range.x;
    let rounds = (total + BATCH_SIZE - 1u) / BATCH_SIZE;

    var done = !inside;
    if done {
        atomicAdd(&done_count, 1u);
    }

    var transmittance = 1.0;
    var color = vec3<f32>(0.0);
    for (var r = 0u; r < rounds; r++) {
        // stop early once every pixel of the tile is saturated
        if li == 0u {
            done_snapshot = atomicLoad(&done_count);
        }
        if workgroupUniformLoad(&done_snapshot) == BATCH_SIZE {
            break;
        }

        // cooperatively fetch the next batch of splats into workgroup memory
        let fetch = range.x + r * BATCH_SIZE + li;
        if fetch < range.y {
            let s = splats[sorted_indices[fetch]];
            let ndc = s.pos_ndc;
            batch_center[li] = vec2<f32>((ndc.x * 0.5 + 0.5) * camera.viewport.x, (0.5 - ndc.y * 0.5) * camera.viewport.y);
            let b_op = unpack2x16float(s.color_b_opacity);
            batch_conic_opacity[li] = vec4<f32>(s.conic_xy, s.conic_z, b_op.y);
            batch_color[li] = vec4<f32>(unpack2x16float(s.color_rg), b_op.x, 0.0);
        }
        workgroupBarrier();

        if !done {
            let count = min(BATCH_SIZE, total - r * BATCH_SIZE);
            for (var j = 0u; j < count; j++) {
                let d = batch_center[j] - pixf;
                let co = batch_conic_opacity[j];
                let power = -0.5 * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
                if power > 0.0 {
                    continue;
                }
                let alpha = min(0.99, co.w * exp(power));
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let next_t = transmittance * (1.0 - alpha);
                if next_t < 0.0001 {
                    done = true;
                    atomicAdd(&done_count, 1u);
                    break;
                }
                color += batch_color[j].rgb * alpha * transmittance;
                transmittance = next_t;
            }
        }
    }

    if inside {
        // black background
        textureStore(output, pix, vec4<f32>(color, 1.0));
    }
}
