struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    pos: vec2<f32>,             // ndc center
    size: u32,                  // quad half size in ndc, 2 x f16
    opacity: f32,
    conic: array<u32,2>,        // 2d conic (a, b, c), f16
    color: array<u32,2>,        // rgb, f16
};

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct DispatchIndirect {
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct TileInfo {
    tiles_x: u32,
    tiles_y: u32,
    width: u32,
    height: u32,
    depth_bits: u32,            // low bits of a key, the tile index is above them
    capacity: u32,              // max number of keys
}

const TILE_SIZE = 16u;
// keys per workgroup of the radix sort (256 threads x 15 keys)
const SORT_BLOCK = 3840u;

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> info: TileInfo;
@group(0) @binding(2)
var<storage, read> splats: array<Splat>;
// visible gaussians from the preprocess pass
@group(0) @binding(3)
var<storage, read> visible_info: array<u32>;
@group(0) @binding(4)
var<storage, read> visible_ids: array<u32>;
@group(0) @binding(5)
var<storage, read> visible_depths: array<u32>;
// one key per (gaussian, tile) pair for the second sorter
@group(0) @binding(6)
var<storage, read_write> tile_infos: SortInfos;
@group(0) @binding(7)
var<storage, read_write> tile_keys: array<u32>;
@group(0) @binding(8)
var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(9)
var<storage, read_write> tile_dispatch: DispatchIndirect;
// [start, end) of every tile in the sorted keys
@group(0) @binding(10)
var<storage, read_write> ranges: array<vec2<u32>>;
@group(0) @binding(11)
var<storage, read_write> image: array<u32>;

fn splat_center(splat: Splat) -> vec2<f32> {
    // ndc y points up, pixel y points down
    return (splat.pos * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport;
}

// one thread per visible gaussian, writes a key for every tile the splat touches
@compute @workgroup_size(256)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= visible_info[0]) {
        return;
    }
    let idx = visible_ids[i];
    let splat = splats[idx];
    let center = splat_center(splat);
    // only pixels with alpha >= 1/255 are drawn: d^T conic d <= 2 ln(255 opacity)
    let power = 2.0 * log(255.0 * splat.opacity);
    if (power <= 0.0) {
        return;
    }
    // x and y extent of that ellipse, from the diagonal of the 2d covariance
    let conic = vec3<f32>(unpack2x16float(splat.conic[0]), unpack2x16float(splat.conic[1]).x);
    let cov_diag = vec2<f32>(conic.z, conic.x) / (conic.x * conic.z - conic.y * conic.y);
    let quad = unpack2x16float(splat.size) * camera.viewport * 0.5;
    let radius = min(sqrt(power * cov_diag), quad);
    let grid = vec2<i32>(i32(info.tiles_x), i32(info.tiles_y));
    let lo = clamp(vec2<i32>(floor((center - radius) / f32(TILE_SIZE))), vec2<i32>(0), grid);
    let hi = clamp(vec2<i32>(ceil((center + radius) / f32(TILE_SIZE))), vec2<i32>(0), grid);
    let count = u32((hi.x - lo.x) * (hi.y - lo.y));
    if (count == 0u) {
        return;
    }

    let base = atomicAdd(&tile_infos.keys_size, count);
    // view depth bits without the sign bit, near splats first
    let z_bits = 0xffffffffu - visible_depths[i];
    let depth = (z_bits << 1u) >> (32u - info.depth_bits);
    var k = base;
    for (var y = lo.y; y < hi.y; y++) {
        for (var x = lo.x; x < hi.x; x++) {
            // keys past the end of the buffer are dropped
            if (k < info.capacity) {
                tile_keys[k] = ((u32(y) * info.tiles_x + u32(x)) << info.depth_bits) | depth;
                tile_ids[k] = idx;
            }
            k++;
        }
    }
}

// one thread, clamps the key count and sizes the sort dispatch
@compute @workgroup_size(1)
fn finish() {
    let n = min(atomicLoad(&tile_infos.keys_size), info.capacity);
    atomicStore(&tile_infos.keys_size, n);
    let blocks = (n + SORT_BLOCK - 1u) / SORT_BLOCK;
    tile_infos.padded_size = blocks * SORT_BLOCK;
    tile_dispatch.dispatch_x = blocks;
}

// a tile's list starts where the tile index of the sorted keys changes
@compute @workgroup_size(256)
fn find_ranges(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = atomicLoad(&tile_infos.keys_size);
    for (var r = 0u; r < SORT_BLOCK / 256u; r++) {
        let i = wid.x * SORT_BLOCK + r * 256u + lid;
        if (i >= n) {
            return;
        }
        let tile = tile_keys[i] >> info.depth_bits;
        if (i == 0u || (tile_keys[i - 1u] >> info.depth_bits) != tile) {
            ranges[tile].x = i;
        }
        if (i == n - 1u || (tile_keys[i + 1u] >> info.depth_bits) != tile) {
            ranges[tile].y = i + 1u;
        }
    }
}

var<workgroup> tile_range: vec2<u32>;
var<workgroup> done_count: atomic<u32>;
var<workgroup> batch_center: array<vec2<f32>, 256>;
var<workgroup> batch_conic: array<vec4<f32>, 256>;     // conic + opacity
var<workgroup> batch_color: array<vec3<f32>, 256>;

// one workgroup per tile and one thread per pixel, splats are blended front to back
@compute @workgroup_size(16, 16)
fn render(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>,
          @builtin(local_invocation_index) li: u32) {
    let pixel = wid.xy * TILE_SIZE + lid.xy;
    let inside = pixel.x < info.width && pixel.y < info.height;
    if (li == 0u) {
        tile_range = ranges[wid.y * info.tiles_x + wid.x];
    }
    let range = workgroupUniformLoad(&tile_range);

    let p = vec2<f32>(pixel) + 0.5;
    var T = 1.0;
    var color = vec3<f32>(0.0);
    var done = !inside;
    if (done) {
        atomicAdd(&done_count, 1u);
    }
    for (var start = range.x; start < range.y; start += 256u) {
        // stop when every pixel of the tile is opaque
        if (workgroupUniformLoad(&done_count) == 256u) {
            break;
        }
        // the workgroup loads the next 256 splats together
        let j = start + li;
        if (j < range.y) {
            let splat = splats[tile_ids[j]];
            batch_center[li] = splat_center(splat);
            batch_conic[li] = vec4<f32>(unpack2x16float(splat.conic[0]), unpack2x16float(splat.conic[1]).x, splat.opacity);
            batch_color[li] = vec3<f32>(unpack2x16float(splat.color[0]), unpack2x16float(splat.color[1]).x);
        }
        workgroupBarrier();
        let count = min(256u, range.y - start);
        for (var k = 0u; k < count && !done; k++) {
            let d = p - batch_center[k];
            let conic = batch_conic[k];
            let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
            if (power > 0.0) {
                continue;
            }
            let alpha = min(0.99, conic.w * exp(power));
            if (alpha < 1.0 / 255.0) {
                continue;
            }
            let next_T = T * (1.0 - alpha);
            if (next_T < 0.0001) {
                done = true;
                atomicAdd(&done_count, 1u);
                break;
            }
            color += batch_color[k] * alpha * T;
            T = next_T;
        }
        workgroupBarrier();
    }
    if (inside) {
        image[pixel.y * info.width + pixel.x] = pack4x8unorm(vec4<f32>(color, 1.0));
    }
}

// fullscreen triangle that copies the image to the canvas
@vertex
fn blit_vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn blit_fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<u32>(pos.xy);
    return unpack4x8unorm(image[p.y * info.width + p.x]);
}
