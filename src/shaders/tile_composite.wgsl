const TILE_SIZE: u32 = 16u;
const MAX_TILE_COUNT: u32 = 1u << 15u;
// Must match the tile key layout in both preprocessing shaders
const TILE_ID_SHIFT: u32 = 17u;
const DEPTH_KEY_SHIFT: u32 = 32u - TILE_ID_SHIFT;
const SPLAT_BATCH_SIZE: u32 = 256u;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

struct Splat {
    center: vec2<f32>,
    conic: vec3<f32>,
    color: vec4<f32>,
    radius: f32,
    extent: vec2<f32>,
};

struct CompositeSplat {
    conic: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    center_pixel: vec2<f32>,
};

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

@group(0) @binding(0)
var<storage, read> sorted_keys: array<u32>;
@group(0) @binding(1)
var<storage, read> sorted_splat_indices: array<u32>;
@group(0) @binding(2)
var<storage, read> splats: array<Splat>;
@group(0) @binding(3)
var<uniform> camera: CameraUniforms;
@group(0) @binding(4)
var<storage, read_write> sort_infos: SortInfos;
@group(0) @binding(5)
var output_texture: texture_storage_2d<rgba8unorm, write>;

var<workgroup> tile_start: u32;
var<workgroup> tile_end: u32;
var<workgroup> shared_splats: array<CompositeSplat, SPLAT_BATCH_SIZE>;

fn first_key_at_least(target_key: u32, length: u32) -> u32 {
    var low = 0u;
    var high = length;
    loop {
        if (low >= high) {
            break;
        }
        let mid = low + (high - low) / 2u;
        if (sorted_keys[mid] < target_key) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return low;
}

// Composite splats in a tile into the output texture
@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn composite(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let pixel = vec2<u32>(
        workgroup_id.x * TILE_SIZE + local_id.x,
        workgroup_id.y * TILE_SIZE + local_id.y,
    );
    let tile_cols = u32(ceil(camera.viewport.x / f32(TILE_SIZE)));
    let tile_id = workgroup_id.y * tile_cols + workgroup_id.x;
    let pair_count = atomicLoad(&sort_infos.keys_size);
    if (local_id.x == 0u && local_id.y == 0u) {
        let tile_key = tile_id << TILE_ID_SHIFT;
        tile_start = first_key_at_least(tile_key, pair_count);
        if (tile_id + 1u == MAX_TILE_COUNT) {
            tile_end = pair_count;
        } else {
            tile_end = first_key_at_least(tile_key + (1u << TILE_ID_SHIFT), pair_count);
        }
    }
    workgroupBarrier();

    let range_start = workgroupUniformLoad(&tile_start);
    let range_end = workgroupUniformLoad(&tile_end);
    let active_pixel = !(
        f32(pixel.x) >= camera.viewport.x ||
        f32(pixel.y) >= camera.viewport.y
    );

    let pixel_center = vec2<f32>(pixel) + 0.5;
    let workgroup_thread = local_id.y * TILE_SIZE + local_id.x;
    var accumulated = vec3<f32>(0.0);
    var transmittance = 1.0;

    // Process splats in batches to reduce shared memory usage
    var batch_start = range_start;
    loop {
        if (batch_start >= range_end) {
            break;
        }
        let batch_count = min(SPLAT_BATCH_SIZE, range_end - batch_start);
        for (
            var load_index = workgroup_thread;
            load_index < batch_count;
            load_index += TILE_SIZE * TILE_SIZE
        ) {
            let splat = splats[sorted_splat_indices[batch_start + load_index]];
            shared_splats[load_index].conic = splat.conic;
            shared_splats[load_index].radius = splat.radius;
            shared_splats[load_index].color = splat.color;
            shared_splats[load_index].center_pixel = vec2<f32>(
                (splat.center.x + 1.0) * 0.5 * camera.viewport.x,
                (1.0 - splat.center.y) * 0.5 * camera.viewport.y,
            );
        }
        workgroupBarrier();

        for (var i = 0u; i < batch_count && active_pixel && transmittance > 1.0 / 255.0; i += 1u) {
            let splat = shared_splats[i];
            let d = pixel_center - splat.center_pixel;
            if (any(abs(d) > vec2<f32>(splat.radius))) {
                continue;
            }
            let power = -0.5 * (
                splat.conic.x * d.x * d.x
                + 2.0 * splat.conic.y * d.x * d.y
                + splat.conic.z * d.y * d.y
            );
            if (power > 0.0) {
                continue;
            }

            let alpha = min(splat.color.a * exp(power), 1.0);
            if (alpha >= 1.0 / 255.0) {
                accumulated += transmittance * alpha * splat.color.rgb;
                transmittance *= 1.0 - alpha;
            }
        }
        workgroupBarrier();
        batch_start += batch_count;
    }

    if (active_pixel) {
        textureStore(output_texture, pixel, vec4<f32>(accumulated, 1.0 - transmittance));
    }
}
