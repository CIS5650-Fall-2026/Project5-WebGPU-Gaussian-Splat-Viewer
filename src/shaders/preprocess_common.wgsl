// Shared declarations for the preprocess compute shaders.
// The final preprocess module is assembled on the host as:
//   [enable f16;] + preprocess_common.wgsl + (preprocess.wgsl | preprocess_f16.wgsl) + preprocess_entries.wgsl

override workgroupSize: u32;
// keys handled by one workgroup of the radix sort (histogram_wg_size * rs_histogram_block_rows)
override sortKeysPerDispatch: u32;
// tile edge length (pixels) used by the tile based renderer
override tileSize: u32 = 16u;

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
    frustum_culling: f32,   // > 0.5 => cull gaussians outside 1.2x the view frustum
    key_capacity: f32,      // tile renderer: number of sort keys that fit into the sort buffers
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

// 2D splat data consumed by the rasterizers (32 bytes).
// Position and conic stay in f32: the conic of a large splat is ~1e-6 px^-2, which would be a
// denormal in f16, and f16 NDC positions cause visible sub-pixel jitter on large screens.
struct Splat {
    pos_ndc: vec2<f32>,     // center in NDC
    conic_xy: vec2<f32>,    // inverse 2D covariance (pixel space): a, b
    conic_z: f32,           // c
    size: u32,              // f16x2: quad half extent in NDC
    color_rg: u32,          // f16x2
    color_b_opacity: u32,   // f16x2
};

// Result of projecting a single 3D gaussian (produced by the f32 or f16 math backend).
struct Projected {
    visible: bool,
    ndc: vec2<f32>,
    view_z: f32,
    radius: f32,            // pixels
    conic: vec3<f32>,
    color: vec3<f32>,
    opacity: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_buffer: array<u32>;   // 48 f16 (16 coefs x RGB) per gaussian
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
// tile renderer only (kept in one buffer to stay within 8 storage buffers per stage)
struct TileCounters {
    splat_count: atomic<u32>,
    // indirect dispatch of the tile range identification
    range_dispatch_x: u32,
    range_dispatch_y: u32,
    range_dispatch_z: u32,
    requested_keys: u32,    // keys before clamping to the sort buffer capacity
}
@group(2) @binding(4)
var<storage, read_write> tile_counters: TileCounters;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn ndc_to_pixel(ndc: vec2<f32>) -> vec2<f32> {
    // WebGPU framebuffer y points down, NDC y points up
    return vec2<f32>((ndc.x * 0.5 + 0.5) * camera.viewport.x, (0.5 - ndc.y * 0.5) * camera.viewport.y);
}

fn make_splat(p: Projected) -> Splat {
    var s: Splat;
    s.pos_ndc = p.ndc;
    s.conic_xy = p.conic.xy;
    s.conic_z = p.conic.z;
    s.size = pack2x16float(vec2<f32>(p.radius) * 2.0 / camera.viewport);
    s.color_rg = pack2x16float(p.color.rg);
    s.color_b_opacity = pack2x16float(vec2<f32>(p.color.b, p.opacity));
    return s;
}
