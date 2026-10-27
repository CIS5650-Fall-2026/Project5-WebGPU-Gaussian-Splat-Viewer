struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,      // rgb + opacity
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) center_px: vec2<f32>,
    @location(3) @interpolate(flat) size: vec2<f32>,
};

// must match RenderSettings in preprocess_common.wgsl
struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    frustum_culling: f32,
    key_capacity: f32,
    tight_bounds: f32,
    debug_view: f32,
    _pad1: f32,
    _pad2: f32,
}

const DEBUG_NONE: u32 = 0u;
const DEBUG_QUADS: u32 = 1u;        // opaque white quads
const DEBUG_QUAD_SIZE: u32 = 2u;    // quad width / height in NDC as red / green (x4 to be visible)
const DEBUG_QUAD_COLOR: u32 = 3u;   // opaque quads with the SH color
const DEBUG_PIXEL_OFFSET: u32 = 4u; // offset to the splat center in pixels as red / green

struct Splat {
    pos_ndc: vec2<f32>,     // center in NDC
    conic_xy: vec2<f32>,    // inverse 2D covariance (pixel space): a, b
    conic_z: f32,           // c
    size: u32,              // f16x2: quad half extent in NDC
    color_rg: u32,          // f16x2
    color_b_opacity: u32,   // f16x2
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> splats: array<Splat>;
@group(0) @binding(2)
var<storage, read> sort_indices: array<u32>;
@group(0) @binding(3)
var<uniform> render_settings: RenderSettings;

// triangle strip corners
const QUAD = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    // instances are drawn in sorted (back to front) order
    let splat = splats[sort_indices[instance_index]];
    let size = unpack2x16float(splat.size);
    let rg = unpack2x16float(splat.color_rg);
    let b_op = unpack2x16float(splat.color_b_opacity);

    var out: VertexOutput;
    out.position = vec4<f32>(splat.pos_ndc + QUAD[vertex_index] * size, 0.0, 1.0);
    out.color = vec4<f32>(rg, b_op);
    out.conic = vec3<f32>(splat.conic_xy, splat.conic_z);
    out.size = size;
    out.center_px = vec2<f32>(
        (splat.pos_ndc.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - splat.pos_ndc.y * 0.5) * camera.viewport.y
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // offset from the fragment (pixel space, y down) to the splat center
    let d = in.center_px - in.position.xy;

    switch u32(render_settings.debug_view) {
        case DEBUG_QUADS: {
            return vec4<f32>(1.0);
        }
        case DEBUG_QUAD_SIZE: {
            return vec4<f32>(8.0 * in.size, 0.0, 1.0);
        }
        case DEBUG_QUAD_COLOR: {
            return vec4<f32>(in.color.rgb, 1.0);
        }
        case DEBUG_PIXEL_OFFSET: {
            return vec4<f32>(d, 0.0, 1.0);
        }
        default: {}
    }

    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    if power > 0.0 {
        discard;
    }
    let alpha = min(0.99, in.color.a * exp(power));
    if alpha < 1.0 / 255.0 {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
