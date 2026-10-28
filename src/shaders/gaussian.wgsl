struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) @interpolate(flat) color: vec3<f32>,
    @location(1) @interpolate(flat) conic_opacity: vec4<f32>,
    @location(2) @interpolate(flat) center: vec2<f32>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

// same as in the preprocess compute shader
struct Splat {
    //TODO: information defined in preprocess compute shader
    pos_size: array<u32,2>,
    conic_opacity: array<u32,2>,
    color: array<u32,2>,
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(0) @binding(2)
var<storage, read> sort_infos: SortInfos;

@group(1) @binding(0)
var<uniform> camera: CameraUniforms;

const corners = array<vec2<f32>, 6>(
    vec2<f32>(-1., -1.),
    vec2<f32>(1., -1.),
    vec2<f32>(-1., 1.),
    vec2<f32>(-1., 1.),
    vec2<f32>(1., -1.),
    vec2<f32>(1., 1.),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass
    var out: VertexOutput;

    // index back to front
    let splat = splats[sort_indices[sort_infos.keys_size - 1u - instance_index]];
    let pos = unpack2x16float(splat.pos_size[0]);
    let size = unpack2x16float(splat.pos_size[1]);
    let color_a = unpack2x16float(splat.color[0]);
    let color_b = unpack2x16float(splat.color[1]);

    out.position = vec4<f32>(pos + corners[vertex_index] * size, 0., 1.);
    out.color = vec3<f32>(color_a.x, color_a.y, color_b.x);

    let conic_a = unpack2x16float(splat.conic_opacity[0]);
    let conic_b = unpack2x16float(splat.conic_opacity[1]);
    out.conic_opacity = vec4<f32>(conic_a.x, conic_a.y, conic_b.x, conic_b.y);

    // ndc to framebuffer pixels. y flipped => framebuffer y points down
    out.center = vec2<f32>(pos.x + 1., 1. - pos.y) * 0.5 * camera.viewport;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // how much of each pixel is covered by this splat?
    let d = in.position.xy - in.center;
    let conic = in.conic_opacity.xyz;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.) {
        discard;
    }

    let alpha = min(0.99, in.conic_opacity.w * exp(power));
    if (alpha < 1. / 255.) {
        discard;
    }
    return vec4<f32>(in.color, alpha);
}
