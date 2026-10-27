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
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec3<f32>,
    @location(1) conic_opacity: vec4<f32>,
    @location(2) center_px: vec2<f32>,
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    center: u32,
    size: u32,
    conic_xy: u32,
    conic_z_opacity: u32,
    color_rg: u32,
    color_b: u32,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass
    var out: VertexOutput;
    let splat = splats[sort_indices[instance_index]];
    let center = unpack2x16float(splat.center);
    let size = unpack2x16float(splat.size);
    var corners = array<vec2<f32>, 6>(
       vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
       vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );

    out.position = vec4<f32>(center + corners[vertex_index] * size, 0.0, 1.0);

    let color_b = unpack2x16float(splat.color_b);
    out.color = vec3<f32>(unpack2x16float(splat.color_rg), color_b.x);
    out.conic_opacity = vec4<f32>(unpack2x16float(splat.conic_xy), unpack2x16float(splat.conic_z_opacity));
    out.center_px = vec2<f32>(center.x * 0.5 + 0.5, 0.5 - center.y * 0.5) * camera.viewport;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.center_px;
    let conic = in.conic_opacity.xyz;
    // Gaussian falloff
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.0) {
        discard;
    }
    let alpha = min(0.99, in.conic_opacity.w * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    return vec4<f32>(in.color * alpha, alpha);
}