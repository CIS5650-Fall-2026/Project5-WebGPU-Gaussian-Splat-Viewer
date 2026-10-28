struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_offset: vec2<f32>,
    @location(1) @interpolate(flat) conic_opacity: vec4<f32>,
    @location(2) @interpolate(flat) color: vec3<f32>,
};

struct Splat {
    // Must match preprocess.wgsl (32 bytes).
    center: vec2<f32>,
    size: u32,
    color_rg: u32,
    color_b_opacity: u32,
    conic: array<f32,3>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sorted_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let splat = splats[sorted_indices[instance_idx]];
    // Four corners form a triangle strip without a vertex buffer.
    let corner = vec2<f32>(f32(vertex_idx & 1u), f32(vertex_idx >> 1u)) * 2.0 - 1.0;
    let offset_ndc = corner * unpack2x16float(splat.size);
    let color_rg = unpack2x16float(splat.color_rg);
    let color_b_opacity = unpack2x16float(splat.color_b_opacity);
    var out: VertexOutput;
    out.position = vec4<f32>(splat.center + offset_ndc, 0.0, 1.0);
    // Center minus fragment in framebuffer pixels: reverse X, since NDC Y
    // already runs in the opposite direction from framebuffer Y.
    out.pixel_offset = vec2<f32>(-offset_ndc.x, offset_ndc.y) * camera.viewport * 0.5;
    out.conic_opacity = vec4<f32>(splat.conic[0], splat.conic[1], splat.conic[2], color_b_opacity.y);
    out.color = vec3<f32>(color_rg, color_b_opacity.x);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.pixel_offset;
    let conic = in.conic_opacity.xyz;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y)
                - conic.y * d.x * d.y;
    if (power > 0.0) {
        discard;
    }
    let alpha = min(0.99, in.conic_opacity.w * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    return vec4<f32>(in.color, alpha);
}
