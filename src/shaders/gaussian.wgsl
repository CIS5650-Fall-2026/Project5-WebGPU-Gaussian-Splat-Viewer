struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot: array<u32, 2>,
    scale: array<u32, 2>
};

struct Splat {
    center: vec4<f32>,
    color: vec4<f32>,
    half_size: vec2<f32>,
    radius: f32,
    conic_opacity: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;

@group(0) @binding(1)
var<storage, read> sorted_indices: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(linear) pixel_offset: vec2<f32>,
    @location(2) @interpolate(flat) conic_opacity: vec4<f32>
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let splat_idx = sorted_indices[instance_index];
    let center = splats[splat_idx].center;

    // Four corners for a triangle strip.
    let corner = vec2<f32>(
        f32(vertex_index & 1u) * 2.0 - 1.0,
        f32(vertex_index >> 1u) * 2.0 - 1.0
    );

    let offset = corner * splats[splat_idx].half_size;

    out.position = vec4<f32>(
        center.xy + offset * center.w,
        center.z,
        center.w
    );

    out.color = splats[splat_idx].color;
    out.pixel_offset = corner * splats[splat_idx].radius * vec2<f32>(1.0, -1.0);
    out.conic_opacity = splats[splat_idx].conic_opacity;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.pixel_offset;
    let c = in.conic_opacity;
    let power = -0.5 * (c.x * d.x * d.x + 2.0 * c.y * d.x * d.y + c.z * d.y * d.y);
    if (power > 0.0) {
        discard;
    }
    let alpha = min(0.99, c.w * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb * alpha, alpha);
}