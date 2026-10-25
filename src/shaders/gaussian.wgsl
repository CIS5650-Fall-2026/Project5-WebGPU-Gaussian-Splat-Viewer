alias SplatFloat = f32;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

struct Splat {
    center: vec4<f32>,
    half_size: vec2<f32>,
    radius: f32,
    conic: vec4<SplatFloat>,
    color: vec4<SplatFloat>,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let splat = splats[sort_indices[instance_idx]];
    // The four corners form two triangles in a strip.
    let corner = vec2<f32>(f32(vertex_idx & 1u), f32(vertex_idx >> 1u)) * 2. - 1.;
    var out: VertexOutput;
    out.position = splat.center + vec4<f32>(corner * splat.half_size, 0., 0.);
    out.offset = corner * splat.radius;
    out.conic = vec3<f32>(splat.conic.xyz);
    out.color = vec4<f32>(splat.color);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.offset;
    let power = -0.5 * (in.conic.x * d.x * d.x + 2. * in.conic.y * d.x * d.y + in.conic.z * d.y * d.y);
    let alpha = min(0.99, in.color.a * exp(power));
    if (power > 0. || alpha < 1. / 255.) {
        discard;
    }
    return vec4<f32>(in.color.rgb * alpha, alpha);
}
