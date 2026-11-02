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
    @location(0) @interpolate(flat) center: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos_radius: array<u32,2>,
    conic_opacity: array<u32,2>,
    color: array<u32,2>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> splats: array<Splat>;
@group(1) @binding(1)
var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    let splat = splats[sort_indices[instance_index]];
    let pos = unpack2x16float(splat.pos_radius[0]);
    let radius = unpack2x16float(splat.pos_radius[1]);
    let conic_xy = unpack2x16float(splat.conic_opacity[0]);
    let conic_z_opacity = unpack2x16float(splat.conic_opacity[1]);
    let color_rg = unpack2x16float(splat.color[0]);
    let color_b = unpack2x16float(splat.color[1]);
    let corner = vec2<f32>(f32(vertex_index & 1u), f32(vertex_index >> 1u)) * 2. - 1.;

    var out: VertexOutput;
    out.position = vec4<f32>(pos + corner * radius, 0., 1.);
    out.center = (pos * vec2<f32>(0.5, -0.5) + 0.5) * camera.viewport;
    out.conic = vec3<f32>(conic_xy, conic_z_opacity.x);
    out.color = vec4<f32>(color_rg, color_b.x, conic_z_opacity.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.center;
    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;
    let alpha = min(0.99, in.color.a * exp(power));
    if power > 0. || alpha < 1. / 255. {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
