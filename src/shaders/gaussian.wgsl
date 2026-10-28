// This shader draws a set of splats, each defined by a center and an extent

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) conic: vec3<f32>,
    @location(2) center: vec2<f32>,
};

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

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let corners = array<vec2<f32>,4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let splat = splats[sorted_indices[instance_index]];
    let ndc = splat.center + splat.extent * corners[vertex_index % 4u];
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = splat.color;
    out.conic = splat.conic;
    out.center = splat.center;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let center_pixel = vec2<f32>(
        (in.center.x + 1.0) * 0.5 * camera.viewport.x,
        (1.0 - in.center.y) * 0.5 * camera.viewport.y,
    );
    let d = in.position.xy - center_pixel;
    let power = -0.5 * (
        in.conic.x * d.x * d.x
        + 2.0 * in.conic.y * d.x * d.y
        + in.conic.z * d.y * d.y
    );

    if (power > 0.0) {
        discard;
    }

    let alpha = min(in.color.a * exp(power), 1.0);
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    return vec4<f32>(in.color.rgb, alpha);
}
