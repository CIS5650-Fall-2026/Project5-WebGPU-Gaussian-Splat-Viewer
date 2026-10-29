struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> splats: array<Splat>;

@group(1) @binding(1)
var<storage, read> sort_indices: array<u32>;


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) center_px: vec2<f32>,
    @location(1) conic: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct Splat {
    center: vec2<f32>,
    half_size: vec2<f32>,
    conic: vec4<f32>,
    color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) splat_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let splat = splats[sort_indices[splat_idx]];

    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    let pos = splat.center + corners[vertex_idx] * splat.half_size;

    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.center_px = vec2<f32>(
        (splat.center.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - splat.center.y * 0.5) * camera.viewport.y
    );
    out.conic = splat.conic;
    out.color = splat.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = vec2<f32>(in.center_px.x - in.position.x,
                      in.position.y - in.center_px.y);
    let power = -0.5 * (in.conic.x * d.x * d.x
                      + 2.0 * in.conic.y * d.x * d.y
                      + in.conic.z * d.y * d.y);
    if power > 0.0 {
        discard;
    }

    let alpha = min(0.99, in.conic.w * exp(power));
    if alpha < 1.0 / 255.0 {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}
