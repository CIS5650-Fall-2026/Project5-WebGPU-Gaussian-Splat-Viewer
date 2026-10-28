struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) @interpolate(flat) center: vec2<f32>,
    @location(1) @interpolate(flat) conic: vec3<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos: vec3<f32>,
    radius: f32,
    conic: mat2x2<f32>,
    color: vec4<f32>
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> splats : array<Splat>;

@group(1) @binding(1)
var<storage,read> sorted_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let splat = splats[sorted_indices[in_instance_index]];
    let quad = array<vec2f, 6>(
        vec2f(-1, -1),
        vec2f(1, -1.),
        vec2f(-1, 1),
        vec2f(-1, 1),
        vec2f(1, -1),
        vec2f(1, 1),
    );

    // ndc to pixel coords
    out.center = vec2<f32>(
        (splat.pos.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - splat.pos.y * 0.5) * camera.viewport.y
    );
    out.conic = vec3<f32>(splat.conic[0][0], splat.conic[0][1], splat.conic[1][1]);
    out.color = splat.color;
    let offset = quad[in_vertex_index] * splat.radius;
    out.position = vec4<f32>(
    splat.pos.xy + offset,
    splat.pos.z,
    1
);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = in.position.xy - in.center;
    let power = -0.5 * (in.conic.x * dist.x * dist.x 
        + 2.0 * in.conic.y * dist.x * dist.y + in.conic.z * dist.y * dist.y);
    if (power < -10) {
        discard;
    }
    let alpha = min(0.99, in.color.a * exp(power));
    return vec4<f32>(in.color.rgb, alpha);
}