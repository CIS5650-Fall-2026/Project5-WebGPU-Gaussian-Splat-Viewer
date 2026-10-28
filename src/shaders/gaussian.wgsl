const QUAD_OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);




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
    @location(0) color: vec4<f32>,
    @location(1) conic_opacity: vec4<f32>,
    @location(2) center_px: vec2<f32>
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    NDC: vec2<f32>,
    size: vec2<f32>,
    color: vec4<f32>,
    conic_opacity: vec4<f32>

};

@group(0) @binding(0) 
var<storage,read> splat: array<Splat>;
@group(0) @binding(1)
var<storage,read> sort_indices: array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;


@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    
    let instance = splat[sort_indices[in_instance_index]];
    let offset = QUAD_OFFSETS[in_vertex_index] * instance.size;
    out.position = vec4<f32>(instance.NDC.xy + offset, 0.0, 1.0);
    out.color = instance.color;
    out.conic_opacity = instance.conic_opacity;
    out.center_px = vec2<f32>(
        (instance.NDC.x * 0.5 + 0.5) * camera.viewport.x,
        (0.5 - instance.NDC.y * 0.5) * camera.viewport.y
    );

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.center_px;
    let conic = in.conic_opacity.xyz;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.0) {
        discard;
    }
    let alpha = min(0.99, in.conic_opacity.w * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb * alpha, alpha);
}