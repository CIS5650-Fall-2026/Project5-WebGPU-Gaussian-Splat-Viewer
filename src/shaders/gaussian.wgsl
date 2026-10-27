const QUAD_OFFSETS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);




struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    NDC: vec2<f32>,
    size: vec2<f32>,
    color: vec4<f32>

};

@group(0) @binding(0) 
var<storage,read> splat: array<Splat>;
@group(0) @binding(1)
var<storage,read> sort_indices: array<u32>;


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

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}